# Standard library imports
import os
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
import threading

# Third-party imports
import torch
from torch.optim import Adam

# Determine the directory based on whether the environment is a notebook or a script
try:
    # This will raise an error in notebooks since __file__ doesn't exist
    directory = Path(__file__).resolve().parent
except NameError:
    # In Jupyter notebooks, the current working directory is used
    directory = Path().resolve()

sys.path.append(str(directory.parent))

# Go up 3 levels: py -> src -> active-learning
# Folder structure:
# <Parent folder> 
#               ├── active-learning
#               |   └── src
#               |       └── active_learning_HEO.py
#               ├── oxygen-carrier-discovery
#               └── metal

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
os.environ.setdefault('METAL_CHEMSYS_CACHE', os.path.join(project_root, '.cache', 'metal', 'chemsys'))

metal_path = os.path.join(project_root, 'metal')
pyscripts_path = os.path.join(project_root, 'oxygen-carrier-discovery','heoc','pyscripts')

sys.path.append(metal_path)
sys.path.append(pyscripts_path)

# Local imports
from al_config import Config, parse_args
from ANN_functions_new import WAE_modular, train_WAE, same_seeds, get_latents
from al_functions import *
from ensemble_mlp import DeepEnsemble, train_deep_ensemble
from mogp.ensemble_adapter import MOGPEnsembleAdapter, split_constant_columns
from mogp.delta_gp import (DeltaGPRegressor, DeltaGPEnsembleAdapter,
                           DeltaEnsembleAdapter, build_training_matrix,
                           train_delta_gp)
from mogp.models import ICMGPRegressor, fit_gp as fit_mogp
from al_util import (
    save_checkpoint, load_checkpoint, save_rng_state, load_rng_state,
    save_progress_state, should_skip_stage,
    setup_logging, validate_cycle_integrity,
    auto_detect_and_load_checkpoint, log_model_info,
    STAGE_TRAINING_AND_SAMPLING, STAGE_SAMPLE_DB,
    STAGE_SQS, STAGE_PHONON, STAGE_PHASE_STABILITY, STAGE_TRAINING_DB,
    STAGE_COMPARE_PREDICTIONS, log_wae_loss,
)
from al_viz import visualize_latent_space, plot_latent_evolution, analyze_latent_structure
from build_sample_database import build_sample_database
from build_sqs_databases import build_sqs_structures
from perform_phonon_calculations import perform_phonon_calculations
from slurm_phonon import submit_phonon_slurm_jobs
from update_training_database import update_training_database
from data_manager import DataManager

# External project imports
from metal.applications.crystals.stability_factors import update_phase_stability_dbs
from pymatgen.core import SETTINGS


args = parse_args()
config = Config(args)
config.validate()


def _shared_gibbs_scale(data_manager):
    """The single scale every Gibbs column was normalised by.

    The delta arms add the analytic reactor offset -- a physical energy in eV --
    to normalised AR values, so it has to be divided by the scale those values
    carry. That is only well defined when all Gibbs columns share one scale,
    which is what grouped z-score (landscape mode) provides.

    This raises rather than defaulting. The previous code asked for a 'std' key
    with a fallback of 1.0; under min-max normalisation that key does not exist,
    so the offset went through unscaled and FR came out ~148 eV wrong with no
    error and no warning.
    """
    stats = data_manager.norm_stats.get('gibbs_AR_0')
    if stats is None:
        raise KeyError(
            "no normalisation stats for 'gibbs_AR_0'; the delta arms need "
            "landscape-mode grouped z-score")
    if stats.get('type') != 'zscore' or 'std' not in stats:
        raise ValueError(
            f"delta arms need grouped z-scored Gibbs targets, got "
            f"{stats.get('type')!r}. A single additive offset is exact only "
            f"when every Gibbs column shares one scale.")
    return stats['std']


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up initial cycle and log path using config methods
    config.ensure_initial_cycle()
    config.setup_log_path(now)

    # Set up logging early so we can use it for all output
    logger = setup_logging(config.log_path)

    # Log initial configuration
    config.log_initial_info(logger, config.log_path)

    # Save configuration to file for reproducibility
    config_file = config.save_to_file(config.log_path)
    logger.info(f'Configuration saved to: {config_file}')

    # Set random seed for reproducibility. Without --seed this seeds torch only
    # and the run is not reproducible; see same_seeds.
    same_seeds(config.default_seed, full=config.deterministic)
    if config.deterministic:
        logger.info(f'Deterministic run: seed {config.default_seed} (torch, numpy, random)')
    else:
        logger.info(f'Non-deterministic run: torch seeded at {config.default_seed}, '
                    f'numpy unseeded. Pass --seed to make this run reproducible.')

    # Get properties to predict from config
    config.set_method_list()
    n_props = len(config.method_list)

    # Get WAE parameters from config
    params = config.get_wae_params()

    # Set up checkpoint directory using config method
    checkpoint_dir = config.get_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)

    wae_model = WAE_modular(params).to(config.device)

    # Log model architecture and parameters
    log_model_info(wae_model, params, logger, is_ensemble=False)

    # Auto-detect if we should resume from a checkpoint and determine starting cycle
    start_cycle, cycle, progress_state = auto_detect_and_load_checkpoint(
        config.log_path, checkpoint_dir, wae_model, config.device, logger, config.folder_name
    )

    # Resume the RNG streams where the last completed cycle left off. This must
    # come after same_seeds() (line ~107), which it intentionally overrides:
    # seeding gives a fresh stream, and only a resumed stream reproduces the
    # realisation an uninterrupted run would have produced.
    if start_cycle > 1:
        restored = load_rng_state(start_cycle - 1, checkpoint_dir, logger)
        if not restored and config.deterministic:
            logger.warning(
                'Resuming a seeded run without an RNG snapshot: results stay '
                'valid but will not match an uninterrupted run of the same '
                'seed. Note this if the run is used as a matched replicate.')

    # Validate cycle integrity before starting the loop
    validate_cycle_integrity(config.folder_name, logger)

    # ==================================================================================
    # MAIN ACTIVE LEARNING LOOP
    # ==================================================================================
    end_cycle = config.default_cycles

    for cycle in range(start_cycle, end_cycle):
        logger.info('=' * 80)
        logger.info(f'CYCLE {cycle}')
        logger.info('=' * 80)

        # Reduce WAE training epochs after first cycle
        if cycle > 1:
            params['num_epoch'] = config.subsequent_wae_epochs

        new_folder_name = f'{config.heoc_path}/{config.method}/cycle{cycle}'

        # Data Management
        data_manager = DataManager(config, logger)
        data_manager.load_data(config.method_list, config.db_name, config.elements)
        
        # Decide normalization based on models used
        # MOGP shares the ensemble's grouped z-score: the ICM ties all Gibbs
        # tasks through one coregionalization matrix, so they must sit on a
        # common scale. Per-column min-max would make the shared kernel
        # meaningless.
        #
        # The delta arms need the same shared scale, for a different reason.
        # They model the AR curve only and derive FR by adding the analytic
        # reactor offset, which is a physical energy in eV. A single additive
        # offset in normalised space is exact only when every Gibbs column
        # shares one scale -- which grouped z-score gives and per-column
        # min-max does not (each column gets its own min and range). Under
        # min-max the offset was also divided by a `std` key that min-max never
        # stores, so it silently fell back to 1.0 and FR came out ~148 eV wrong.
        norm_mode = ('zscore'
                     if config.model in ('ensemble', 'mogp', 'delta_gp', 'delta_de')
                     else 'minmax')
        data_manager.normalize(mode=norm_mode)

        # Build ensemble noise std for ensemble sampling (perturbed targets)
        if config.predict_landscape and config.use_ensemble:
            gibbs_std = data_manager.norm_stats.get('gibbs_AR_0', {}).get('std', None)
            noise_std = config.build_ensemble_noise_std(gibbs_std=gibbs_std)
            logger.info(f'Ensemble sampling noise_std (normalized): {[f"{x:.4f}" for x in noise_std]}')

        if config.use_validation:
             data_manager.split_data(val_split=config.val_split, mode=config.val_split_mode)
        
        # WAE Data (Includes OTC for WAE training)
        if config.use_validation:
             dataloader, val_dataloader = data_manager.get_wae_data(batch_size=params['batch_size'], device=config.device)
        else:
             dataloader = data_manager.get_wae_data(batch_size=params['batch_size'], device=config.device)
             val_dataloader = None
        
        # Get dataset reference for later use (visualization)
        dataset = dataloader.dataset
             
        # Ensemble/GP Data 
        # get_ensemble_data always returns 6 values: train_x, train_y, val_x, val_y, test_x, test_y
        ens_train_x, ens_train_y_dict, ens_val_x, ens_val_y_dict, _, _ = data_manager.get_ensemble_data()
        
        # Determine effective output dimension for ensemble
        # If landscape mode, we exclude OTC, so dim is n_props - 1
        # If not, dim is n_props (usually 1 for OTC)
        if config.predict_landscape:
            ensemble_output_dim = n_props - 1
        else:
            ensemble_output_dim = n_props

        # ---------------------------Train models and generate samples---------------------
        if not should_skip_stage(STAGE_TRAINING_AND_SAMPLING, progress_state):
            # Ensure cycle folder exists before any operations
            os.makedirs(new_folder_name, exist_ok=True)

            # Initialize model_dict and ensemble_model
            model_dict = None
            ensemble_model = None

            # Train GP models if enabled
            if config.use_gp:
                logger.info('\nTraining Gaussian Process models...')
                # Prepare GP input: use latent vectors if flag is set, otherwise use compositions
                gp_train_x = ens_train_x # Use ensemble/GP training data
                model_dict = {}
                for i in range(ensemble_output_dim):
                    model_key = f'model{i}'
                    # Use keys y0, y1... from DataManager
                    model_dict[model_key] = GPRegressor(gp_train_x, ens_train_y_dict[f'y{i}'], device=config.device).to(config.device)
                    _gp_loss_history = train_GP_new(model_dict[model_key], gp_train_x, ens_train_y_dict[f'y{i}'], config.default_gp_training_iters, logger=logger)
                    model_dict[model_key].print_parameters(logger)

            # Train the multi-output GP (ICM) if enabled
            if config.use_mogp:
                logger.info('\nTraining multi-output GP (ICM) on the Gibbs landscape...')
                mogp_train_x = ens_train_x  # Compositions [N, n_elements]
                mogp_train_y = torch.stack(
                    [ens_train_y_dict[f'y{i}'] for i in range(ensemble_output_dim)], dim=1
                )  # [N, 16] normalized Gibbs energies

                # gibbs_AR_0 / gibbs_FR_0 are identically zero by construction
                # (the v=0 reference). Left in, they hand the coregionalization
                # matrix zero rows and push the per-task noise onto its
                # constraint floor, so they are dropped here and reinstated at
                # their constant normalized value by the adapter.
                active_idx, const_map = split_constant_columns(mogp_train_y)
                logger.info(f'MOGP tasks: {ensemble_output_dim} total, {len(active_idx)} varying '
                            f'(constant: {sorted(const_map)})')

                mogp_model = ICMGPRegressor(
                    mogp_train_x, mogp_train_y[:, active_idx],
                    rank=config.mogp_rank,
                    device=config.device,
                    max_cholesky=config.mogp_max_cholesky,
                )
                if config.mogp_max_cholesky <= len(mogp_train_x):
                    logger.warning(
                        f'--mogp-max-cholesky ({config.mogp_max_cholesky}) <= N_train '
                        f'({len(mogp_train_x)}); GPyTorch will fall back to CG, which does not '
                        f'converge here. Raise it above N_train.'
                    )

                _mogp_loss_history = fit_mogp(
                    mogp_model, mogp_train_x, mogp_train_y[:, active_idx],
                    n_iters=config.mogp_iters, lr=config.mogp_lr,
                    patience=config.mogp_patience, logger=logger
                )

                # Convergence guard. Under-convergence here is silent: the loss
                # still looks like it is decreasing, but the per-task noises
                # never separate and the landscape comes out over-smoothed. The
                # benchmark that selected rank 1 early-stopped between 1153 and
                # 1801 iterations at a loss near -0.73; a fit that runs into the
                # iteration cap has not settled.
                if len(_mogp_loss_history) >= config.mogp_iters:
                    logger.warning(
                        f'MOGP hit the {config.mogp_iters}-iteration cap without early stopping '
                        f'(final loss {_mogp_loss_history[-1]:.4f}). The fit is probably '
                        f'under-converged -- raise --mogp-iters.'
                    )

                task_noise = mogp_model.task_noise_std()
                if task_noise is not None:
                    gibbs_std = data_manager.norm_stats.get('gibbs_AR_0', {}).get('std', 1.0)
                    noise_ev = task_noise.numpy() * gibbs_std
                    logger.info('MOGP learned per-task noise (eV): '
                                + ' '.join(f'{v:.2f}' for v in noise_ev))
                    # A converged fit recovers a ramp with vacancy level (roughly
                    # 0.37 -> 0.66 eV in the benchmark). Flat noise across tasks
                    # is the signature of a fit stuck near its initialisation.
                    spread = float(noise_ev.max() - noise_ev.min())
                    logger.info(f'MOGP task-noise spread: {spread:.3f} eV '
                                f'(converged reference ~0.29 eV)')
                    if spread < 0.10:
                        logger.warning(
                            'MOGP per-task noise is nearly flat across vacancy levels, which '
                            'indicates the marginal likelihood has not converged. Predictions '
                            'will be over-smoothed and OTC ranking degraded.'
                        )

                # Present the GP through the DeepEnsemble interface: one
                # posterior landscape draw per "member". Drawing a member
                # uniformly at random (al_functions._apply_acquisition_strategy)
                # then becomes exact Thompson sampling for the GP rather than the
                # parametric Gaussian approximation samples_GP uses.
                ensemble_model = MOGPEnsembleAdapter(
                    mogp_model, active_idx, const_map,
                    output_dim=ensemble_output_dim,
                    n_draws=config.mogp_draws,
                    chunk_size=config.mogp_chunk_size,
                )
                logger.info(f'MOGP wrapped as {ensemble_model.n_members} posterior draws '
                            f'({ensemble_output_dim} landscape dims)')

            # Train the Delta-GP if enabled
            if config.use_delta_gp:
                logger.info('\nTraining Delta-GP on the air-reactor Gibbs curve...')
                n_vac = ensemble_output_dim // 2
                vac_points = torch.arange(n_vac, dtype=torch.float64).numpy()

                # Only the AR half is modelled. The FR half is the same curve
                # shifted by the oxygen chemical potential difference between the
                # reactor conditions, which is a constant of those conditions
                # rather than something to learn: measured over the training data
                # the offset matches the analytic value to four decimals.
                ar_cols = torch.stack(
                    [ens_train_y_dict[f'y{i}'] for i in range(n_vac)], dim=1
                )  # [N, n_vac] normalised AR curve

                dg_x_np, dg_y_np = build_training_matrix(
                    ens_train_x.detach().cpu().numpy(),
                    ar_cols.detach().cpu().numpy(),
                    vac_points)
                dg_x = torch.tensor(dg_x_np, dtype=torch.float32, device=config.device)
                dg_y = torch.tensor(dg_y_np, dtype=torch.float32, device=config.device)
                logger.info(f'Delta-GP inputs: {dg_x.shape[0]} rows '
                            f'({ens_train_x.shape[0]} compositions x {n_vac} vacancy levels), '
                            f'{dg_x.shape[1]} features')

                delta_model = DeltaGPRegressor(
                    dg_x, dg_y, device=config.device,
                    max_cholesky=config.delta_gp_max_cholesky).to(config.device)
                if config.delta_gp_max_cholesky <= dg_x.shape[0]:
                    logger.warning(
                        f'--delta-gp-max-cholesky ({config.delta_gp_max_cholesky}) <= '
                        f'training rows ({dg_x.shape[0]}); GPyTorch will fall back to CG, '
                        f'which does not converge here. Raise it above the row count.')
                _dg_hist = train_delta_gp(
                    delta_model, dg_x, dg_y,
                    iters=config.delta_gp_iters, lr=config.delta_gp_lr,
                    patience=config.delta_gp_patience,
                    min_delta=config.delta_gp_min_delta, logger=logger)
                if len(_dg_hist) >= config.delta_gp_iters:
                    logger.warning(
                        f'Delta-GP hit the {config.delta_gp_iters}-iteration cap without early '
                        f'stopping (final loss {_dg_hist[-1]:.4f}); raise --delta-gp-iters.')
                else:
                    logger.info(f'Delta-GP converged after {len(_dg_hist)} iters '
                                f'(final loss {_dg_hist[-1]:.4f})')

                # The offset is a physical energy, so it has to be scaled into the
                # same normalised units as the targets. Landscape mode gives every
                # Gibbs column one shared z-score, so a single std applies.
                gibbs_std = _shared_gibbs_scale(data_manager)
                ensemble_model = DeltaGPEnsembleAdapter(
                    delta_model, vac_points,
                    output_dim=ensemble_output_dim,
                    n_draws=config.delta_gp_draws,
                    chunk_size=config.delta_gp_chunk_size,
                    gibbs_scale=gibbs_std,
                )
                logger.info(f'Delta-GP wrapped as {ensemble_model.n_members} posterior draws '
                            f'({ensemble_output_dim} landscape dims, FR from analytic offset)')

            # Train the Delta-DE if enabled
            if config.use_delta_de:
                logger.info('\nTraining Delta-DE (deep ensemble on the delta-as-input target)...')
                n_vac = ensemble_output_dim // 2
                vac_points = torch.arange(n_vac, dtype=torch.float64).numpy()

                # Same target as the Delta-GP: predict the air-reactor curve
                # only, with the vacancy level as an input column. The ensemble,
                # its training loop, validation split, early stopping and
                # anchoring are the production ones -- only the data shape
                # differs -- so this isolates the target representation from the
                # choice of surrogate.
                def _expand(x, y_cols=None):
                    xs = x.repeat_interleave(n_vac, dim=0)
                    vac = torch.arange(n_vac, dtype=x.dtype, device=x.device)
                    vs = vac.repeat(x.shape[0]).unsqueeze(-1)
                    xe = torch.cat([xs, vs], dim=-1)
                    if y_cols is None:
                        return xe, None
                    return xe, torch.stack(y_cols, dim=1).reshape(-1, 1)

                dde_train_x, dde_train_y = _expand(
                    ens_train_x, [ens_train_y_dict[f'y{i}'] for i in range(n_vac)])
                if config.use_validation and ens_val_x is not None:
                    dde_val_x, dde_val_y = _expand(
                        ens_val_x, [ens_val_y_dict[f'y{i}'] for i in range(n_vac)])
                else:
                    dde_val_x, dde_val_y = None, None
                logger.info(f'Delta-DE inputs: {dde_train_x.shape[0]} rows '
                            f'({ens_train_x.shape[0]} compositions x {n_vac} vacancy levels), '
                            f'{dde_train_x.shape[1]} features'
                            + (f', {dde_val_x.shape[0]} validation rows'
                               if dde_val_x is not None else ', no validation'))

                # Constructed exactly as the --model ensemble arm is, except for
                # input_dim (one extra column for the vacancy level) and
                # output_dim (one curve value per row instead of 16 per
                # composition).
                dde_model = DeepEnsemble(
                    n_members=config.ensemble_size,
                    input_dim=dde_train_x.shape[1],
                    hidden_dims=config.ensemble_hidden_dims,
                    dropout_rate=config.ensemble_dropout_rate,
                    device=config.device,
                    output_dim=1,
                    prior_variance=config.ensemble_prior_variance,
                )
                _dde_hist = train_deep_ensemble(
                    dde_model, dde_train_x, dde_train_y,
                    val_x=dde_val_x, val_y=dde_val_y,
                    bootstrap_fraction=config.bootstrap_fraction,
                    epochs=config.delta_de_epochs,
                    batch_size=config.ensemble_batch_size,
                    lr=config.delta_de_lr,
                    weight_decay=config.ensemble_weight_decay,
                    base_seed=config.default_seed,
                    config=config,
                    patience=config.early_stopping_patience,
                    prior_reg=config.ensemble_prior_reg,
                    noise_std=config.ensemble_noise_std,
                    logger=logger,
                )

                gibbs_std = _shared_gibbs_scale(data_manager)
                ensemble_model = DeltaEnsembleAdapter(
                    dde_model, vac_points,
                    output_dim=ensemble_output_dim,
                    chunk_size=config.delta_gp_chunk_size,
                    gibbs_scale=gibbs_std,
                )
                logger.info(f'Delta-DE wrapped as {ensemble_model.n_members} members '
                            f'({ensemble_output_dim} landscape dims, FR from analytic offset)')

            if config.use_ensemble:
                # Prepare ensemble training data: use compositions (ens_train_x)
                ensemble_train_x = ens_train_x  # Compositions [N, n_elements]
                
                # Prepare ensemble targets: stack all properties
                if ensemble_output_dim > 1:
                    # Stack properties [N, ensemble_output_dim]
                    train_y_list = [ens_train_y_dict[f'y{i}'] for i in range(ensemble_output_dim)]
                    ensemble_train_y = torch.stack(train_y_list, dim=1)
                    
                    if config.use_validation:
                        val_y_list = [ens_val_y_dict[f'y{i}'] for i in range(ensemble_output_dim)]
                        ensemble_val_y = torch.stack(val_y_list, dim=1)
                        ensemble_val_x = ens_val_x
                    else:
                        ensemble_val_y = None
                        ensemble_val_x = None
                else:
                    # Single property [N]
                    ensemble_train_y = ens_train_y_dict['y0']
                    ensemble_val_x = ens_val_x if config.use_validation else None
                    ensemble_val_y = ens_val_y_dict['y0'] if config.use_validation else None

                # Get input dimension from train_x (number of elements)
                input_dim = ensemble_train_x.shape[1]
                
                # Assertions to verify dimensions
                if ensemble_train_y.ndim == 1:
                    assert ensemble_output_dim == 1, f"Expected 1 output dim but got {ensemble_output_dim}"
                else:
                    assert ensemble_train_y.shape[1] == ensemble_output_dim, \
                        f"Ensemble targets have {ensemble_train_y.shape[1]} dims but expected {ensemble_output_dim}"

                if config.predict_landscape:
                    # In landscape mode, we expect Gibbs only (16)
                    # Total methods = 17 (OTC + 16 Gibbs). 
                    # Ensemble output should be 16.
                    expected_dim = n_props - 1
                    assert ensemble_output_dim == expected_dim, \
                        f"Landscape mode: Expected {expected_dim} outputs (Gibbs), got {ensemble_output_dim}"
                    logger.info(f"Verified ensemble dimensions for landscape mode: {ensemble_output_dim} outputs (OTC excluded)")
                else:
                    # standard mode
                    logger.info(f"Verified ensemble dimensions: {ensemble_output_dim} outputs")


                # Create ensemble
                ensemble_model = DeepEnsemble(
                    n_members=config.ensemble_size,
                    input_dim=input_dim,
                    hidden_dims=config.ensemble_hidden_dims,
                    dropout_rate=config.ensemble_dropout_rate,
                    device=config.device,
                    output_dim=ensemble_output_dim,  # 16 if landscape (OTC excluded), 1 if OTC
                    prior_variance=config.ensemble_prior_variance,
                )

                # Train ensemble
                _ensemble_loss_histories = train_deep_ensemble(
                    ensemble_model,
                    ensemble_train_x,
                    ensemble_train_y,
                    val_x=ensemble_val_x,
                    val_y=ensemble_val_y,
                    bootstrap_fraction=config.bootstrap_fraction,
                    epochs=config.ensemble_epochs,
                    batch_size=config.ensemble_batch_size,
                    lr=config.ensemble_lr,
                    weight_decay=config.ensemble_weight_decay,
                    base_seed=config.default_seed,
                    config=config,
                    patience=config.early_stopping_patience,
                    prior_reg=config.ensemble_prior_reg,
                    noise_std=config.ensemble_noise_std,
                    logger=logger
                )

                # Apply post-hoc calibration if requested
                if config.ensemble_calibrate and ensemble_val_x is not None:
                    logger.info('Applying post-hoc temperature calibration to ensemble...')
                    from ensemble_mlp import calibrate_ensemble
                    temperature = calibrate_ensemble(ensemble_model, ensemble_val_x, ensemble_val_y, logger)
                    ensemble_model.temperature = temperature
                    logger.info(f'Calibration temperature: {temperature:.4f}')
                elif config.ensemble_calibrate:
                    logger.warning('--ensemble-calibrate requires --use-validation, skipping calibration')

            # Train WAE (after regression models so latent space learns from their predictions)
            cycle_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_cycle_{cycle}.pt')
            if os.path.exists(cycle_checkpoint):
                logger.info(f'WAE already trained for cycle {cycle}, loading checkpoint')
                load_checkpoint(cycle_checkpoint, wae_model, config.device)
            else:
                logger.info('Train WAE')
                params = config.get_wae_params(cycle)
                optimizer = Adam(wae_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
                wae_loss_history = train_WAE(wae_model, optimizer, dataloader, params, logger=logger, val_dataloader=val_dataloader)
                log_wae_loss(wae_loss_history, cycle, config.log_path)

                save_checkpoint(wae_model, cycle, checkpoint_dir, optimizer=optimizer)

            # --------------------Visualize latent space------------------------
            property_values = dataset.x[:, 0]  # OTC values
            viz_results = visualize_latent_space(
                wae_model=wae_model,
                dataset=dataset,
                property_values=property_values,
                cycle=cycle,
                log_path=config.log_path,
                property_name=config.method_list[0],
                device=config.device,
                logger=logger
            )

            # --------------------Analyze latent structure------------------------
            analyze_latent_structure(
                latent_reps=viz_results['latent_reps'],
                property_values=viz_results['property_values'],
                logger=logger,
                cycle=cycle,
                log_path=config.log_path
            )

            # --------------------Generate samples-----------------------------
            sample_name = new_folder_name + f'/sample.csv'
            latents = get_latents(wae_model, dataset)

            if (config.use_ensemble or config.use_mogp or config.use_delta_gp
                    or config.use_delta_de):
                # Deep Ensemble mode, or a GP surrogate presented as one through
                # MOGPEnsembleAdapter / DeltaGPEnsembleAdapter
                samples_ensemble(wae_model, ensemble_model, sample_name, latents, cycle,
                                config.method_list, config, config.log_path, logger=logger, cycle=cycle,
                                existing_df=data_manager.raw_df)
            elif config.use_gp:
                # GP mode
                samples_GP(wae_model, model_dict, sample_name, latents, cycle,
                           config.method_list, config, config.log_path, logger=logger, cycle=cycle,
                           existing_df=data_manager.raw_df)
            else:
                # MC Dropout mode
                samples_mcdrop(wae_model, sample_name, latents, cycle,
                               config.method_list, config, config.log_path, logger=logger, cycle=cycle,
                               existing_df=data_manager.raw_df)
            # Save progress after both training and sampling complete
            save_progress_state(config.log_path, cycle, STAGE_TRAINING_AND_SAMPLING)
        else:
            logger.info('Skipping training and sampling - already completed')
    
        # Reviewer package: stop once the candidates of this cycle are selected,
        # before the CHGNet/Phonopy oracle and the Materials Project queries.
        if os.environ.get('AL_STOP_AFTER_SAMPLING'):
            logger.info('AL_STOP_AFTER_SAMPLING is set: stopping after candidate selection')
            break

        # ==================================================================================
        # STAGE 2: PREPARE SAMPLES FOR PROPERTY CALCULATIONS
        # ==================================================================================
        # Build sample database from selected candidates
        if not should_skip_stage(STAGE_SAMPLE_DB, progress_state):
            logger.info('Building sample database...')
            build_sample_database(directory=f'{config.method}', random_seed=config.default_seed)
            logger.info('Sample db created!')
            save_progress_state(config.log_path, cycle, STAGE_SAMPLE_DB)
        else:
            logger.info('Skipping sample database - already created')

        if not should_skip_stage(STAGE_SQS, progress_state):
            logger.info('Building SQS structures...')
            build_sqs_structures(processes=config.default_samples_per_cycle, directory=f'{config.method}',
                                 random_seed=config.default_seed,
                                 max_cations=config.max_element_count)
            logger.info('SQS structures built!')
            save_progress_state(config.log_path, cycle, STAGE_SQS)
        else:
            logger.info('Skipping SQS structures - already built')

        if not should_skip_stage(STAGE_PHONON, progress_state):
            logger.info('Performing phonon calculations (this may take a while)...')
            if config.phonon_slurm_jobs > 0:
                # Fan out phonon calculations across multiple SLURM GPU jobs
                submit_phonon_slurm_jobs(config, cycle, logger)
            else:
                # Run locally on current node (threaded relaxations run inside if configured)
                perform_phonon_calculations(
                    processes=1, directory=f'{config.method}', use_gpus=config.use_gpu,
                    logging=logger, batch_size=config.phonon_batch_size,
                    n_relax_threads=config.relax_threads or 0)
            logger.info('Phonon calculations complete!')
            save_progress_state(config.log_path, cycle, STAGE_PHONON)
        else:
            logger.info('Skipping phonon calculations - already completed')

        if not should_skip_stage(STAGE_PHASE_STABILITY, progress_state):
            logger.info('Updating phase stability...')
            api_key = SETTINGS['PMG_MAPI_KEY']
            # The Materials Project rate limits aggressively when several runs
            # query it at once, so keep the per-run worker count modest.
            update_phase_stability_dbs(
                filenames=glob(f'{new_folder_name}/sqs_*.db'),
                api_key=api_key,
                processes=config.phase_stability_processes)
            logger.info('Phase stability updated!')
            save_progress_state(config.log_path, cycle, STAGE_PHASE_STABILITY)
        else:
            logger.info('Skipping phase stability - already updated')

        if not should_skip_stage(STAGE_TRAINING_DB, progress_state):
            logger.info('Updating training database...')
            update_training_database(
                directory=f'{config.method}',
                quadratic_fit=config.quadratic_fit_otc
            )
            logger.info('Training database updated!')
            save_progress_state(config.log_path, cycle, STAGE_TRAINING_DB)
        else:
            logger.info('Skipping training database update - already completed')

        if not should_skip_stage(STAGE_COMPARE_PREDICTIONS, progress_state):
            # Compare predictions with actual values for this cycle
            logger.info('Comparing predictions with actual values...')
            from al_functions import compare_predictions_with_actuals
            compare_predictions_with_actuals(
                cycle=cycle,
                log_path=config.log_path,
                sample_file=f'{new_folder_name}/sample.csv',
                db_name=config.db_name,
                elements=config.elements,
                method_list=config.method_list,
                config=config,
                logger=logger
            )
            save_progress_state(config.log_path, cycle, STAGE_COMPARE_PREDICTIONS)
        else:
            logger.info('Skipping prediction comparison - already completed')

        # Snapshot the RNG streams so a resume at cycle+1 continues this exact
        # realisation instead of restarting from the seed. See save_rng_state.
        save_rng_state(cycle, checkpoint_dir)

        logger.info(f'Cycle {cycle} complete!')
        progress_state = None  # Reset progress for new cycle

    # ---------------------------Create evolution plot---------------------
    logger.info('Creating latent space evolution plot...')
    plot_latent_evolution(log_path=config.log_path, property_name=config.method_list[0], logger=logger)
    logger.info('Active learning pipeline complete!')


if __name__ == '__main__':
    main()
