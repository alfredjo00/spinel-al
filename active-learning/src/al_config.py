"""Configuration of the active learning loop: command-line options and the Config class.

run_arm.sh at the repository root gives the options of each strategy in the article.
An example, the Gaussian process with the Gibbs-vector target, seed 1:

    python -u active_learning_HEO.py --method spinel_rev_mogp_s1 --auto-resume --seed 1 \
        --model mogp --ensemble-size 16 --mogp-rank 1 \
        --predict-landscape --quadratic-fit-landscape-otc \
        --cycles 21 -b 3 -t 10 --n-samples-total 25000 --max-element-count 8 \
        --use-validation --val-split 0.15 --val-split-mode random \
        --round-compositions --quadratic-fit-otc --bootstrap-fraction none
"""

import argparse
import os
import re
import torch
from glob import glob
import shutil




class Config:
    """Consolidated configuration for HEO Active Learning Pipeline.
    
    This class consolidates all constants, parameters, spinel structure configuration,
    and parsed arguments into a single source of truth.
    """
    
    def __init__(self, args):
        """Initialize configuration from parsed arguments.
        
        Args:
            args: Parsed command-line arguments from argparse
        """
        # ==================== ARGUMENTS ====================
        # Store all parsed arguments
        self.args = args
        self.log_path = args.log_path
        self.checkpoint_dir = args.checkpoint_dir
        self.model = args.model
        self.ensemble_size = args.ensemble_size
        self.bootstrap_fraction = args.bootstrap_fraction
        self.method = args.method
        self.max_element_count = args.max_element_count
        self.b = args.b
        self.t = args.t
        self.n_samples_total = args.n_samples_total
        self.use_validation = args.use_validation
        self.val_split = args.val_split
        self.overwrite = args.overwrite
        self.val_split_mode = args.val_split_mode
        self.round_compositions = args.round_compositions
        self.predict_landscape = args.predict_landscape
        self.clip_otc_physical = args.clip_otc_physical
        self.quadratic_fit_otc = args.quadratic_fit_otc
        self.quadratic_fit_landscape_otc = args.quadratic_fit_landscape_otc
        self.auto_resume = args.auto_resume

        self.use_gp = self.model == 'gp'
        self.use_mc = self.model == 'mc'
        self.use_ensemble = self.model == 'ensemble'
        self.use_mogp = self.model == 'mogp'
        self.use_delta_gp = self.model == 'delta_gp'
        self.use_delta_de = self.model == 'delta_de'
        self.use_ceiling = self.model == 'ceiling'
        self.sampling = 'thompson'  # acquisition by Thompson sampling, as in all runs

        # ==================== CONSTANTS ====================
        self.default_cycles = args.cycles
        self.default_samples_per_cycle = 10
        # --seed omitted reproduces the historical behaviour exactly: torch
        # seeded at 42, numpy and `random` left alone. Passing --seed makes the
        # run repeatable end to end (see same_seeds) and is what replicate
        # studies should use, with one seed per replicate.
        self.deterministic = args.seed is not None
        self.default_seed = args.seed if args.seed is not None else 42
        self.initial_wae_epochs = 1500
        self.subsequent_wae_epochs = 500
        self.default_gp_training_iters = 750
        
        # ==================== SPINEL STRUCTURE ====================
        self.atoms_in_structure = 84  # Total atoms in spinel supercell
        self.a_scaling = 1.0/7.0  # A sublattice occupies 1/7 of total atoms
        self.b_scaling = 2.0/7.0  # B sublattice occupies 2/7 of total atoms
        self.x_scaling = 4.0/7.0  # X (oxygen) sublattice occupies 4/7 of total atoms
        self.n_vacancies_max = 8  # Range of vacancies in spinel supercell (84 atoms), from 0 to 7
        self.sqs_dimensions = (1, 2, 3)  # SQS supercell dimensions
        
        # ==================== ELEMENTS ====================
        # Spinel structure: A[B]2O4 where A occupies tetrahedral sites, B occupies octahedral sites
        self.A_elements = ['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca', 'Li']  # Tetrahedral sites (1/7 of structure)
        self.B_elements = ['Al', 'Cr', 'Ga', 'Ti', 'Fe', 'Mn', 'Zr', 'V', 'Sn', 'In']  # Octahedral sites (2/7)
        self.X_elements = ['O']  # Oxygen (4/7 of structure)
        self.elements = self.A_elements + self.B_elements + self.X_elements
        
        # ==================== SCALING DICTIONARY ====================
        # Create scaling dictionary for sublattice concentrations
        # These fractions ensure concentrations sum correctly for the spinel AB2O4 structure
        self.scaling_dictionary = {
            **{elem: self.a_scaling for elem in self.A_elements},  # 1/7
            **{elem: self.b_scaling for elem in self.B_elements},  # 2/7
            **{elem: self.x_scaling for elem in self.X_elements},  # 4/7
        }
        
        # ==================== ENSEMBLE PARAMETERS ====================
        # Architecture parameters, smaller networks seem better, less overfitting
        self.ensemble_hidden_dims = [64, 64]  # Hidden layer dimensions
        self.ensemble_dropout_rate = 0.0  # Dropout rate for ensemble members

        # Training parameters
        self.ensemble_epochs = 500  # Maximum training epochs
        self.ensemble_lr = 2e-4  # Learning rate
        self.ensemble_weight_decay = 0  # Weight decay for regularization, WARNING: May cause ensemble collapse, careful
        self.ensemble_batch_size = 64  # Batch size for training        
        self.early_stopping_patience = 150
        
        # Learning rate scheduler parameters
        self.ensemble_lr_scheduler_patience = 50  # Patience for LR scheduler
        self.ensemble_lr_scheduler_gamma = 0.5  # LR decay factor
        self.ensemble_use_scheduler = True  # Whether to use LR scheduler
        
        self.ensemble_otc_quadratic_fit_samples = 1 # Number of samples per member for otc distribution
        self.ensemble_otc_quadratic_fit_member_mean = True # Whether to use member mean for otc distribution, req. 1 sample

        # Ensemble Sampling parameters (Lu & Van Roy, 2017)
        # Loss_m = NLL(y + Z̃_m, f_θ) + (1/λ) ||θ - θ̃_m||²
        # where λ is the prior variance, likewies with pearce 2020 anchored ensembles
        # where θ̃_m ~ N(0, λI) and Z̃_m ~ N(0, noise_std²)
        self.ensemble_prior_variance = 1   # λ: Prior variance for Gaussian weight init
        self.ensemble_prior_reg = True    # 0 = disabled, 1 = standard ensemble sampling
        
        if self.ensemble_weight_decay:
            assert not self.ensemble_prior_reg, "When using weight decay, prior_reg must be False, or they interfere. Generally best to avoid weight decay."

        # Target perturbation noise std (in normalized units, after z-score)
        # Raw Gibbs noise increases linearly (close enough): 0.2 to 0.8 eV across vacancy range
        # Normalized noise = raw_noise / gibbs_std
        self.ensemble_noise_std = None  # Set by build_ensemble_noise_std() if enabled
        self._gibbs_norm_std_default = 7.331  # Fallback if not available from data
        self._noise_raw_min = 0.2     # Raw noise std at vacancy=0 
        self._noise_raw_max = 0.8     # Raw noise std at vacancy=7 

        # ==================== MULTI-OUTPUT GP (ICM) PARAMETERS ====================
        # Intrinsic coregionalization model over the Gibbs landscape:
        #   K = k_Matern-5/2,ARD(x, x') (x) B,   B = W W^T + diag(kappa),  W in R^{T x rank}
        # Consumed through MOGPEnsembleAdapter, so it reuses the ensemble's OTC
        # pipeline (quadratic fit -> Delta v -> OTC) unchanged; one posterior draw
        # plays the role of one ensemble member.
        self.delta_gp_iters = args.delta_gp_iters
        self.delta_gp_lr = args.delta_gp_lr
        self.delta_gp_patience = args.delta_gp_patience
        self.delta_gp_min_delta = args.delta_gp_min_delta
        self.delta_de_epochs = (args.delta_de_epochs
                                if args.delta_de_epochs is not None
                                else self.ensemble_epochs)
        self.delta_de_lr = (args.delta_de_lr
                            if args.delta_de_lr is not None
                            else self.ensemble_lr)
        self.delta_gp_chunk_size = args.delta_gp_chunk_size
        self.delta_gp_max_cholesky = args.delta_gp_max_cholesky
        self.delta_gp_draws = (args.delta_gp_draws
                               if args.delta_gp_draws is not None
                               else self.ensemble_size)
        self.mogp_rank = args.mogp_rank
        self.mogp_iters = args.mogp_iters
        self.mogp_lr = args.mogp_lr
        self.mogp_patience = args.mogp_patience
        # Posterior draws per prediction = "ensemble members" for acquisition.
        # Defaults to --ensemble-size so Thompson sampling has the same
        # granularity across surrogates.
        self.mogp_draws = args.mogp_draws if args.mogp_draws is not None else self.ensemble_size
        self.mogp_chunk_size = args.mogp_chunk_size
        # GPyTorch falls back to CG above max_cholesky_size (default 800) without
        # warning, and CG does not converge here (residual ~1e5 vs a tolerance of
        # 1) besides being ~100x slower. Must exceed N_train.
        self.mogp_max_cholesky = args.mogp_max_cholesky

        # ==================== WAE PARAMETERS ====================
        # Network architecture parameters
        self.latent_dim = 8
        self.use_regression = True
        self.decoder_scale = 4
        self.regression_scale = 2

        # Dropout configuration
        self.full_dropout = True  # Use full dropout for MC model
        self.encoder_dropout = 0.1
        self.decoder_dropout = 0.1
        self.regression_dropout = 0.1

        # Training parameters
        self.wae_batch_size = 32
        self.wae_lr = 1e-3
        self.subsequent_wae_lr = 2e-4
        self.wae_weight_decay = 0
        self.wae_sigma = 1.0
        self.wae_mmd_lambda = 1e-2
        self.wae_early_stopping_patience = 800
        self.wae_lr_patience = 100        
        self.subsequent_wae_lr_patience = 50
        self.wae_max_grad_norm = 1.0

        # Normalization method (set during data loading)
        # 'zscore' = z-score (mean=0, std=1) for ensemble/NLL
        # 'minmax' = min-max [0,1] for GP/MC
        self.normalization = None  # Will be set by load_training_data()

        # Worker processes querying the Materials Project during the phase
        # stability stage. Kept low because concurrent runs share one API quota
        # and trigger rate limiting well before the CPUs are saturated.
        self.phase_stability_processes = 4

        # ==================== DERIVED PROPERTIES ====================
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.phonon_batch_size = self.b if self.use_gpu else None
        self.relax_threads = self.t if self.use_gpu else None
        self.tol = 1 / self.atoms_in_structure

        # Inverse transform function (set after data loading)
        # For landscape mode, this will be an identity function
        self.inverse_transform = None
                
        # ==================== PATHS ====================
        # Determine project root and paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        self.heoc_path = os.path.join(self.project_root, 'oxygen-carrier-discovery', 'heoc')
        self.folder_name = f'{self.heoc_path}/{self.method}/cycle'
        self.db_name = f'{self.heoc_path}/{self.method}/training.db'
        
    def validate(self):
        """Validate argument compatibility."""
        if self.use_ensemble and self.use_mc:
            raise ValueError("Cannot use both --use-ensemble and --use-mc. Choose one uncertainty estimation method.")

        if self.use_mogp:
            # The ICM's whole point is the joint landscape; on scalar OTC it
            # degenerates to the plain --model gp baseline, and the adapter has
            # no landscape to hand samples_ensemble.
            if not self.predict_landscape:
                raise ValueError(
                    "--model mogp requires --predict-landscape: the ICM models the joint Gibbs "
                    "landscape. For a scalar-OTC GP use --model gp."
                )
            if not self.quadratic_fit_landscape_otc:
                raise ValueError(
                    "--model mogp requires --quadratic-fit-landscape-otc: OTC is derived per "
                    "posterior draw by fitting each drawn G(v) curve, which is the MC-sampling "
                    "path's alternative."
                )
            if self.mogp_draws < 2:
                raise ValueError(f"--mogp-draws must be >= 2 to give a spread, got {self.mogp_draws}")

        if self.use_delta_de:
            # Same target as the Delta-GP, so the same requirements: it predicts
            # a Gibbs curve and OTC is derived from each member's curve. The
            # ensemble, its validation split and early stopping are the
            # production ones, so that the only difference against --model
            # ensemble --predict-landscape is the target representation.
            if not self.predict_landscape:
                raise ValueError(
                    "--model delta_de requires --predict-landscape: it models the Gibbs "
                    "curve G(composition, vacancies)."
                )
            if not self.quadratic_fit_landscape_otc:
                raise ValueError(
                    "--model delta_de requires --quadratic-fit-landscape-otc: OTC comes "
                    "from fitting each member's G(v) curve."
                )

        if self.use_delta_gp:
            # Same requirements as the ICM: the model predicts a Gibbs curve,
            # and OTC is derived from each drawn curve rather than regressed
            # directly. On scalar OTC this would just be --model gp with an
            # extra input column.
            if not self.predict_landscape:
                raise ValueError(
                    "--model delta_gp requires --predict-landscape: it models the Gibbs "
                    "curve G(composition, vacancies). For a scalar-OTC GP use --model gp."
                )
            if not self.quadratic_fit_landscape_otc:
                raise ValueError(
                    "--model delta_gp requires --quadratic-fit-landscape-otc: OTC comes from "
                    "fitting each drawn G(v) curve. Reading the minimum off a fine grid was "
                    "tried and was measurably worse, since the parabola averages prediction "
                    "error over all points while an argmin follows single-point wiggles."
                )
            if self.delta_gp_draws < 2:
                raise ValueError(
                    f"--delta-gp-draws must be >= 2 to give a spread, got {self.delta_gp_draws}")


    def ensure_initial_cycle(self):
        """Ensure cycle0 folder exists."""
        os.makedirs(f'{self.folder_name}0', exist_ok=True)
    
    def setup_log_path(self, timestamp):
        """Set up log path and handle cycle cleanup if needed.
        
        Args:
            timestamp: Timestamp string for new log directories (format: YYYYMMDD_HHMMSS)
            
        Returns:
            str: The log path to use
            
        Raises:
            ValueError: If existing cycles found but no log path provided
        """
        
        # Check if cycles already exist (beyond cycle0)
        existing_cycles = glob(f'{self.folder_name}*')
        non_initial_cycles = [c for c in existing_cycles if not c.endswith('cycle0')]
        
        # Handle overwrite flag - remove existing cycles except cycle0
        if self.overwrite and non_initial_cycles:
            print(f"--overwrite flag detected. Removing {len(non_initial_cycles)} existing cycle(s)...")
            for cycle_dir in non_initial_cycles:
                print(f"  Removing {os.path.basename(cycle_dir)}")
                shutil.rmtree(cycle_dir)
            non_initial_cycles = []
        
        # Determine log path
        if self.log_path:
            log_path = self.log_path
        elif non_initial_cycles and self.auto_resume:
            # Resubmitting the same jobscript should just continue the run.
            # Without this the operator has to paste the right --log-path by
            # hand for every resume; pasting the wrong one silently restores
            # another run's WAE weights and RNG history, which is worse than
            # any failure this check was guarding against.
            candidates = sorted(
                d for d in glob(f'{self.heoc_path}/{self.method}/*')
                if os.path.isdir(d) and re.fullmatch(r'\d{8}_\d{6}', os.path.basename(d))
            )
            if not candidates:
                raise ValueError(
                    f"--auto-resume was given and {len(non_initial_cycles)} cycle(s) exist, "
                    f"but no timestamped log directory was found under "
                    f"{self.heoc_path}/{self.method}. Cannot determine what to resume.")
            log_path = candidates[-1]
            print(f"--auto-resume: continuing run in {log_path} "
                  f"({len(candidates)} log dir(s) present, using most recent)")
        else:
            # No log path provided - check if resuming is needed
            if non_initial_cycles:
                raise ValueError(
                    f"ERROR: Found {len(non_initial_cycles)} existing cycle(s) but no --log-path was provided.\n"
                    f"Existing cycles: {[os.path.basename(c) for c in sorted(non_initial_cycles)]}\n\n"
                    f"To resume training, pass --auto-resume (picks the most recent log dir), "
                    f"or --log-path to name it explicitly.\n"
                    f"Example: --log-path {self.heoc_path}/{self.method}/20251119_172345\n\n"
                    f"To start fresh, delete existing cycles first or use --overwrite flag."
                )
            log_path = f'{self.heoc_path}/{self.method}/{timestamp}'
        
        # Ensure log_path is within the method directory
        assert f'/{self.method}/' in log_path or log_path.endswith(f'/{self.method}'), \
            f"Log path must be within the method directory. Expected '/{self.method}/' in path '{log_path}'"
            
        self.log_path = log_path
    
    def get_checkpoint_dir(self):
        """Get checkpoint directory path.
        
        Args:
            log_path: The log path being used
            
        Returns:
            str: Path to checkpoint directory
        """
        import os
        if self.checkpoint_dir:
            return self.checkpoint_dir
        else:
            return os.path.join(self.log_path, 'checkpoints')

    def log_initial_info(self, logger, log_path):
        """Log initial configuration information.
        
        Args:
            logger: Logger instance
            log_path: Path to log directory
        """
        logger.info(f'Using device: {self.device}')
        logger.info(f'folder: {self.folder_name}')
        logger.info(f'db_name: {self.db_name}')
        logger.info(f'logs saved to: {log_path}')
        
        logger.info('Input arguments: ')
        for k,v in vars(self.args).items():
            logger.info(f'{k}:\t{v}')
            
        self.set_method_list()
        if self.predict_landscape:
            logger.info(f'GP/Ensemble will predict OTC + Gibbs energy landscape ({len(self.method_list)} dimensions total: 1 OTC + {self.n_vacancies_max*2} Gibbs energies)')
        else:
            logger.info('GP/Ensemble will predict scalar OTC')
        
        # WAE always uses only the first property (OTC)
        logger.info('WAE will predict scalar OTC (first property in method_list)')
        
        logger.info('Initialise WAE')
        logger.info(f'Uncertainty estimation method: {self.model}')
        if self.use_ensemble:
            if self.bootstrap_fraction is None:
                logger.info(f'Deep Ensemble: {self.ensemble_size} members, NO bootstrapping (full dataset per member)')
            else:
                logger.info(f'Deep Ensemble: {self.ensemble_size} members, bootstrap fraction {self.bootstrap_fraction}')
        elif self.use_mogp:
            logger.info(f'Multi-output GP (ICM): coregionalization rank {self.mogp_rank}, '
                        f'{self.mogp_draws} posterior draws used as ensemble members')
            logger.info(f'MOGP training: {self.mogp_iters} iters, lr {self.mogp_lr}, '
                        f'patience {self.mogp_patience}, max_cholesky {self.mogp_max_cholesky}')
        elif self.use_delta_de:
            logger.info(f'Delta-DE: deep ensemble of {self.ensemble_size} members over '
                        f'(composition, vacancy level)')
            logger.info(f'Delta-DE training: {self.delta_de_epochs} epochs, '
                        f'lr {self.delta_de_lr}')
            logger.info('Delta-DE: fuel-reactor curve from the analytic oxygen '
                        'chemical potential offset, not modelled')
        elif self.use_delta_gp:
            logger.info(f'Delta-GP: one GP over (composition, vacancy level), '
                        f'{self.delta_gp_draws} posterior draws used as ensemble members')
            logger.info(f'Delta-GP training: {self.delta_gp_iters} iters, lr {self.delta_gp_lr}, '
                        f'patience {self.delta_gp_patience}, '
                        f'max_cholesky {self.delta_gp_max_cholesky}')
            logger.info('Delta-GP: fuel-reactor curve from the analytic oxygen '
                        'chemical potential offset, not modelled')
        elif self.use_gp:
            logger.info('Using Gaussian Process for uncertainty estimation')
        elif self.use_ceiling:
            logger.info('No surrogate: ranking the WAE pool by the physical OTC ceiling')
        else:
            logger.info(f'Using MC Dropout for uncertainty estimation (full dropout enabled)')
    
    def set_method_list(self):
        """Get list of properties to predict based on configuration.

        Returns:
            list: Property names to predict
        """
        if self.predict_landscape:
            # Predict Gibbs energies for AR and FR across vacancy range
            # IMPORTANT: OTC must be first for WAE training
            method_list = ['otc']  # OTC first for WAE
            for r in ['AR', 'FR']:
                for v in range(self.n_vacancies_max):
                    method_list.append(f'gibbs_{r}_{v}')
            self.method_list = method_list
        else:
            self.method_list = ['otc']

    def build_ensemble_noise_std(self, gibbs_std=None):
        """Build per-dimension noise std vector for ensemble sampling.

        Ensemble only predicts Gibbs energies (no OTC).
        Gibbs noise increases linearly with vacancy index:
            noise_raw(v) = noise_min + (noise_max - noise_min) * v / (n_vac - 1)
            noise_normalized = noise_raw / gibbs_std

        Args:
            gibbs_std: Gibbs normalization std from data. If None, uses default.

        Returns:
            list: [gibbs_AR_0, ..., gibbs_AR_N, gibbs_FR_0, ..., gibbs_FR_N]
        """
        import numpy as np

        n_vac = self.n_vacancies_max
        noise_min = self._noise_raw_min
        noise_max = self._noise_raw_max

        if gibbs_std is None:
            gibbs_std = self._gibbs_norm_std_default

        # Linear interpolation of raw noise across vacancy range
        vacancy_indices = np.arange(n_vac)
        raw_noise = noise_min + (noise_max - noise_min) * vacancy_indices / (n_vac - 1)
        normalized_noise = raw_noise / gibbs_std

        # Build vector: [AR_0..AR_N, FR_0..FR_N] (no OTC)
        noise_std = normalized_noise.tolist()  # AR
        noise_std.extend(normalized_noise.tolist())  # FR (same pattern)

        self.ensemble_noise_std = noise_std
        return noise_std

    def get_wae_params(self, cycle=1):
        """Get WAE training parameters for a given cycle.

        Args:
            cycle: Current AL cycle number. Cycle 1 uses initial params,
                   subsequent cycles use reduced lr/patience for fine-tuning.

        Returns:
            dict: WAE parameters
        """
        is_initial = (cycle == 1)

        return {
            # Training params (cycle-dependent)
            'num_epoch': self.initial_wae_epochs if is_initial else self.subsequent_wae_epochs,
            'batch_size': self.wae_batch_size,
            'lr': self.wae_lr if is_initial else self.subsequent_wae_lr,
            'lr_patience': self.wae_lr_patience if is_initial else self.subsequent_wae_lr_patience,
            'weight_decay': self.wae_weight_decay,
            'sigma': self.wae_sigma,
            'MMD_lambda': self.wae_mmd_lambda,
            'early_stopping_patience': self.wae_early_stopping_patience,
            'max_grad_norm': self.wae_max_grad_norm,

            # Network architecture parameters
            'latent_dim': self.latent_dim,
            'use_regression': self.use_regression,
            'lattice_A_size': len(self.A_elements),
            'lattice_B_size': len(self.B_elements),
            'n_properties': 1,  # WAE always predicts 1 property (OTC)
            'decoder_scale': self.decoder_scale,
            'regression_scale': self.regression_scale,

            # Dropout configuration
            'use_full_dropout': self.full_dropout,
            'encoder_dropout': self.encoder_dropout,
            'decoder_dropout': self.decoder_dropout,
            'regression_dropout': self.regression_dropout,
        }


    def save_to_file(self, log_path):
        """Save configuration to JSON file for reproducibility.

        Args:
            log_path: Directory to save config file

        Returns:
            str: Path to saved config file
        """
        import json
        import os

        config_dict = {}

        # Iterate through all attributes
        for key, value in vars(self).items():
            # Skip private attributes, args object, and functions
            if key.startswith('_') or key == 'args' or callable(value):
                continue

            # Convert non-serializable types
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            elif key == 'inverse_transform':
                # Skip inverse_transform (it's a function)
                config_dict[key] = 'identity' if self.predict_landscape else 'normalized'
            elif isinstance(value, (str, int, float, bool, type(None))):
                config_dict[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples (should be serializable)
                config_dict[key] = list(value)
            elif isinstance(value, dict):
                # Convert dicts (should be serializable)
                config_dict[key] = value
            else:
                # For other types, convert to string
                config_dict[key] = str(value)

        # Also add parsed arguments for reference
        config_dict['_command_line_args'] = vars(self.args) if hasattr(self, 'args') else {}

        # Save to file
        config_file = os.path.join(log_path, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        return config_file

def parse_args(argv=None):
    """Parse command line arguments, from sys.argv or the list argv.

    If --config-file is provided, loads saved command-line args from that file
    and uses them as defaults. Any additional CLI args will override the saved values.
    """
    import sys
    import json

    # Check for --config-file early to load saved args as defaults
    config_file = None
    args_list = sys.argv[1:] if argv is None else list(argv)
    for i, arg in enumerate(args_list):
        if arg == '--config-file' and i + 1 < len(args_list):
            config_file = args_list[i + 1]
            break
        elif arg.startswith('--config-file='):
            config_file = arg.split('=', 1)[1]
            break

    # Load saved arguments if config file provided
    saved_args = {}
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        saved_args = config_dict.get('_command_line_args', {})
        # Remove config_file from saved args to avoid recursion
        saved_args.pop('config_file', None)
        print(f"Loaded saved arguments from: {config_file}")
        
    parser = argparse.ArgumentParser(description='HEO Active Learning Pipeline')
    parser.add_argument('--config-file', type=str,
                       help='Load saved arguments from config.json file (CLI args override saved values)')
    parser.add_argument('--log-path', type=str,
                       help='Custom path for logs/saves. If not provided, uses default: heoc_path/method/timestamp')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Directory to save checkpoints. If not provided, uses log_path')
    parser.add_argument('--model', type=str, default='ensemble', choices=['gp', 'mc', 'ensemble', 'mogp', 'delta_gp', 'delta_de', 'ceiling'],
                       help='Method to use for uncertainty estimation (default: gp). '
                            '"mogp" is the ICM multi-output GP and requires --predict-landscape. '
                            '"ceiling" fits no surrogate and ranks the WAE pool by the physical OTC ceiling (baseline).')
    parser.add_argument('--ensemble-size', type=int, default=16,
                       help='Number of ensemble members (default: 16)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Master RNG seed. Omitted (default), torch is seeded at 42 and numpy '
                            'is left unseeded, which is the historical behaviour -- runs are NOT '
                            'reproducible, since the empirical Thompson draw uses np.random. Pass '
                            'an explicit seed to make a run repeatable; use one seed per replicate '
                            'so arms can be compared on matched seeds.')
    parser.add_argument('--cycles', type=int, default=50,
                       help='Number of active learning cycles to run (default: 50). The loop is '
                            'range(start_cycle, cycles), so on a resume this is the stop index, '
                            'not a count of remaining cycles.')

    # ---- Delta-GP, --model delta_gp ----
    # One GP over (composition, vacancy level) predicting the air-reactor Gibbs
    # curve; the fuel-reactor curve follows from the oxygen chemical potential
    # difference between the reactor conditions, so it is not modelled.
    parser.add_argument('--delta-gp-iters', type=int, default=1000,
                        help='Delta-GP training iterations (default: 1000)')
    parser.add_argument('--delta-gp-lr', type=float, default=0.02,
                        help='Delta-GP Adam learning rate (default: 0.02)')
    parser.add_argument('--delta-gp-patience', type=int, default=200,
                        help='Stop when the loss has not improved for this many iters')
    # ---- Delta-DE, --model delta_de ----
    # Defaults fall through to the shared --ensemble-* settings so the arm stays
    # comparable with --model ensemble; these only override them. The delta
    # target gives the network eight times as many rows per composition, so it
    # sees far more gradient steps per epoch than the 16-output ensemble does.
    parser.add_argument('--delta-de-epochs', type=int, default=None,
                        help='Override --ensemble-epochs for the Delta-DE arm')
    parser.add_argument('--delta-de-lr', type=float, default=None,
                        help='Override --ensemble-lr for the Delta-DE arm')

    parser.add_argument('--delta-gp-min-delta', type=float, default=1e-4,
                        help='Loss improvement below this counts as noise for early '
                             'stopping; too small and the fit never stops because the '
                             'marginal likelihood creeps down indefinitely')
    parser.add_argument('--delta-gp-draws', type=int, default=None,
                        help='Posterior draws used as ensemble members '
                             '(default: --ensemble-size)')
    parser.add_argument('--delta-gp-max-cholesky', type=int, default=32768,
                        help='Dense Cholesky threshold; must exceed compositions x '
                             'vacancy levels or GPyTorch falls back to CG, which '
                             'does not converge on this kernel')
    parser.add_argument('--delta-gp-chunk-size', type=int, default=256,
                        help='Candidates per prediction batch; each expands to '
                             'chunk * n_vacancies GP rows')

    # ---- Multi-output GP (ICM), --model mogp ----
    parser.add_argument('--mogp-rank', type=int, default=1,
                       help='ICM coregionalization rank (default: 1). Chosen on downstream OTC '
                            'error, not Gibbs RMSE: 5-fold CV gave OTC RMSE 0.315 wt%% / rho 0.859 '
                            'at rank 1, degrading monotonically to 0.439 / 0.811 at rank 5.')
    parser.add_argument('--mogp-iters', type=int, default=3000,
                       help='Adam iterations on the exact marginal log likelihood (default: 3000). '
                            'Matches the benchmark that selected rank 1, where the ICM needed '
                            '1150-1800 iterations to converge. Do not lower this: an '
                            'under-converged fit leaves the per-task noise flat near its '
                            'initialisation instead of recovering the vacancy ramp, which '
                            'over-smooths the landscape. ~35 ms/iter at N=850.')
    parser.add_argument('--mogp-lr', type=float, default=0.02,
                       help='Learning rate for MOGP hyperparameter fitting (default: 0.02)')
    parser.add_argument('--mogp-patience', type=int, default=500,
                       help='Early-stopping patience on the MLL loss (default: 500). Generous on '
                            'purpose -- the multitask MLL plateaus in long stretches before the '
                            'task noises separate.')
    parser.add_argument('--mogp-draws', type=int, default=None,
                       help='Posterior landscape draws per prediction, used as ensemble members '
                            'for acquisition (default: --ensemble-size)')
    parser.add_argument('--mogp-chunk-size', type=int, default=128,
                       help='Candidates per posterior batch (default: 128). Smaller is cheaper: '
                            'sampling cost scales as N * chunk^2 * T^3.')
    parser.add_argument('--mogp-max-cholesky', type=int, default=4096,
                       help='GPyTorch max_cholesky_size (default: 4096). Must exceed the number of '
                            'training points or GPyTorch silently falls back to CG, which does not '
                            'converge on this problem.')
    parser.add_argument('--quadratic-fit-otc', action='store_true',
                       help='Use quadratic fit for OTC (default: False)')
    parser.add_argument('--quadratic-fit-landscape-otc', action='store_true',
                       help='Use quadratic fit on predictions of OTC when in Gibbs landscape mode (default: False)')
    

    parser.add_argument('--bootstrap-fraction', type=lambda x: None if x.lower() == 'none' else float(x), default=None,
                       help='Bootstrap sampling fraction for ensemble training (default: 0.8, use "None" to train all members on full dataset)')
    parser.add_argument('--method', type=str, default='spinel',
                       help='Method/directory name for organizing outputs (default: spinel)')
    parser.add_argument('--max-element-count', type=int, default=None,
                       help='Maximum number of unique cations allowed in decoded samples.')
    parser.add_argument('-b', type=int, default=None,
                       help='Phonon batch size, A100 can do 2.')
    parser.add_argument('-t', type=int, default=None,
                       help='Relaxation threads, A100 can do 10.')
    parser.add_argument('--n-samples-total', type=int, default=25000,
                       help='Number of samples total, before selective sampling.')
    parser.add_argument('--use-validation', action='store_true',
                       help='Enable train/validation split for monitoring (default: False)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio (default: 0.15)')
    

    parser.add_argument('--overwrite', action='store_true',
                       help='Remove all existing cycles (except cycle0) before starting (default: False)')
    

    parser.add_argument('--val-split-mode', type=str, default='random',
                       choices=['random', 'sequential'],
                       help='Validation split mode: "random" (default) or "sequential" (first N%% samples)')
    parser.add_argument('--round-compositions', action='store_true',
                       help='Round decoded compositions to discrete atom counts (n*1/84) to eliminate discretization noise (default: False)')
    parser.add_argument('--predict-landscape', action='store_true',
                       help='Predict full Gibbs energy landscape instead of scalar OTC (default: False)')
    parser.add_argument('--clip-otc-physical', action='store_true',
                       help='Clip Thompson samples to per-composition physical OTC upper bound (default: False)')
    parser.add_argument('--auto-resume', action='store_true',
                       help='If cycles beyond cycle0 already exist, continue the most recent run '
                            'in this method directory instead of erroring. Makes a jobscript safe '
                            'to resubmit after a walltime kill or preemption. Default: False.')

    # Apply saved args as defaults (CLI args override them)
    if saved_args:
        # Convert underscores back to match argparse attribute names
        parser.set_defaults(**saved_args)
        
    return parser.parse_args(argv)
