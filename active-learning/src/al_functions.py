# Created 2024-11-08 by Joel Hildingsson
# This is just a file to store all the helper functions for active learning 
# used in the active_learning scripts so it is less bloated by functions

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch import kernels

from ase import Atoms
from ase.db import connect
from sklearn.mixture import GaussianMixture

import os
import sys
from pathlib import Path

# Determine the directory based on whether the environment is a notebook or a script
try:
    # This will raise an error in notebooks since __file__ doesn't exist
    directory = Path(__file__).resolve().parent
except NameError:
    # In Jupyter notebooks, the current working directory is used
    directory = Path().resolve()

# Add the src directory to the system path (two levels up from the current directory)
sys.path.append(str(directory.parent))


# Add paths to script

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

metal_path = os.path.join(project_root, 'metal')
sys.path.append(metal_path)
from generate_samples import generate_fake_samples_batch
from metal.applications.crystals.stability_factors import SpinelTolerance

def generate_candidate_samples(wae_model, latents, n_samples, device, use_rand=False, random_seed=0, batch_size=None):
    """
    Generate candidate samples by sampling from latent space (GMM) and decoding,
    or by generating random fake samples.
    
    Returns
    -------
    torch.Tensor
        Decoded compositions [n_samples, n_elements] on device
    """
    if use_rand:
        # Generate fake compositions directly
        samples = generate_fake_samples_batch(n_samples, device, 1)
        samples = samples[:, 0:-1]
        return samples
    
    # Fit GMM and sample latents
    gmm = GaussianMixture(n_components=1, random_state=random_seed)
    gmm.fit(latents)
    samples_np = gmm.sample(n_samples)[0]
    
    # Determine batch size
    if batch_size is None:
        batch_size = 2048 if device.type == 'cuda' else 512
        
    # Decode in batches
    decoded_samples = None
    samples_tensor = torch.from_numpy(samples_np).to(dtype=torch.float, device=device)
    
    with torch.no_grad():
        wae_model.eval()
        n_total = len(samples_tensor)
        batch_results = []
        
        for i in range(0, n_total, batch_size):
            batch = samples_tensor[i:i+batch_size]
            decoded = wae_model._decode(batch).detach()
            batch_results.append(decoded)
            
        decoded_samples = torch.cat(batch_results, dim=0)
        
    return decoded_samples


def round_compositions_to_atoms(compositions, n_A_elements, n_B_elements, elements=None, 
                                scaling_dictionary=None, natoms=84):
    """
    Round continuous compositions to discrete atom counts while preserving
    sublattice stoichiometry for spinel structures.
    
    Uses a greedy algorithm that:
    1. Floors all concentrations to get base atom counts
    2. Adds remaining atoms to elements with highest fractional remainders
    
    This eliminates discretization noise by ensuring compositions match the
    actual discrete atom counts that will be used in SQS structure generation.
    
    Parameters
    ----------
    compositions : torch.Tensor or np.ndarray
        [N, n_elements-1] compositions (excluding O)
    n_A_elements : int
        Number of A sublattice elements
    n_B_elements : int
        Number of B sublattice elements
    elements : list, optional
        List of element symbols. If provided with scaling_dictionary, will derive
        sublattice sizes from scaling factors instead of hardcoded values.
    scaling_dictionary : dict, optional
        Dictionary mapping element symbols to their sublattice scaling factors.
        If provided with elements, will derive sublattice sizes (e.g., 1/7 -> 12 atoms)
    natoms : int
        Total atoms in structure (default: 84 for spinel)
    
    Returns
    -------
    torch.Tensor or np.ndarray
        Rounded compositions (same type as input)
        
    Notes
    -----
    For spinel AB2O4 with 84 atoms:
    - A sublattice: 12 atoms (1/7 of structure)
    - B sublattice: 24 atoms (2/7 of structure)
    - O sublattice: 48 atoms (4/7 of structure, not included in input)
    """
    # Determine sublattice sizes
    if elements is not None and scaling_dictionary is not None:
        # Derive from scaling_dictionary
        # Get first A element and first B element to look up their scaling factors
        A_element = elements[0]  # First element is from A sublattice
        B_element = elements[n_A_elements]  # First B element
        
        A_scaling = scaling_dictionary.get(A_element)
        B_scaling = scaling_dictionary.get(B_element)
        
        if A_scaling is None or B_scaling is None:
            raise ValueError(f"Could not find scaling factors for {A_element} or {B_element} in scaling_dictionary")
        
        # Calculate atoms from scaling factors: scaling = n_atoms / natoms
        n_A = int(round(A_scaling * natoms))
        n_B = int(round(B_scaling * natoms))
        
        # Verify these are correct
        assert n_A > 0 and n_B > 0, f"Invalid sublattice sizes: n_A={n_A}, n_B={n_B}"
    else:
        # Use hardcoded spinel values as fallback
        n_A = natoms // 7      # 12 atoms
        n_B = 2 * natoms // 7  # 24 atoms
    
    # Convert to numpy if needed
    is_torch = isinstance(compositions, torch.Tensor)
    if is_torch:
        device = compositions.device
        comps_np = compositions.detach().cpu().numpy()
    else:
        comps_np = np.asarray(compositions)
    
    # Defensive assertion: verify dimensions
    expected_elements = n_A_elements + n_B_elements
    actual_elements = comps_np.shape[1] if len(comps_np.shape) > 1 else len(comps_np)
    assert actual_elements == expected_elements, \
        f"Expected {expected_elements} elements ({n_A_elements} A + {n_B_elements} B), got {actual_elements}"
    
    rounded = np.zeros_like(comps_np)
    
    for i in range(len(comps_np)):
        # Split into sublattices
        A_concs = comps_np[i, :n_A_elements]  # First n_A_elements
        B_concs = comps_np[i, n_A_elements:]  # Next n_B_elements
        
        # Round A sublattice to n_A atoms (returns sublattice-relative concentrations)
        A_rounded = _round_sublattice_greedy(A_concs, n_A)
        
        # Round B sublattice to n_B atoms (returns sublattice-relative concentrations)
        B_rounded = _round_sublattice_greedy(B_concs, n_B)
        
        # Combine
        rounded[i, :n_A_elements] = A_rounded
        rounded[i, n_A_elements:] = B_rounded
    
    # Convert back to torch if needed
    if is_torch:
        rounded = torch.from_numpy(rounded).float().to(device)
    
    return rounded


def _round_sublattice_greedy(concentrations, natoms_sublattice):
    """
    Round concentrations to integer atom counts that sum to natoms_sublattice.
    Uses greedy algorithm: floor all, then add 1 to highest remainders.
    
    Returns concentrations RELATIVE TO THE SUBLATTICE (not full structure),
    because the scaling to full structure happens later in the pipeline.
    
    Parameters
    ----------
    concentrations : np.ndarray
        [n_elements] concentrations for this sublattice (relative to sublattice, sums to ~1.0)
    natoms_sublattice : int
        Total atoms that must be allocated in this sublattice (e.g., 12 for A, 24 for B)
        
    Returns
    -------
    np.ndarray
        Rounded concentrations relative to SUBLATTICE (sum to 1.0)
        These will later be scaled by sublattice fraction (1/7 or 2/7) in the pipeline
    """
    # Defensive assertion: input should sum to approximately 1.0
    input_sum = concentrations.sum()
    assert abs(input_sum - 1.0) < 0.01, \
        f"Input concentrations must sum to ~1.0, got {input_sum}"
    
    # Defensive assertion: all concentrations should be non-negative
    assert np.all(concentrations >= 0), \
        f"All concentrations must be non-negative, got min={concentrations.min()}"
    
    # Calculate target atom counts for this sublattice
    target_counts = concentrations * natoms_sublattice
    
    # Floor all counts
    floor_counts = np.floor(target_counts).astype(int)
    remainders = target_counts - floor_counts
    
    # How many atoms still need to be allocated?
    missing = natoms_sublattice - floor_counts.sum()
    
    # Defensive assertion: missing should be non-negative and reasonable
    assert 0 <= missing <= len(concentrations), \
        f"Invalid missing atom count: {missing} (natoms={natoms_sublattice}, floor_sum={floor_counts.sum()})"
    
    # Add 1 to the elements with highest remainder
    if missing > 0:
        # Get indices of top 'missing' remainders
        top_indices = np.argsort(-remainders)[:missing]
        floor_counts[top_indices] += 1
    
    # Defensive assertion: total atoms should equal natoms_sublattice
    total_atoms = floor_counts.sum()
    assert total_atoms == natoms_sublattice, \
        f"Rounded atom counts must sum to {natoms_sublattice}, got {total_atoms}"
    
    # Convert to concentrations RELATIVE TO SUBLATTICE
    rounded_concs = floor_counts / natoms_sublattice
    
    # Defensive assertion: output should sum to exactly 1.0
    output_sum = rounded_concs.sum()
    assert abs(output_sum - 1.0) < 1e-8, \
        f"Output concentrations must sum to 1.0, got {output_sum}"
    
    # Defensive assertion: all values should be valid fractions (multiples of 1/natoms_sublattice)
    for i, val in enumerate(rounded_concs):
        atom_count = val * natoms_sublattice
        assert abs(atom_count - round(atom_count)) < 1e-8, \
            f"Element {i}: {val} × {natoms_sublattice} = {atom_count} is not an integer!"
    
    return rounded_concs


class GPRegressor(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood=None, device=None):
        # Determine device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # choose the standard observation model as required by exact GP inference
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # Constrain noise to be small (between 1e-6 and 1e-2)
            # This prevents GP from explaining everything with observation noise
            # likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Interval(1e-6, 1e-2))
            

        # Move likelihood to device
        likelihood = likelihood.to(device)

        # Move data to device
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        # initiate the superclass ExactGP to refresh the posterior
        super().__init__(train_x, train_y, likelihood)
        # store attributes
        input_dim = train_x.size(-1)  # Get the input dimension
        self.likelihood = likelihood
        self.mean = gpytorch.means.ConstantMean()
        self.kernel = kernels.ScaleKernel(kernels.MaternKernel(ard_num_dims=input_dim))

        # Move model components to device
        self.mean = self.mean.to(device)
        self.kernel = self.kernel.to(device)

        # compute the posterior distribution after conditioning on the
    
    def print_parameters(self, logger=None):
        """Print GP model parameters including kernel hyperparameters and noise."""
        lines = [
            "=" * 60,
            "GP Model Parameters",
            "=" * 60,
            f"Device: {self.device}",
            f"Training points: {self.train_inputs[0].shape[0]}",
            f"Input dimension: {self.train_inputs[0].shape[1]}",
            f"\nKernel: {self.kernel}",
            f"  Output scale: {self.kernel.outputscale.item():.6f}",
        ]

        lengthscales = self.kernel.base_kernel.lengthscale.squeeze().detach().cpu().numpy()
        if lengthscales.ndim == 0:
            lines.append(f"  Lengthscale: {lengthscales.item():.6f}")
        else:
            lines.append(f"  Lengthscales (ARD): {lengthscales}")

        lines.extend([
            f"\nLikelihood: {self.likelihood}",
            f"  Noise: {self.likelihood.noise.item():.6f}",
            f"  Noise constraint: {self.likelihood.noise_covar.raw_noise_constraint}",
            f"\nMean: {self.mean}",
            f"  Constant: {self.mean.constant.item():.6f}",
            "=" * 60
        ])

        if logger:
            for line in lines:
                logger.info(line)
        else:
            print("\n".join(lines))

    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.kernel(x)
        # return a posterior multivariate normal distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    # compute the marginal predictive distribution of y given x
    def predict(self, x):
        # set to evaluation mode
        self.eval()
        with torch.no_grad():
            # get posterior distribution p(f|x)
            pred = self(x)
            # convert posterior distribution p(f|x) to p(y|x)
            return self.likelihood(pred)

    def predict_decomposed(self, x):
        """Split the predictive uncertainty into its epistemic and total parts.

        `epistemic` is the std of the latent f -- the part that shrinks as data
        arrives -- while `total` adds the likelihood's learned noise sigma_n.
        Acquisition wants the former: sigma_n is irreducible, so it says nothing
        about where sampling would help. Mirrors ICMGPRegressor.predict_decomposed
        so GP and MOGP report uncertainty on the same footing.

        Returns
        -------
        (mean, epistemic_std, total_std) : tuple of torch.Tensor
        """
        self.eval()
        with torch.no_grad():
            latent = self(x)
            observed = self.likelihood(latent)
            return latent.mean, latent.stddev, observed.stddev


def train_GP_new(model, train_x, train_y, n_epochs, lr=0.0075, verbose=True, logger=None):
    # switch to model training mode
    model.train()

    # Ensure training data is on the same device as the model
    if hasattr(model, 'device'):
        train_x = train_x.to(model.device)
        train_y = train_y.to(model.device)

    # Use the adam optimizer
    training_parameters = model.parameters()
    optimizer = torch.optim.Adam(training_parameters, lr=lr)
    # initiate a GP model with Gaussian likelihood for MLL calculation
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    # start the training loop
    _loss = []
    for e in range(n_epochs):
        # clear gradients to prevent gradient accumulation
        optimizer.zero_grad()
        # get posterior distribution p(f|x)
        out = model(train_x)
        # get negated exact marginal log likelihood p(y|x)
        loss = -mll(out, train_y)
        # calculate gradients using autograd
        loss.backward()
        # perform gradient update
        optimizer.step()
        # print optimization log
        _loss.append(loss.item())
        if verbose or logger:
            if e % 50 == 0 or e == n_epochs - 1:
                lengthscale = model.kernel.base_kernel.lengthscale.squeeze(0).detach().cpu().numpy()
                msg = f"Epoch: {e}, loss: {loss.item():.3f}, lengthscale:{lengthscale}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
                # print(f"Loss: {loss.item():.3f}, lengthscale:{lengthscale}")
    return _loss


def _passes_stability_filters(row, elements, a_elements, b_elements,
                              scaling_dictionary, natoms):
    """Check if a candidate composition passes tolerance and charge neutrality filters.

    Parameters
    ----------
    row : pd.Series
        Row from a DataFrame containing element fractions.
    elements : list
        List of element symbols (excluding O/X).
    a_elements : list
        A-site element symbols.
    b_elements : list
        B-site element symbols.
    scaling_dictionary : dict
        Maps element symbols to their sublattice scaling factors
        (e.g. A-site: 1/7, B-site: 2/7, O: 4/7).
    natoms : int
        Total number of atoms in the structure.

    Returns
    -------
    bool
        True if the composition passes both tolerance and charge neutrality.
    """
    from metal.applications.crystals.tools import find_matching_atom_counts

    # Build chemical symbols using find_matching_atom_counts per sublattice,
    # matching the rounding used by build_db_random_configurations.
    cation_elements = set(a_elements) | set(b_elements)

    symbols = []
    for sublattice_elements in [a_elements, b_elements]:
        # Number of sites on this sublattice (scaling is per-sublattice, same for all elements)
        n_sublattice = int(round(scaling_dictionary[sublattice_elements[0]] * natoms))
        # Only include elements with non-zero concentration
        target = {}
        for el in sublattice_elements:
            frac = row.get(el, 0.0)
            if frac > 0 and n_sublattice > 0:
                target[el] = frac * natoms / n_sublattice
        if not target:
            continue
        try:
            counts = find_matching_atom_counts(n_sublattice, target)
        except (ValueError, IndexError):
            return False
        for el, count in counts.items():
            symbols.extend([el] * count)

    # Add anion atoms (elements in scaling_dictionary but not cations)
    for el, scaling in scaling_dictionary.items():
        if el not in cation_elements:
            n_anion = int(round(scaling * natoms))
            symbols.extend([el] * n_anion)

    if len(symbols) == 0:
        return False
    
    ############### Stability Factor Filtering ###############

    try:
        atoms = Atoms(symbols)
        print(f'Formula: {atoms.get_chemical_formula()}')
        st = SpinelTolerance(atoms, possible_a_elements=list(a_elements),
                             possible_b_elements=list(b_elements))
        return 0.77 <= st.spinel_tolerance <= 1.06 and st.charge_neutrality
    except Exception as e:
        print(f'Exception: {e}')
        return False


def _otc_physical_upper_bound(rounded_df, elements, natoms, config):
    """Per-composition physical ceiling on OTC, matched to the units of the ranking column.

    OTC is a mass fraction, so it cannot exceed what the structure is able to
    release: every one of the (n_vacancies_max - 1) oxygens leaving, over the
    formula mass. A candidate predicted above this is unphysical no matter what
    the surrogate says.

    Computed from the *rounded* compositions, since those are the discrete atom
    counts the SQS structures will actually be built from -- `rounded_df` holds
    sublattice-scaled fractions, so atom counts are simply value * natoms.

    Returns the bound in whatever space `method_list[0]` lives in:
    - landscape mode derives OTC arithmetically (al_functions.py:1935) and is
      already in physical mass fraction;
    - scalar mode ranks on normalized model output, so the bound is pushed
      through `config.forward_transform`, which knows whether the run used
      z-score (ensemble/MOGP) or min-max (GP/MC).

    Returns None if clipping is disabled or the run has no normalization stats.
    """
    from ase.data import atomic_masses, atomic_numbers

    mass_O = atomic_masses[atomic_numbers['O']]
    cation_elements = [el for el in elements if el != 'O']

    # Atom counts, then formula mass including the full oxygen sublattice.
    counts = rounded_df[cation_elements].values * natoms
    cation_mass = counts @ np.array(
        [atomic_masses[atomic_numbers[el]] for el in cation_elements]
    )
    n_O = round(config.scaling_dictionary['O'] * natoms)
    m_total = cation_mass + mass_O * n_O

    otc_max = mass_O * (config.n_vacancies_max - 1) / m_total   # physical mass fraction

    if config.predict_landscape or getattr(config, 'use_ceiling', False):
        return otc_max

    forward = getattr(config, 'forward_transform', None)
    if forward is None:
        return None
    return forward(otc_max, 'otc')


def export_top_unique(new_samples, elements, method_list, n_samples, sample_name, tol,
                      max_element_count=None, existing_df=None, logger=None,
                      a_elements=None, b_elements=None,
                      scaling_dictionary=None, natoms=None, config=None):
    """Take a df of all samples decoded and predicted, then export a csv containing the top n
    unique materials. Uniqueness in checked from elemental concetrations rounded to tol which
    should be set to 1/(n atoms in cell during METAL)

    Parameters
    ----------
    max_element_count : int, optional
        Maximum number of non-oxygen cations allowed. Filters applied before uniqueness check.
    existing_df : pd.DataFrame, optional
        DataFrame of existing samples in the database. If provided, samples with compositions
        matching existing entries (after rounding to tol) will be excluded.
    logger : logging.Logger, optional
        Logger to emit filtering statistics.
    a_elements : list, optional
        A-site element symbols for stability pre-screening.
    b_elements : list, optional
        B-site element symbols for stability pre-screening.
    scaling_dictionary : dict, optional
        Maps element symbols to sublattice scaling factors. Required with a/b_elements.
    natoms : int, optional
        Total atoms in structure. Required with a/b_elements.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the selected top unique samples
    """
    # Copy and round to tol
    rounded_df = new_samples.copy()
    rounded_df.loc[:, elements[0]:elements[-1]] = (rounded_df.loc[:, elements[0]:elements[-1]].div(tol).round().mul(tol))

    # --model ceiling: no surrogate, so the ranking column is the physical
    # ceiling itself (lightest fully oxidised 84-atom cell first)
    if config is not None and getattr(config, 'use_ceiling', False):
        new_samples = new_samples.copy()
        new_samples[method_list[0]] = _otc_physical_upper_bound(rounded_df, elements, natoms, config)
        rounded_df[method_list[0]] = new_samples[method_list[0]]

    # Clip the ranking column to the physical OTC ceiling before anything is
    # selected. Done here rather than in the acquisition function so that it
    # applies uniformly to every surrogate -- samples_GP, samples_mcdrop and
    # samples_ensemble all funnel through this one place, and all of them have
    # rounded compositions available by now. Only the value used for ranking is
    # touched; logged predictions and uncertainties keep their raw values.
    if config is not None and getattr(config, 'clip_otc_physical', False) and natoms is not None:
        otc_clip_max = _otc_physical_upper_bound(rounded_df, elements, natoms, config)
        if otc_clip_max is not None:
            prop = method_list[0]
            before = new_samples[prop].values
            n_clipped = int(np.sum(before > otc_clip_max))
            new_samples = new_samples.copy()
            new_samples[prop] = np.minimum(before, otc_clip_max)
            rounded_df[prop] = new_samples[prop]
            msg = (f'Physical OTC clip: {n_clipped}/{len(before)} candidates exceeded the '
                   f'per-composition ceiling and were capped.')
            if logger:
                logger.info(msg)
            else:
                print(msg)

    # Filter by max element count (before uniqueness check)
    if max_element_count is not None:
        cation_elements = [el for el in elements if el != 'O']
        n_cations = (rounded_df[cation_elements] > 0).sum(axis=1)
        element_count_mask = n_cations <= max_element_count

        before_filter = len(new_samples)
        new_samples = new_samples.loc[element_count_mask]
        rounded_df = rounded_df.loc[element_count_mask]
        removed_by_count = before_filter - len(new_samples)

        if removed_by_count > 0:
            msg = f'Removed {removed_by_count} candidates exceeding {max_element_count} cations.'
            if logger:
                logger.info(msg)
            else:
                print(msg)

    # Filter out samples that already exist in the database
    if existing_df is not None:
        print('Checking for duplicates.')
        # Round existing compositions the same way
        existing_rounded = existing_df[elements].copy()
        existing_rounded = existing_rounded.div(tol).round().mul(tol)

        # Create set of existing composition tuples for fast lookup
        existing_set = set(map(tuple, existing_rounded.values))

        # Find indices of new samples that are NOT in existing database
        new_comp_tuples = [tuple(row) for row in rounded_df[elements].values]
        not_in_db_mask = [comp not in existing_set for comp in new_comp_tuples]

        before_db_filter = len(new_samples)
        new_samples = new_samples.loc[not_in_db_mask]  # Keep original indices
        rounded_df = rounded_df.loc[not_in_db_mask]  # Keep original indices
        removed_by_db = before_db_filter - len(new_samples)

        if removed_by_db > 0:
            msg = f'Removed {removed_by_db} candidates already in database.'
            if logger:
                logger.info(msg)
            else:
                print(msg)

    # Identify duplicates in the rounded DataFrame (with regard to elements)
    unique_df = rounded_df.drop_duplicates(subset=elements)
    dropped_indices = rounded_df.index.difference(unique_df.index) # Get index of dropped duplicates
    filtered_df = new_samples.drop(index=dropped_indices) # Drop same indices in org database

    # Apply tolerance and charge neutrality pre-screening on top candidates only
    if a_elements is not None and b_elements is not None:
        import warnings
        import io
        import logging as _logging
        sorted_df = filtered_df.nlargest(len(filtered_df), method_list[0])
        passed = []
        checked = 0
        for _, row in sorted_df.iterrows():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Suppress stdout/stderr prints and logging from stability filters
                _prev_handlers = _logging.root.handlers[:]
                _logging.root.handlers = []
                _old_stdout, _old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    passes = _passes_stability_filters(
                            row, elements, a_elements, b_elements,
                            scaling_dictionary, natoms)
                finally:
                    sys.stdout, sys.stderr = _old_stdout, _old_stderr
                    _logging.root.handlers = _prev_handlers
                if passes:
                    passed.append(row)
                    if len(passed) >= n_samples:
                        break
            checked += 1
        msg = f'Screened {checked} candidates, {len(passed)} passed spinel tolerance/charge neutrality filters.'
        if logger:
            logger.info(msg)
        else:
            print(msg)
        highest_prop_df = pd.DataFrame(passed)
    else:
        highest_prop_df = filtered_df.nlargest(n_samples, method_list[0])

    # Export selected samples
    highest_prop_df.to_csv(sample_name, sep='\t', index=False)

    return highest_prop_df


# Take samples using thompson sampling from wae latent-space
def samples_mcdrop(model, sample_name, latents, iteration, method_list, config, log_path,
                   logger=None, cycle=None, existing_df=None):
    """
    Create a df which is exported to csv containing samples to be verified.

    Currently has options
    to enable or disable thompson sampling and scoring function.
    If thompson == False and use_score == False, this is a greedy aquisition function but using
    mc-dropout to increase predictions accuracy.

    How it works
    ------------
    Fit a GMM to latents and take a bunch of samples which is then conditioned on training data.
    Decode them using mc-dropout with 150 iteration for each sample and accumulate a sum which
    is used to get the mean, the mean in this case becomes the prediction.
    If thompson sampling is enabled also accumulate the sum of squares (ss) which is used for
    calculation of variance and std.
    the properties of each material and the concentration of each element is gathered in a list of lists
    which is then turned into a df.
    From this database it is possible to easily pick the n samples with highest values or apply
    a score function to pick samples.
    The subset of materials chosen to be verified are exported to a csv.

    Parameters
    ----------
    model : torch.nn.Module
        A trained pytorch WAE model which can be used to decode latent representations
        into compositions with predicted properties
    sample_name : str
        path where to save csv with samples example "cycle1/sample.csv"
    latents : np.ndarray
        numpy array with all datapoints compressed into latent space
        (see function 'get_latents' in ANN_functions)
    iteration : int
        iteration in active learning loop
    method_list : list
        list of properties used from ASE database, example ['otc','Sconf_R','bartel_tolerance']
    config : Config
        Configuration object containing all parameters (device, sampling strategy, elements, etc.)
    log_path : str
        Directory path for saving logs
    logger : logging.Logger, optional
        Optional logger to report filtering statistics.
    cycle : int, optional
        Current cycle number
    existing_df : pd.DataFrame, optional
        DataFrame of existing samples in the database to filter out duplicates
    """
    # Extract parameters from config
    sampling_arg = config.sampling
    device = config.device
    n_samples = config.default_samples_per_cycle
    elements = config.elements
    scaling_dictionary = config.scaling_dictionary
    tol = config.tol
    samples_total = config.n_samples_total
    max_element_count = config.max_element_count

    mc_iterations = 150

    # Determine optimal batch size based on device
    # GPU: larger batches for parallelism
    # CPU: smaller batches to avoid memory issues
    if device.type == 'cuda':
        batch_size = 2048  # Optimized for A100 (40GB/80GB VRAM)
        # Note: With 150 MC iterations, effective memory usage is batch_size * 150
        # If you encounter OOM errors, reduce to 1024 or 1536
    else:
        batch_size = 256  # Conservative for CPU

    # Always use GMM sampling (use_rand is always False in current usage)
    gmm = GaussianMixture(n_components=1, random_state=0)
    gmm.fit(latents)
    samples = gmm.sample(samples_total)[0]

    decoded_samples = []
    uncertainties_dict = {}  # Store uncertainties for all candidates
    predictions_dict = {}  # Store mean predictions for all candidates
    all_uncertainties = []  # Accumulate uncertainties across batches
    all_predictions = []  # Accumulate predictions across batches
    n_total_samples = len(samples)

    with torch.no_grad():
        model.train()

        # Process samples in batches
        for batch_start in range(0, n_total_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_total_samples)
            batch_samples = samples[batch_start:batch_end]

            # Convert batch to tensor and move to device once
            batch_tensor = torch.from_numpy(batch_samples).to(dtype=torch.float, device=device)
            current_batch_size = batch_tensor.shape[0]

            # Initialize accumulators on device (GPU or CPU)
            sum_of_outputs = None
            if sampling_arg == 'thompson' or sampling_arg == 'explore':
                ss_list = [None] * len(method_list)

            # MC dropout iterations
            for i in range(mc_iterations):
                # Generate outputs for entire batch
                output = model._decode_regression(batch_tensor)

                # Accumulate outputs on device (no CPU transfer)
                if sum_of_outputs is None:
                    sum_of_outputs = output
                else:
                    sum_of_outputs += output

                # Accumulate squared outputs for variance calculation
                if sampling_arg == 'thompson' or sampling_arg == 'explore':
                    for j in range(len(method_list)):
                        output_j_sq = output[:, j] ** 2
                        if ss_list[j] is None:
                            ss_list[j] = output_j_sq
                        else:
                            ss_list[j] += output_j_sq

            # Compute mean over all MC iterations
            mean_outputs = sum_of_outputs / mc_iterations

            # Store original predictions BEFORE sampling strategy modifies them
            batch_predictions = mean_outputs[:, 0].cpu().numpy()  # First property only (OTC)

            # Calculate and store uncertainties BEFORE applying sampling strategy
            batch_uncertainties = None
            if sampling_arg == 'thompson' or sampling_arg == 'explore':
                # Calculate uncertainties for all properties (over all MC iterations)
                std_list = []
                for j in range(len(method_list)):
                    variance_prop = (ss_list[j] / mc_iterations) - (mean_outputs[:, j] ** 2)
                    # Clamp variance to avoid numerical issues
                    variance_prop = torch.clamp(variance_prop, min=1e-10)
                    std_prop = torch.sqrt(variance_prop)
                    std_list.append(std_prop)

                # Store uncertainties (first property) for this batch
                batch_uncertainties = std_list[0].cpu().numpy()

                # NOW apply sampling strategy
                if sampling_arg == 'explore':
                    # Use variance as the acquisition function (only first property)
                    mean_outputs[:, 0] = std_list[0] ** 2
                elif sampling_arg == 'thompson':
                    # Thompson sampling: sample from predictive distribution
                    for j in range(len(method_list)):
                        sampled_values = torch.randn(current_batch_size, device=device) * std_list[j] + mean_outputs[:, j]
                        mean_outputs[:, j] = sampled_values

            # Accumulate predictions and uncertainties for this batch
            all_predictions.append(batch_predictions)
            if batch_uncertainties is not None:
                all_uncertainties.append(batch_uncertainties)

            # Transfer final results to CPU and convert to list
            batch_results = mean_outputs.cpu().numpy().tolist()
            decoded_samples.extend(batch_results)

    # Build predictions and uncertainties dictionaries (concatenate all batches)
    property_name = method_list[0] if len(method_list) > 0 else 'property0'

    if all_predictions:
        full_predictions = np.concatenate(all_predictions)
        predictions_dict[property_name] = full_predictions

    if all_uncertainties:
        # Concatenate uncertainties from all batches
        full_uncertainties = np.concatenate(all_uncertainties)
        # Store in dict with property name as key (using first property)
        uncertainties_dict[property_name] = full_uncertainties

    # Turn into df and do postprocessing
    col_names = elements[0:-1].copy()
    for i, method in enumerate(method_list):
        col_names.insert(i, method)

    new_samples = pd.DataFrame(decoded_samples, columns=col_names, dtype=float)

    for k, v in scaling_dictionary.items():
        if k in new_samples.columns:
            new_samples[k] = new_samples[k] * v

    new_samples['O'] = scaling_dictionary['O']

    # Export top unique samples and get the result
    selected_df = export_top_unique(new_samples, elements, method_list, n_samples, sample_name, tol,
                                    max_element_count=max_element_count, existing_df=existing_df, logger=logger,
                                    a_elements=config.A_elements, b_elements=config.B_elements,
                                    scaling_dictionary=config.scaling_dictionary,
                                    natoms=config.atoms_in_structure, config=config)

    # Log predictions and uncertainties of selected samples
    if predictions_dict:
        selected_indices = selected_df.index.values
        log_selected_sample_predictions(selected_indices, predictions_dict, uncertainties_dict,
                                       iteration, method_list, sampling_arg, log_path, cycle=cycle)

# Take samples using thompson sampling from wae latent-space
def samples_GP(wae_model, model_dict, sample_name, latents, iteration, method_list, config, log_path,
               logger=None, cycle=None, existing_df=None):
    """
    Create a df which is exported to csv containing samples to be verified.

    Currently has options
    to enable or disable thompson sampling and scoring function.
    If thompson == False and use_score == False, this is a greedy aquisition function but using
    mc-dropout to increase predictions accuracy.

    How it works
    ------------
    Based on previous function but adapted for when gp inference is used.

    Parameters
    ----------
    wae_model : torch.nn.Module
        a trained pytorch WAE model which can be used to decode latent representations into compositions
        with predicted propertes (see class 'WAE' in ANN_functions)
    model_dict : dict
        dictioanry including all the trained GPs, at the start only one is pressumed to be used.
        But the method should be general for when multiple are used
    sample_name : str
        path where to save csv with samples example "cycle1/sample.csv"
    latents : np.ndarray
        numpy array with all datapoints compressed into latent space
        (see function 'get_latents' in ANN_functions)
    iteration : int
        iteration in active learning loop
    method_list : list
        list of properties used from ASE database, example ['otc','Sconf_R','bartel_tolerance']
    config : Config
        Configuration object containing all parameters (device, sampling strategy, elements, etc.)
    log_path : str
        Directory path for saving logs
    logger : logging.Logger, optional
        Optional logger to report filtering statistics.
    cycle : int, optional
        Current cycle number
    existing_df : pd.DataFrame, optional
        DataFrame of existing samples in the database to filter out duplicates

    Work in progress:
    TODO Some parameteres set to static here should be outsourced"""
    # Extract parameters from config
    sampling_arg = config.sampling
    device = config.device
    n_samples = config.default_samples_per_cycle
    elements = config.elements
    scaling_dictionary = config.scaling_dictionary
    tol = config.tol
    samples_total = config.n_samples_total
    max_element_count = config.max_element_count
    round_compositions = config.round_compositions
    n_A_elements = len(config.A_elements)
    n_B_elements = len(config.B_elements)

    n_props = len(method_list)

    # Generate candidate samples (decoded compositions)
    # Always use GMM sampling (use_rand is always False in current usage)
    decoded_samples = generate_candidate_samples(
        wae_model, latents, samples_total, device,
        use_rand=False, random_seed=0
    )

    # Round compositions to discrete atom counts if requested
    if round_compositions:
        decoded_samples = round_compositions_to_atoms(
            decoded_samples, n_A_elements, n_B_elements,
            elements=elements, scaling_dictionary=scaling_dictionary
        )
        if logger:
            logger.info('Rounded compositions to discrete atom counts (n*1/84)')


    # Predict values of properties and apply strategy
    sample_list = [] # For each property, in this case just clas
    predictions_dict = {}  # Store mean predictions for logging selected samples
    uncertainties_dict = {}  # Store uncertainties for logging selected samples

    # --model ceiling has no GP: placeholder values, ranked in export_top_unique
    if model_dict is None:
        sample_list = [[0.0] * len(decoded_samples)] * n_props

    for i in range(n_props if model_dict is not None else 0):
        model = f'model{i}'

        # Move data to model's device and set model to eval mode
        model_dict[model].eval()

        
        # Use decoded compositions as GP input (original behavior)
        if hasattr(model_dict[model], 'device'):
            decoded_samples = decoded_samples.to(model_dict[model].device)
        gp_input = decoded_samples

        all_means, epistemic_stds, total_stds = model_dict[model].predict_decomposed(gp_input)

        # Acquisition uses the latent (epistemic) std. The observation noise
        # sigma_n is irreducible and homoscedastic, so including it would add
        # undirected randomness to the Thompson draw. This matches MC dropout, the
        # deep ensemble and the multi-output GP, which also sample epistemic only.
        all_stds = epistemic_stds

        # Log prediction statistics for all candidates
        property_name = method_list[i] if i < len(method_list) else f'model{i}'
        log_gp_statistics(all_means, epistemic_stds, total_stds, iteration, property_name, log_path)

        # Store predictions and uncertainties for later logging
        predictions_dict[property_name] = all_means.detach().cpu().numpy()
        uncertainties_dict[property_name] = all_stds.detach().cpu().numpy()

        if sampling_arg == 'explore':
            sample_list.append(all_stds.tolist())
        elif sampling_arg == 'thompson':
            sampled_values = torch.randn(len(gp_input), device=all_stds.device) * all_stds + all_means
            sample_list.append(sampled_values.tolist())
        else: # Pressumed to be greedy
            sample_list.append(all_means.tolist())

    # Insert property values to list of compositions before creating df
    new_samples = decoded_samples.tolist()
    for i,row in enumerate(new_samples):
        for j in range(n_props-1,-1,-1):
            row.insert(0,sample_list[j][i])

    # Create df and export to csv
    new_samples = pd.DataFrame(new_samples, columns = (method_list + elements[0:-1]), dtype = float)
    # Set the 'O' column to 4.0/7.0

    for k,v in scaling_dictionary.items():
        if k in new_samples.columns:
            new_samples[k] = new_samples[k] * v

    new_samples['O'] = scaling_dictionary['O']

    # Export top unique samples and get the result
    selected_df = export_top_unique(new_samples, elements, method_list, n_samples, sample_name, tol,
                                    max_element_count=max_element_count, existing_df=existing_df, logger=logger,
                                    a_elements=config.A_elements, b_elements=config.B_elements,
                                    scaling_dictionary=config.scaling_dictionary,
                                    natoms=config.atoms_in_structure, config=config)

    # Log predictions and uncertainties of selected samples
    if predictions_dict:
        selected_indices = selected_df.index.values
        log_selected_sample_predictions(selected_indices, predictions_dict, uncertainties_dict,
                                       iteration, method_list, sampling_arg, log_path, cycle=cycle)


def log_gp_statistics(means, epistemic_stds, total_stds, iteration, property_name, log_path):
    """
    Log GP prediction statistics (mean and uncertainty) to CSV.

    Both uncertainty components are recorded. `unc_*` stays the total predictive
    std so the column keeps its historical meaning, and the epistemic/aleatoric
    split is added alongside, matching what the ensemble and MOGP arms log.

    Parameters
    ----------
    means : torch.Tensor
        Posterior mean over all candidates
    epistemic_stds : torch.Tensor
        Std of the latent f -- the reducible part, and what acquisition uses
    total_stds : torch.Tensor
        Predictive std, i.e. epistemic plus the likelihood's sigma_n
    iteration : int
        Current active learning iteration
    property_name : str
        Name of the property being predicted
    log_path : str
        Directory path where statistics CSV will be saved
    """
    means = means.detach().cpu().numpy()
    stds = total_stds.detach().cpu().numpy()
    epi = epistemic_stds.detach().cpu().numpy()
    ale = np.sqrt(np.clip(stds ** 2 - epi ** 2, 1e-12, None))

    # Compute statistics
    stats = {
        'iteration': iteration,
        'property': property_name,
        'unc_min': float(np.min(stds)),
        'unc_mean': float(np.mean(stds)),
        'unc_max': float(np.max(stds)),
        'unc_std': float(np.std(stds)),
        'epistemic_min': float(np.min(epi)),
        'epistemic_mean': float(np.mean(epi)),
        'epistemic_max': float(np.max(epi)),
        'aleatoric_mean': float(np.mean(ale)),
        'mu_min': float(np.min(means)),
        'mu_mean': float(np.mean(means)),
        'mu_max': float(np.max(means)),
        'mu_std': float(np.std(means)),
        'n_points': int(stds.size)
    }

    # Write to CSV
    os.makedirs(log_path, exist_ok=True)
    stats_file = os.path.join(log_path, 'gp_stats.csv')
    stats_df = pd.DataFrame([stats])

    if os.path.exists(stats_file):
        stats_df.to_csv(stats_file, index=False, mode='a', header=False)
    else:
        stats_df.to_csv(stats_file, index=False)


def log_selected_sample_predictions(selected_indices, predictions_dict, uncertainties_dict,
                                    iteration, method_list, sampling_arg, log_path, cycle=None, **kwargs):
    """
    Log predictions (mean and uncertainty) for selected samples to CSV for later comparison.

    Saves individual predictions for the top N selected samples, allowing comparison
    with actual values once they're computed.

    Parameters
    ----------
    selected_indices : array-like
        Indices of selected samples in the candidate pool
    predictions_dict : dict
        Dictionary mapping property names to arrays of mean predictions for all candidates
    uncertainties_dict : dict
        Dictionary mapping property names to arrays of uncertainties for all candidates
    iteration : int
        Current active learning iteration
    method_list : list
        List of property names
    sampling_arg : str
        Acquisition strategy used ('thompson', 'greedy', 'explore')
    log_path : str
        Directory path where predictions CSV will be saved
    cycle : int, optional
        Current cycle number (if different from iteration)
    """
    os.makedirs(log_path, exist_ok=True)
    predictions_file = os.path.join(log_path, 'sample_predictions.csv')

    # Create a row for each selected sample
    rows = []
    for idx, orig_idx in enumerate(selected_indices):
        row = {
            'iteration': iteration,
            'cycle': cycle if cycle is not None else iteration,
            'sample_rank': idx,  # Rank within selected samples (0 = best)
            'sampling_strategy': sampling_arg,
        }
        for k, v in kwargs.items():
            row[k] = v[orig_idx]

        # Add predictions and uncertainties for each property
        for prop in method_list:
            if prop in predictions_dict:
                row[f'{prop}_pred'] = float(predictions_dict[prop][orig_idx])
            if prop in uncertainties_dict:
                row[f'{prop}_unc'] = float(uncertainties_dict[prop][orig_idx])

        rows.append(row)

    # Write to CSV
    df = pd.DataFrame(rows)

    if os.path.exists(predictions_file):
        df.to_csv(predictions_file, index=False, mode='a', header=False)
    else:
        df.to_csv(predictions_file, index=False)


def log_landscape_predictions_per_member(selected_indices, per_member_data, m_total_np,
                                         iteration, method_list, sampling_arg, log_path, cycle=None):
    """
    Log per-member predictions for landscape mode to separate CSV file.

    Creates one row per (selected_sample, ensemble_member) combination, allowing
    detailed analysis of ensemble member predictions.

    Parameters
    ----------
    selected_indices : array-like
        Indices of selected samples in the candidate pool
    per_member_data : dict
        Dictionary with per-member arrays:
        - 'otc_per_member': [M, N_total] - OTC predictions per member
        - 'delta_v_per_member': [M, N_total] - delta_v per member
        - 'gibbs_per_member_means': [M, N_total, D] - Gibbs predictions per member
        - 'gibbs_per_member_aleatoric': [M, N_total, D] - Aleatoric uncertainties per member
        - 'sampled_member_indices': [N_total] or None - Which member was sampled for Thompson sampling
    m_total_np : np.ndarray
        Mass array [N_total]
    iteration : int
        Current active learning iteration
    method_list : list
        List of property names (e.g., ['otc', 'gibbs_AR_0', 'gibbs_AR_1', ...])
    sampling_arg : str
        Acquisition strategy used ('thompson', 'greedy', 'explore')
    log_path : str
        Directory path where predictions CSV will be saved
    cycle : int, optional
        Current cycle number (if different from iteration)

    Notes
    -----
    The output CSV includes a 'was_sampled' column indicating which ensemble member
    was actually sampled for each candidate during Thompson sampling. This will be
    False for all members if not using Thompson sampling with empirical distribution.
    """
    if per_member_data is None:
        return  # No per-member data available (MC mode or n_samples_per_member > 1)

    os.makedirs(log_path, exist_ok=True)
    predictions_file = os.path.join(log_path, 'sample_predictions_per_member.csv')

    # Extract per-member arrays
    otc_per_member = per_member_data['otc_per_member']  # [M, N_total]
    delta_v_per_member = per_member_data['delta_v_per_member']  # [M, N_total]
    gibbs_per_member_means = per_member_data['gibbs_per_member_means']  # [M, N_total, D]
    gibbs_per_member_aleatoric = per_member_data['gibbs_per_member_aleatoric']  # [M, N_total, D]
    sampled_member_indices = per_member_data.get('sampled_member_indices', None)  # [N_total] or None

    n_members = otc_per_member.shape[0]
    n_gibbs_dims = gibbs_per_member_means.shape[2]

    # Get Gibbs property names (skip 'otc' which is first in method_list)
    gibbs_property_names = method_list[1:] if len(method_list) > 1 else []

    # Create a row for each (selected_sample, member) pair
    rows = []
    for rank_idx, orig_idx in enumerate(selected_indices):
        # Determine which member was sampled for this candidate (if Thompson sampling was used)
        sampled_member_for_this_idx = None
        if sampled_member_indices is not None:
            sampled_member_for_this_idx = int(sampled_member_indices[orig_idx])

        for member_idx in range(n_members):
            # Check if this member was the one sampled for Thompson sampling
            was_sampled = (sampled_member_for_this_idx == member_idx) if sampled_member_for_this_idx is not None else False

            row = {
                'iteration': iteration,
                'cycle': cycle if cycle is not None else iteration,
                'sample_rank': rank_idx,  # Rank within selected samples (0 = best)
                'member_idx': member_idx,  # Ensemble member index
                'sampling_strategy': sampling_arg,
                'was_sampled': was_sampled,  # Boolean: True if this member was sampled for Thompson sampling
                'otc_pred': float(otc_per_member[member_idx, orig_idx]),
                'delta_v': float(delta_v_per_member[member_idx, orig_idx]),
                'mass': float(m_total_np[orig_idx]),
            }

            # Add Gibbs energy predictions and aleatoric uncertainties
            for dim_idx in range(n_gibbs_dims):
                if dim_idx < len(gibbs_property_names):
                    prop_name = gibbs_property_names[dim_idx]
                else:
                    prop_name = f'gibbs_dim_{dim_idx}'

                row[f'{prop_name}_pred'] = float(gibbs_per_member_means[member_idx, orig_idx, dim_idx])
                row[f'{prop_name}_aleatoric_unc'] = float(gibbs_per_member_aleatoric[member_idx, orig_idx, dim_idx])

            rows.append(row)

    # Write to CSV
    df = pd.DataFrame(rows)

    if os.path.exists(predictions_file):
        df.to_csv(predictions_file, index=False, mode='a', header=False)
    else:
        df.to_csv(predictions_file, index=False)


def _log_message(logger, msg, level='info'):
    """Helper to log message or print if no logger."""
    if logger:
        getattr(logger, level)(msg)
    else:
        print(msg)


def _load_csv_or_warn(file_path, logger, sep=',', **kwargs):
    """Load CSV file or return None with warning if not found."""
    if not os.path.exists(file_path):
        _log_message(logger, f'File not found: {file_path}', 'warning')
        return None
    return pd.read_csv(file_path, sep=sep, **kwargs)


def _get_property_names(method_list, use_quadratic_fit):
    """Map property names for predictions CSV vs database."""
    ml_pred = method_list.copy()  # Property names in predictions CSV
    ml_calc = method_list.copy()  # Property names in database

    if use_quadratic_fit and 'otc' in ml_calc:
        idx = ml_calc.index('otc')
        ml_calc[idx] = 'otc_quadratic_fit'

    return ml_pred, ml_calc


def _normalize_composition(composition):
    """Normalize composition to sum to 1."""
    total = sum(composition.values())
    return {k: v/total for k, v in composition.items()} if total > 0 else composition


def _build_composition_index(db, elements, selection='filtered=False'):
    """Build a list of (normalized_comp, row) for fast matching."""
    rows = []
    for row in db.select(selection):
        db_comp = _normalize_composition(row.count_atoms())
        rows.append((db_comp, row))
    return rows


def _find_matching_composition(composition, elements, db_or_index, tol=0.01):
    """Find database row with matching composition within tolerance.

    db_or_index can be a Database (legacy) or a pre-built index from
    _build_composition_index (fast path).
    """
    if isinstance(db_or_index, list):
        index = db_or_index
    else:
        index = _build_composition_index(db_or_index, elements)

    for db_comp, row in index:
        if all(abs(composition.get(el, 0.0) - db_comp.get(el, 0.0)) <= tol for el in elements):
            return row
    return None


def _denormalize_prediction(pred_value, pred_unc_value, prop_pred, config):
    """Apply inverse transform to denormalize predictions."""
    # Per-member predictions are already denormalized
    if config.predict_landscape and hasattr(config, 'inverse_transform_gibbs'):
        if prop_pred == 'otc':
            # OTC already in physical units
            return pred_value, pred_unc_value
        else:
            # Gibbs energies need inverse transform
            inverse_fn = config.inverse_transform_gibbs
            pred_denorm = inverse_fn(pred_value, prop_pred, is_uncertainty=False)
            unc_denorm = inverse_fn(pred_unc_value, prop_pred, is_uncertainty=True) if not np.isnan(pred_unc_value) else np.nan
            return pred_denorm, unc_denorm
    elif config.inverse_transform is not None:
        pred_denorm = config.inverse_transform(pred_value, prop_pred, is_uncertainty=False)
        unc_denorm = config.inverse_transform(pred_unc_value, prop_pred, is_uncertainty=True) if not np.isnan(pred_unc_value) else np.nan
        return pred_denorm, unc_denorm
    else:
        return pred_value, pred_unc_value


def _extract_prediction_values(pred_row, prop_pred, use_per_member):
    """Extract prediction values and metadata from prediction row."""
    prop_ = 'otc' if 'otc' in prop_pred else prop_pred

    if use_per_member:
        return {
            'pred_value': pred_row.get(f'{prop_}_pred', np.nan),
            'pred_unc': pred_row.get(f'{prop_pred}_aleatoric_unc', np.nan),
            'member_idx': pred_row.get('member_idx', np.nan),
            'delta_v': pred_row.get('delta_v', np.nan),
            'mass': pred_row.get('mass', np.nan),
            'already_denormalized': True
        }
    else:
        return {
            'pred_value': pred_row.get(f'{prop_}_pred', np.nan),
            'pred_unc': pred_row.get(f'{prop_}_unc', np.nan),
            'member_idx': np.nan,
            'delta_v': np.nan,
            'mass': np.nan,
            'already_denormalized': False
        }


def compare_predictions_with_actuals(cycle, log_path, sample_file, db_name, elements, method_list,
                                    config, logger=None):
    """
    Compare predictions with actual computed values for a completed cycle.

    Loads predictions and compares with actual property values from the database.
    Results are saved to a comparison CSV.

    Parameters
    ----------
    cycle : int
        Cycle number to analyze
    log_path : str
        Directory path where predictions and comparison CSVs are saved
    sample_file : str
        Path to the sample.csv file for this cycle
    db_name : str
        Path to the training database containing actual values
    elements : list
        List of element symbols
    method_list : list
        List of property names (e.g., ['otc'])
    config : Config
        Configuration object (uses config.inverse_transform for denormalization)
    logger : logging.Logger, optional
        Logger for info/warning messages
    """
    import os
    import pandas as pd
    from ase.db import connect

    # Determine which predictions file to use
    per_member_file = os.path.join(log_path, 'sample_predictions_per_member.csv')
    ensemble_file = os.path.join(log_path, 'sample_predictions.csv')
    comparison_file = os.path.join(log_path, 'prediction_comparison.csv')

    use_per_member = os.path.exists(per_member_file)
    predictions_file = per_member_file if use_per_member else ensemble_file

    if use_per_member:
        _log_message(logger, f'Using per-member predictions (sampled member) for cycle {cycle}')

    # Load data files
    all_predictions = _load_csv_or_warn(predictions_file, logger)
    samples = _load_csv_or_warn(sample_file, logger, sep='\t')

    if all_predictions is None or samples is None:
        return

    if not os.path.exists(db_name):
        _log_message(logger, f'Training database not found: {db_name}', 'warning')
        return

    db = connect(db_name)
    comp_index = _build_composition_index(db, elements)

    # Filter predictions for this cycle
    if use_per_member:
        cycle_predictions = all_predictions[
            (all_predictions['cycle'] == cycle) & (all_predictions['was_sampled'] == True)
        ].copy()
    else:
        cycle_predictions = all_predictions[all_predictions['cycle'] == cycle].copy()

    if len(cycle_predictions) == 0:
        _log_message(logger, f'No predictions found for cycle {cycle}', 'warning')
        return

    # Get property name mappings
    ml_pred, ml_calc = _get_property_names(method_list, config.quadratic_fit_otc)

    # Process each prediction
    comparison_rows = []
    for _, pred_row in cycle_predictions.iterrows():
        rank = int(pred_row['sample_rank'])

        if rank >= len(samples):
            _log_message(logger, f'Sample rank {rank} exceeds number of samples ({len(samples)})', 'warning')
            continue

        # Get and normalize composition from sample file
        sample = samples.iloc[rank]
        composition = _normalize_composition({el: sample[el] for el in elements if el in sample})

        # Find matching composition in database
        matched_row = _find_matching_composition(composition, elements, comp_index)

        if matched_row is None:
            _log_message(logger, f'Could not find matching composition in database for sample rank {rank}', 'warning')
            continue

        # Extract predictions and actual values for each property
        for i, prop_pred in enumerate(ml_pred):
            prop_calc = ml_calc[i]

            if not hasattr(matched_row, prop_calc):
                continue

            actual_value = getattr(matched_row, prop_calc)
            pred_values = _extract_prediction_values(pred_row, prop_pred, use_per_member)

            # Denormalize predictions if needed
            if pred_values['already_denormalized']:
                pred_denorm = pred_values['pred_value']
                unc_denorm = pred_values['pred_unc']
            else:
                pred_denorm, unc_denorm = _denormalize_prediction(
                    pred_values['pred_value'], pred_values['pred_unc'], prop_pred, config
                )

            # Build comparison row
            comp_row = {
                'cycle': cycle,
                'iteration': pred_row['iteration'],
                'sample_rank': rank,
                'sampling_strategy': pred_row['sampling_strategy'],
                'prediction_source': 'sampled_member' if use_per_member else 'ensemble_mean',
                'member_idx': pred_values['member_idx'],
                'property': prop_calc,
                'predicted_mean': pred_denorm,
                'predicted_uncertainty': unc_denorm,
                'actual_value': actual_value,
                'error': pred_denorm - actual_value,
                'abs_error': abs(pred_denorm - actual_value),
            }

            # Add optional fields
            if use_per_member:
                comp_row['delta_v'] = pred_values['delta_v']
                comp_row['mass'] = pred_values['mass']

            # Add composition
            for el in elements:
                comp_row[f'comp_{el}'] = composition.get(el, 0.0)

            comparison_rows.append(comp_row)

    # Save results
    if comparison_rows:
        df = pd.DataFrame(comparison_rows)
        mode = 'a' if os.path.exists(comparison_file) else 'w'
        df.to_csv(comparison_file, index=False, mode=mode, header=(mode == 'w'))
        _log_message(logger, f'Saved {len(comparison_rows)} prediction comparisons for cycle {cycle}')
    else:
        _log_message(logger, f'No matching samples found for cycle {cycle}', 'warning')


def create_df(method_list, db_name, elements, use_quadratic_fit_otc=False, selection=None):
    """Read a ASE database of materials and properties to return a df where the first columns
    correspond to each of the properties and the rest correspond to an element present each.

    Parameters
    ----------
    method_list
        list of properties used for example ['otc','Sconf_R','bartel_tolerance'], NOTE
         names in method_list needs to correspond to row names in the ASE database
    db_name
        path to database to read example 'training.db'
    elements
        list of elements present in dataset, example ['Ca', 'Ba', 'Mn', 'Ti']
    use_quadratic_fit_otc : bool, optional
        If True, load 'otc_quadratic_fit' instead of 'otc' from database (default: False)
    selection : str, optional
        ASE database selection string (e.g., 'cycle<=2,filtered=False'). If None, uses 'filtered=False'


    """
    # Load database
    n_properties = len(method_list)
    db = connect(db_name)

    # Map method_list to actual database column names
    # If quadratic fit is requested, replace 'otc' with 'otc_quadratic_fit'
    db_method_list = method_list.copy()
    if use_quadratic_fit_otc and 'otc' in method_list:
        otc_idx = method_list.index('otc')
        db_method_list[otc_idx] = 'otc_quadratic_fit'

    # Query database ONCE and cache all rows (much faster than querying per property)
    # Build query string to ensure all properties exist
    if selection is not None:
        query_str = selection + ',' + ','.join(db_method_list)
    else:
        query_str = 'filtered=False,' + ','.join(db_method_list)
    all_rows = list(db.select(query_str))

    # Extract atom counts and concentrations from cached rows
    all_counts = [row.count_atoms() for row in all_rows]
    all_concs = [{k: v / sum(counts.values()) for k, v in counts.items()} for counts in all_counts]
    # all_concs is a list of dictionaries, each dictionary has elements as keys to concentrations
    # for each element in the material

    # Dictionary containing a list for each property given in method_list
    prop_dict = {}
    # Grab the object from each method name from cached rows
    # Use db_method_list for actual database column names
    # Use method_list for output column names (so 'otc' is always the column name)
    col_names = elements.copy()
    for i, (method, db_method) in enumerate(zip(method_list, db_method_list)):
        temp_list = [getattr(row, db_method) for row in all_rows]
        prop_dict[method] = temp_list  # Store with output name
        col_names.insert(i, method)  # Use output name for column

    # map each element to a index and store in dict
    db2vec = {}
    for i, element in enumerate(elements):
        db2vec[element] = i
    print(db2vec)

    # Create list to be used for df
    pre_df = []
    for i, conc_dict in enumerate(all_concs):
        new_row = [0]*(len(elements)+n_properties)#+1 originally changed to 3
        # Grab properties using dictionary of 
        for j, method in enumerate(method_list):
            new_row[j] = prop_dict[method][i]

        # Grab concentrations
        for key in conc_dict:
            new_row[db2vec[key]+n_properties] = conc_dict[key]
        pre_df.append(new_row)

    df = pd.DataFrame(pre_df, columns =col_names, dtype = float)
    return df


def _apply_acquisition_strategy(means, stds, sampling_arg, device, distribution=None):
    """
    Apply acquisition strategy (thompson, explore, greedy) to predictions.

    Parameters
    ----------
    means : torch.Tensor
        Predicted mean values [N]
    stds : torch.Tensor
        Predicted standard deviations [N]
    sampling_arg : str
        Acquisition strategy ('thompson', 'explore', 'greedy')
    device : torch.device
        Device for computation
    distribution : np.ndarray, optional
        Full empirical distribution [n_samples, N]. If provided for Thompson sampling,
        directly samples from empirical distribution instead of assuming Gaussian.
    Notes
    -----
    The physical OTC ceiling is not applied here. It is applied in
    `export_top_unique`, which every surrogate reaches, so it bounds the ranking
    column for GP, MC dropout, the deep ensemble and the MOGP alike.

    Returns
    -------
    tuple
        (values, sampled_member_indices) where:
        - values: list of values to use for ranking samples
        - sampled_member_indices: np.ndarray or None, indices of sampled members [N] for Thompson sampling
    """
    if sampling_arg == 'explore':
        # Use uncertainty for exploration
        return stds.tolist(), None
    elif sampling_arg == 'thompson':
        if distribution is not None:
            # Empirical Thompson sampling: randomly pick one sample from the empirical distribution
            # distribution shape: [n_ensemble_samples, n_candidates]
            if distribution.ndim != 2:
                raise ValueError(
                    f'empirical Thompson needs a 2-D [n_samples, n_candidates] '
                    f'distribution, got shape {distribution.shape}')
            n_samples, n_candidates = distribution.shape
            # A transposed distribution still unpacks into two names and would
            # silently rank on the wrong axis, so the candidate axis is checked
            # against the means it must line up with.
            if n_candidates != len(means):
                raise ValueError(
                    f'empirical Thompson distribution has {n_candidates} candidates '
                    f'on axis 1 but {len(means)} means were given; expected '
                    f'[n_samples, n_candidates] -- is it transposed?')
            # For each candidate, randomly select one of the ensemble samples
            random_indices = np.random.randint(0, n_samples, size=n_candidates)
            sampled_values = distribution[random_indices, np.arange(n_candidates)]
            return sampled_values.tolist(), random_indices
        else:
            # Parametric Thompson sampling: assume Gaussian N(mean, std)
            sampled_values = torch.randn(len(means), device=device) * stds + means
            return sampled_values.tolist(), None
    else:  # greedy
        return means.tolist(), None


def _calculate_composition_masses(decoded_samples, elements, n_A_elements, n_B_elements,
                                  scaling_dictionary, device, config):
    """
    Calculate mean atomic mass for each composition.

    Parameters
    ----------
    decoded_samples : torch.Tensor
        Decoded compositions [N, n_elements-1] (excluding O)
    elements : list
        List of element symbols
    n_A_elements : int
        Number of A sublattice elements
    n_B_elements : int
        Number of B sublattice elements
    scaling_dictionary : dict
        Sublattice scaling factors
    device : torch.device
        Device for computation

    Returns
    -------
    torch.Tensor
        Atomic mass for each composition [N]
    """
    from ase.data import atomic_masses, atomic_numbers

    # Create mass dictionary (excluding oxygen)
    mass_dict = {el: atomic_masses[atomic_numbers[el]] for el in elements[:-1]}
    mass_O = atomic_masses[atomic_numbers['O']]

    def get_sublattice_mass(compositions, sublattice_elements):
        masses = torch.tensor([mass_dict[el] for el in sublattice_elements],
                            device=device, dtype=torch.float32)
        return torch.matmul(compositions, masses)

    # Split compositions by sublattice
    A_elements_list = elements[:n_A_elements]
    B_elements_list = elements[n_A_elements:n_A_elements+n_B_elements]

    A_scaling = scaling_dictionary[A_elements_list[0]] * config.atoms_in_structure
    B_scaling = scaling_dictionary[B_elements_list[0]] * config.atoms_in_structure

    A_compositions = torch.round(decoded_samples[:, :n_A_elements] * A_scaling)
    B_compositions = torch.round(decoded_samples[:, n_A_elements:n_A_elements+n_B_elements] * B_scaling)

    # Verify stoichiometry: A + B atoms should equal total - O atoms
    expected_cation_atoms = config.atoms_in_structure - round(scaling_dictionary['O'] * config.atoms_in_structure)
    total_cations = (A_compositions.sum(dim=1) + B_compositions.sum(dim=1))
    assert torch.allclose(total_cations, torch.tensor(expected_cation_atoms, dtype=torch.float32, device=device)), \
        f"Stoichiometry check failed: A+B atoms should sum to {expected_cation_atoms}, got {total_cations.unique()}"

    # Calculate masses
    m_A = get_sublattice_mass(A_compositions, A_elements_list)
    m_B = get_sublattice_mass(B_compositions, B_elements_list)
    m_o = mass_O * round(scaling_dictionary['O'] * config.atoms_in_structure)

    return m_A + m_B + m_o


def _fit_quadratic_and_find_minimum(energies, vacancy_points):
    """
    Vectorized quadratic fitting to find optimal vacancy levels.

    For each sample, fits G(v) = a*v² + b*v + c and finds argmin.

    Parameters
    ----------
    energies : np.ndarray
        Gibbs energies [n_samples, n_vacancies_max]
    vacancy_points : np.ndarray
        Vacancy levels [n_vacancies_max], e.g., [0, 1, 2, ..., 7]

    Returns
    -------
    np.ndarray
        Optimal vacancy levels [n_samples]
    """
    n_samples = energies.shape[0]
    min_vac = vacancy_points[0]
    max_vac = vacancy_points[-1]
    v_opt = np.zeros(n_samples)

    # Fit each sample individually using np.polyfit
    for i in range(n_samples):
        # np.polyfit returns [a, b, c] for a*x^2 + b*x + c
        coeffs = np.polyfit(vacancy_points, energies[i], 2)

        a = coeffs[0]  # Quadratic coefficient
        b = coeffs[1]  # Linear coefficient

        if abs(a) < 1e-10:  # Nearly linear
            v_opt[i] = min_vac if b > 0 else max_vac
        elif a < 0:  # Parabola opens downward - minimum at boundary
            val_min = np.polyval(coeffs, min_vac)
            val_max = np.polyval(coeffs, max_vac)
            v_opt[i] = min_vac if val_min < val_max else max_vac
        else:  # Parabola opens upward - minimum at critical point
            v_min = -b / (2 * a)
            v_opt[i] = np.clip(v_min, min_vac, max_vac)

    return v_opt


def _predict_otc_from_landscape(ensemble, decoded_samples, config, iteration=None, log_path=None, logger=None):
    """
    Derive OTC from predicted Gibbs energy landscape.

    Two modes:
    1. MC sampling (default): Sample from mixture of Gaussians
    2. Quadratic fitting (if config.quadratic_fit_landscape_otc):
       Fit quadratic to each member's predictions, then aggregate

    Parameters
    ----------
    ensemble : DeepEnsemble
        Trained ensemble model
    decoded_samples : torch.Tensor
        Decoded compositions [N, n_elements-1]
    config : Config
        Configuration object
    logger : logging.Logger, optional
        Logger for messages

    Returns
    -------
    tuple
        (otc_mean, otc_std, gibbs_means, gibbs_stds, otc_distribution, delta_v_array, m_total_np,
         per_member_data) where:
        - otc_mean: OTC predictions [N]
        - otc_std: OTC uncertainties [N]
        - gibbs_means: Gibbs energy predictions [N, n_gibbs_dims]
        - gibbs_stds: Gibbs energy uncertainties [N, n_gibbs_dims]
        - otc_distribution: Full OTC distribution [n_samples, N] or None (only for quadratic fit mode)
        - delta_v_array: Delta-v array [M, N] or None
        - m_total_np: Mass array [N]
        - per_member_data: Dict with per-member data or None (only for quadratic fit, n_samples=1)
            {
                'otc_per_member': [M, N],
                'delta_v_per_member': [M, N],
                'gibbs_per_member_means': [M, N, D],
                'gibbs_per_member_aleatoric': [M, N, D]
            }
    """
    # Move data to ensemble device
    if hasattr(ensemble, 'device'):
        decoded_samples = decoded_samples.to(ensemble.device)

    # Get ensemble predictions
    predictions = ensemble.predict(decoded_samples)
    individual_means = predictions['individual_means']  # [N, D, M]
    individual_log_vars = predictions['individual_log_vars']  # [N, D, M]
    
    # Log ensemble statistics if log_path is provided
    if log_path is not None and iteration is not None:
        log_ensemble_statistics(predictions, iteration, 'gibbs_energy', log_path)

    # Extract Gibbs energy predictions and uncertainties
    gibbs_means = predictions['mean']  # [N, D] - ensemble mean for each Gibbs energy
    gibbs_stds = predictions['total_std']  # [N, D] - total uncertainty for each Gibbs energy

    n_candidates = len(decoded_samples)
    n_members = ensemble.n_members
    n_gibbs_dims = individual_means.shape[1]  # Get actual dimension from predictions
    n_vacancies_max = n_gibbs_dims // 2  # Vacancy steps per reactor

    # Choose OTC derivation method
    use_quadratic_fit = getattr(config, 'quadratic_fit_landscape_otc', False)    
    
    from ase.data import atomic_masses, atomic_numbers
    mass_O = atomic_masses[atomic_numbers['O']]

    # Calculate mean atomic mass for each composition
    m_total = _calculate_composition_masses(
        decoded_samples, config.elements, len(config.A_elements),
        len(config.B_elements), config.scaling_dictionary, ensemble.device,
        config=config
    )
    m_total_np = m_total.detach().cpu().numpy()

    if use_quadratic_fit:
        # ==================================================================
        # QUADRATIC FITTING APPROACH WITH UNCERTAINTY SAMPLING
        # n_samples_per_member = 1 for sampling from member means only
        # ==================================================================
        if logger:
            logger.info('Deriving OTC via hybrid sampling + quadratic fitting...')

        vacancy_points = np.arange(n_vacancies_max)

        # Hybrid approach: sample from predictive distribution + quadratic fitting
        # Number of samples to draw from each member's distribution
        n_samples_per_member = getattr(config, 'ensemble_otc_quadratic_fit_samples', 1)  
        if n_samples_per_member == 1 and getattr(config, 'ensemble_otc_quadratic_fit_member_mean', True):
            if logger:
                logger.info('Using member mean predictions directly for quadratic fitting (no noise).')

        # Storage for OTC from all samples (members × samples_per_member)
        otc_all_samples = []
        delta_v_all_samples = []

        # Storage for per-member data (only when n_samples_per_member == 1)
        per_member_otc_list = []
        per_member_delta_v_list = []
        per_member_gibbs_means_list = []
        per_member_gibbs_aleatoric_list = []

        # Get temperature for calibration if available
        temperature = getattr(ensemble, 'temperature', 1.0)

        # Process each ensemble member
        for member_idx in range(n_members):
            # Get predictions for this member: [N, D]
            member_means = individual_means[:, :, member_idx]  # Keep on device
            member_log_vars = individual_log_vars[:, :, member_idx]  # Keep on device

            # Compute standard deviations (with temperature scaling)
            member_stds = temperature * torch.exp(0.5 * member_log_vars)

            # Sample from the member's predictive distribution
            for _ in range(n_samples_per_member):
                # Sample one curve: G ~ N(mean, std)
                if n_samples_per_member == 1 and getattr(config, 'ensemble_otc_quadratic_fit_member_mean', True):
                    # Use mean prediction directly (no noise)
                    sampled_gibbs = member_means  # [N, D]
                else:
                    noise = torch.randn_like(member_means)
                    sampled_gibbs = member_means + member_stds * noise  # [N, D]

                # Move to CPU and numpy for quadratic fitting
                sampled_gibbs_np = sampled_gibbs.detach().cpu().numpy()

                # Denormalize Gibbs energies if needed
                sampled_gibbs_denorm_np = sampled_gibbs_np.copy()
                if hasattr(config, 'inverse_transform_gibbs') and config.inverse_transform_gibbs is not None:
                    for dim_idx in range(n_gibbs_dims):
                        gibbs_property_name = f'gibbs_{"AR" if dim_idx < n_vacancies_max else "FR"}_{dim_idx % n_vacancies_max}'
                        sampled_gibbs_denorm_np[:, dim_idx] = config.inverse_transform_gibbs(
                            sampled_gibbs_np[:, dim_idx], gibbs_property_name, is_uncertainty=False
                        )

                # Split into AR and FR: [N, n_vacancies_max]
                ar_energies = sampled_gibbs_denorm_np[:, :n_vacancies_max]
                fr_energies = sampled_gibbs_denorm_np[:, n_vacancies_max:]

                # Fit quadratics and find optimal vacancies
                v_ar_opt = _fit_quadratic_and_find_minimum(ar_energies, vacancy_points)
                v_fr_opt = _fit_quadratic_and_find_minimum(fr_energies, vacancy_points)

                # Compute delta_v
                delta_v = v_fr_opt - v_ar_opt
                delta_v_all_samples.append(delta_v)

                # Convert to OTC
                otc_sample = (mass_O * delta_v) / (m_total_np)

                otc_all_samples.append(otc_sample)

                # Store per-member data (only when n_samples_per_member == 1)
                if n_samples_per_member == 1:
                    per_member_otc_list.append(otc_sample)
                    per_member_delta_v_list.append(delta_v)
                    # Store denormalized Gibbs means for this member
                    per_member_gibbs_means_list.append(sampled_gibbs_denorm_np)
                    # Compute aleatoric uncertainty (std from log_vars, denormalized)
                    member_aleatoric = member_stds.detach().cpu().numpy()  # [N, D]
                    # Denormalize aleatoric uncertainty if needed
                    if hasattr(config, 'inverse_transform_gibbs') and config.inverse_transform_gibbs is not None:
                        for dim_idx in range(n_gibbs_dims):
                            gibbs_property_name = f'gibbs_{"AR" if dim_idx < n_vacancies_max else "FR"}_{dim_idx % n_vacancies_max}'
                            member_aleatoric[:, dim_idx] = config.inverse_transform_gibbs(
                                member_aleatoric[:, dim_idx], gibbs_property_name, is_uncertainty=True
                            )
                    per_member_gibbs_aleatoric_list.append(member_aleatoric)

        # Stack all samples: [n_members * n_samples_per_member, N]
        otc_array = np.stack(otc_all_samples, axis=0)  # [M*S, N]

        if n_samples_per_member == 1:
            delta_v_array = np.stack(delta_v_all_samples, axis=0)  # [M*S, N]
        else:
            delta_v_array = None  # Not stored if multiple samples per member

        # Compute statistics across all samples
        otc_mean = torch.from_numpy(otc_array.mean(axis=0)).to(ensemble.device)  # [N]
        otc_std = torch.from_numpy(otc_array.std(axis=0)).to(ensemble.device)  # [N]

        if logger:
            logger.info(f'Quadratic fitting with sampling: {n_members} members × {n_samples_per_member} samples/member')
            logger.info(f'OTC mean range [{otc_mean.min():.4f}, {otc_mean.max():.4f}]')
            logger.info(f'OTC std mean {otc_std.mean():.4f}, range [{otc_std.min():.4f}, {otc_std.max():.4f}]')

        # Store the full distribution for later analysis
        otc_distribution = otc_array  # [M*S, N] where M=n_members, S=n_samples_per_member

        # Create per-member data dictionary (only when n_samples_per_member == 1)
        if n_samples_per_member == 1:
            per_member_data = {
                'otc_per_member': np.stack(per_member_otc_list, axis=0),  # [M, N]
                'delta_v_per_member': np.stack(per_member_delta_v_list, axis=0),  # [M, N]
                'gibbs_per_member_means': np.stack(per_member_gibbs_means_list, axis=0),  # [M, N, D]
                'gibbs_per_member_aleatoric': np.stack(per_member_gibbs_aleatoric_list, axis=0)  # [M, N, D]
            }
            if logger:
                logger.info(f'Per-member data shapes: OTC {per_member_data["otc_per_member"].shape}, '
                           f'Gibbs {per_member_data["gibbs_per_member_means"].shape}')
        else:
            per_member_data = None
            
    # Log OTC statistics if log_path is not provided
    if log_path is not None and iteration is not None:
        # Construct a pseudo-predictions dict to reuse the logging function
        # We treat the sample std dev as the 'total' and 'epistemic' uncertainty for logging purposes
        otc_predictions = {
            'mean': otc_mean,
            'total_std': otc_std,
            'aleatoric_std': torch.zeros_like(otc_std),  # Placeholder
            'epistemic_std': otc_std,  # Treat derived variance as primarily epistemic
        }
        log_ensemble_statistics(otc_predictions, iteration, 'otc_derived', log_path)

    else:
        # ==================================================================
        # MONTE CARLO SAMPLING APPROACH (original)
        # ==================================================================
        if logger:
            logger.info('Predicting Gibbs energy landscape and deriving OTC with MC sampling...')

        # Monte Carlo sampling parameters
        n_mc_draws = 100

        # 1. Randomly select ensemble member for each MC draw (Mixture of Gaussians)
        member_indices = torch.randint(0, n_members, (n_candidates, n_mc_draws),
                                      device=ensemble.device)

        # 2. Gather mean and variance for selected members
        indices_expanded = member_indices.unsqueeze(1).expand(-1, n_gibbs_dims, -1)
        selected_means = torch.gather(individual_means, 2, indices_expanded)
        selected_log_vars = torch.gather(individual_log_vars, 2, indices_expanded)

        # Transpose to [N, n_mc_draws, D]
        selected_means = selected_means.permute(0, 2, 1)
        selected_log_vars = selected_log_vars.permute(0, 2, 1)

        # 3. Sample from the selected Gaussian components
        temperature = getattr(ensemble, 'temperature', 1.0)
        selected_stds = temperature * torch.exp(0.5 * selected_log_vars)
        noise = torch.randn(n_candidates, n_mc_draws, n_gibbs_dims, device=ensemble.device)
        draws = selected_means + selected_stds * noise

        # 4. Split into AR and FR landscapes and find minimum energy vacancies
        ar_energies = draws[:, :, :n_vacancies_max]
        fr_energies = draws[:, :, n_vacancies_max:]
        ar_vacancies = torch.argmin(ar_energies, dim=2).float()
        fr_vacancies = torch.argmin(fr_energies, dim=2).float()
        delta_v = fr_vacancies - ar_vacancies

        otc_draws = (mass_O * delta_v) / ( m_total.unsqueeze(1))

        # Compute OTC statistics
        otc_mean = otc_draws.mean(dim=1)
        otc_std = otc_draws.std(dim=1)

        # No full distribution or per-member data saved for MC sampling mode
        otc_distribution = None
        delta_v_array = None
        per_member_data = None

    return otc_mean, otc_std, gibbs_means, gibbs_stds, otc_distribution, delta_v_array, m_total_np, per_member_data


def samples_ensemble(wae_model, ensemble, sample_name, latents, iteration, method_list, config, log_path,
                    logger=None, cycle=None, existing_df=None):
    """
    Create a df which is exported to csv containing samples to be verified using Deep Ensemble.

    Similar to samples_GP but uses Deep Ensemble for predictions with uncertainty decomposition.

    How it works
    ------------
    1. Fit GMM to latents and sample 25k candidates
    2. Decode latents to compositions using WAE
    3. Predict properties with ensemble (gets mean, aleatoric, epistemic, total uncertainty)
    4. Apply acquisition function (thompson/explore/greedy)
    5. Filter for uniqueness and export top N

    Parameters
    ----------
    wae_model : torch.nn.Module
        Trained WAE model for decoding latents to compositions
    ensemble : DeepEnsemble
        Trained deep ensemble model for property prediction
    sample_name : str
        Path to save samples CSV
    latents : np.ndarray
        Training latent representations [N, latent_dim]
    iteration : int
        Current AL iteration
    method_list : list
        List of property names (e.g., ['otc'])
    config : Config
        Configuration object containing all parameters (device, sampling strategy, elements, etc.)
    log_path : str
        Directory for logging statistics
    logger : logging.Logger, optional
        Logger for messages
    cycle : int, optional
        Current cycle number
    existing_df : pd.DataFrame, optional
        DataFrame of existing samples in the database to filter out duplicates

    Returns
    -------
    pd.DataFrame
        Selected samples (returned from export_top_unique)
    """
    # Extract parameters from config
    n_props = len(method_list)

    # 1. Generate candidate samples (decoded compositions)
    decoded_samples = generate_candidate_samples(
        wae_model, latents, config.n_samples_total, config.device,
        use_rand=False, random_seed=0
    )

    # Round compositions to discrete atom counts if requested
    if config.round_compositions:
        decoded_samples = round_compositions_to_atoms(
            decoded_samples, len(config.A_elements), len(config.B_elements),
            elements=config.elements, scaling_dictionary=config.scaling_dictionary
        )
        if logger:
            logger.info('Rounded compositions to discrete atom counts.')

    # 2. Predict properties and uncertainties
    sample_list = []  # For each property
    predictions_dict = {}  # Store mean predictions for logging
    uncertainties_dict = {}  # Store uncertainties for logging

    if config.predict_landscape:
        # Landscape mode: derive OTC from predicted Gibbs energy landscape
        otc_mean, otc_std, gibbs_means, gibbs_stds, otc_distribution, delta_v_array, m_total_np, per_member_data = _predict_otc_from_landscape(
            ensemble, decoded_samples, config, iteration=iteration, log_path=log_path, logger=logger
        )

        # Store OTC predictions
        predictions_dict['otc'] = otc_mean.detach().cpu().numpy()
        uncertainties_dict['otc'] = otc_std.detach().cpu().numpy()

        # Store Gibbs energy predictions (skip 'otc' which is first in method_list)
        gibbs_property_names = method_list[1:]  # All Gibbs energy names
        gibbs_means_np = gibbs_means.detach().cpu().numpy()  # [N, n_gibbs_dims]
        gibbs_stds_np = gibbs_stds.detach().cpu().numpy()  # [N, n_gibbs_dims]

        for i, prop_name in enumerate(gibbs_property_names):
            predictions_dict[prop_name] = gibbs_means_np[:, i]
            uncertainties_dict[prop_name] = gibbs_stds_np[:, i]

        # Apply acquisition strategy (only for OTC, used for ranking)
        # Pass empirical distribution for true Thompson sampling (no Gaussian assumption)
        acquisition_values, sampled_member_indices = _apply_acquisition_strategy(
            otc_mean, otc_std, config.sampling, ensemble.device, distribution=otc_distribution
        )
        sample_list.append(acquisition_values)

        # Add sampled member indices to per_member_data if available
        if per_member_data is not None and sampled_member_indices is not None:
            per_member_data['sampled_member_indices'] = sampled_member_indices

        export_method_list = ['otc']

    else:
        # Direct property prediction mode
        # Move data to ensemble device
        if hasattr(ensemble, 'device'):
            decoded_samples = decoded_samples.to(ensemble.device)

        # Populated in the loop below for the ranked property. It carries only
        # OTC per member, not the gibbs_* arrays landscape mode records, so the
        # landscape per-member logger cannot consume it (see the logging call at
        # the end of this function, which stays gated on predict_landscape).
        per_member_data = None
        m_total_np = None

        # The physical OTC ceiling is applied in export_top_unique, which every
        # surrogate funnels through, so it is not computed here any more. The
        # old version was gated on config.use_ensemble and hardcoded the
        # z-score forward transform, which made --clip-otc-physical a silent
        # no-op for --model gp and --model mc.

        for i in range(n_props):
            property_name = method_list[i] if i < len(method_list) else f'property{i}'

            # Get ensemble predictions
            predictions = ensemble.predict(decoded_samples)
            all_means = predictions['mean'].squeeze()  # [N]
            all_stds_epistemic = predictions['epistemic_std'].squeeze()  # [N]

            # Log ensemble statistics for all candidates
            log_ensemble_statistics(predictions, iteration, property_name, log_path)

            # Acquisition uses member spread only -- never the aleatoric term.
            # `individual_means` is [N, M] for output_dim == 1 (ensemble_mlp.py
            # squeezes it), and empirical Thompson wants [M, N] to match the
            # landscape branch's [n_samples, N] contract, so it is transposed.
            #
            # Ranking on member means is the same acquisition the landscape arms
            # use: one member is drawn per candidate and its own prediction is
            # the ranking value. Folding in the learned aleatoric sigma instead
            # (`total_std`) samples irreducible noise that no amount of querying
            # the candidate can reduce, so it only adds undirected randomness to
            # the ranking -- the same argument that keeps --gp-observation-noise
            # off by default for the GP arm (see samples_GP above).
            member_means = predictions['individual_means']  # [N, M]
            if member_means.dim() == 3:
                # output_dim > 1: [N, output_dim, M] -> take this property's row
                member_means = member_means[:, i, :]
            otc_distribution = member_means.detach().cpu().numpy().T  # [M, N]

            # Store predictions and uncertainties for later logging. The logged
            # sigma is now the epistemic std, matching what acquisition ranks on
            # and putting this arm on the same footing as gp/mc/landscape arms.
            predictions_dict[property_name] = all_means.detach().cpu().numpy()
            uncertainties_dict[property_name] = all_stds_epistemic.detach().cpu().numpy()

            # Apply acquisition strategy
            acquisition_values, sampled_member_indices = _apply_acquisition_strategy(
                all_means, all_stds_epistemic, config.sampling,
                all_stds_epistemic.device, distribution=otc_distribution
            )
            sample_list.append(acquisition_values)

            # Per-member record for the ranked property, so the direct-mode arms
            # can report which member won a candidate just as landscape mode does.
            if i == 0 and sampled_member_indices is not None:
                per_member_data = {
                    'otc_per_member': otc_distribution,             # [M, N]
                    'sampled_member_indices': sampled_member_indices,
                }

        export_method_list = method_list

    # 3. Create DataFrame with property values and compositions
    new_samples = decoded_samples.tolist()
    for i, row in enumerate(new_samples):
        for j in range(len(export_method_list) - 1, -1, -1):
            row.insert(0, sample_list[j][i])

    new_samples = pd.DataFrame(new_samples, columns=(export_method_list + config.elements[0:-1]), dtype=float)

    # Apply sublattice scaling
    for k, v in config.scaling_dictionary.items():
        if k in new_samples.columns:
            new_samples[k] = new_samples[k] * v

    new_samples['O'] = config.scaling_dictionary['O']

    # 4. Export top unique samples
    selected_df = export_top_unique(
        new_samples, config.elements, export_method_list, config.default_samples_per_cycle,
        sample_name, config.tol, max_element_count=config.max_element_count,
        existing_df=existing_df, logger=logger,
        a_elements=config.A_elements, b_elements=config.B_elements,
        scaling_dictionary=config.scaling_dictionary,
        natoms=config.atoms_in_structure, config=config
    )

    # Log predictions and uncertainties of selected samples
    # Use method_list (not export_method_list) to log ALL properties including Gibbs energies
    selected_indices = selected_df.index.values

    # Standard aggregated logging (all modes)
    log_selected_sample_predictions(
        selected_indices, predictions_dict, uncertainties_dict,
        iteration, method_list, config.sampling, log_path, cycle=cycle
    )

    # Per-member logging (landscape mode only)
    if config.predict_landscape:
        log_landscape_predictions_per_member(
            selected_indices, per_member_data, m_total_np,
            iteration, method_list, config.sampling, log_path, cycle=cycle
        )

    return selected_df


def log_ensemble_statistics(predictions, iteration, property_name, log_path):
    """
    Log ensemble prediction statistics (mean and uncertainties) to CSV.

    Parameters
    ----------
    predictions : dict
        Ensemble predictions containing 'mean', 'aleatoric_std', 'epistemic_std', 'total_std'
    iteration : int
        Current active learning iteration
    property_name : str
        Name of the property being predicted
    log_path : str
        Directory path where statistics CSV will be saved
    """
    means = predictions['mean'].detach().cpu().numpy()
    stds_aleatoric = predictions['aleatoric_std'].detach().cpu().numpy()
    stds_epistemic = predictions['epistemic_std'].detach().cpu().numpy()
    stds_total = predictions['total_std'].detach().cpu().numpy()

    # Compute statistics
    stats = {
        'iteration': iteration,
        'property': property_name,
        'unc_aleatoric_min': float(np.min(stds_aleatoric)),
        'unc_aleatoric_mean': float(np.mean(stds_aleatoric)),
        'unc_aleatoric_max': float(np.max(stds_aleatoric)),
        'unc_aleatoric_std': float(np.std(stds_aleatoric)),
        
        'unc_epistemic_min': float(np.min(stds_epistemic)),
        'unc_epistemic_mean': float(np.mean(stds_epistemic)),
        'unc_epistemic_max': float(np.max(stds_epistemic)),
        'unc_epistemic_std': float(np.std(stds_epistemic)),
        
        'unc_total_min': float(np.min(stds_total)),
        'unc_total_mean': float(np.mean(stds_total)),
        'unc_total_max': float(np.max(stds_total)),
        'unc_total_std': float(np.std(stds_total)),
        'mu_min': float(np.min(means)),
        'mu_mean': float(np.mean(means)),
        'mu_max': float(np.max(means)),
        'mu_std': float(np.std(means)),
        'n_points': int(means.size)
    }

    # Write to CSV
    os.makedirs(log_path, exist_ok=True)
    stats_file = os.path.join(log_path, 'ensemble_stats.csv')
    stats_df = pd.DataFrame([stats])

    if os.path.exists(stats_file):
        stats_df.to_csv(stats_file, index=False, mode='a', header=False)
    else:
        stats_df.to_csv(stats_file, index=False)
