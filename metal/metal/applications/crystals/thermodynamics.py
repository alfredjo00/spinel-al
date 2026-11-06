from collections.abc import Sequence
from contextlib import nullcontext
from math import factorial
import numbers
import numpy as np
import os
import re
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
import tempfile
import traceback
from typing import Callable, Dict, List, Union, Tuple
import warnings

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.units import _hplanck, J, kJ, mol, kB

from icet import ClusterSpace

from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io.vasp.sets import MPRelaxSet, VaspInputSet

try:
    import torch
    from .torch_tools import get_torch_device
    pytorch_error = None
except Exception:
    pytorch_error = traceback.format_exc()

from .tools import (_add_structure_db, determine_supercell_dimensions,
                    get_row_ids)
from ...utils import decompress_db, compress_db, apply_function_db
from .base_optimizer import optimizer_keys


def _relax_structure(
        structure: Atoms,
        potential: str = 'CHGNet',
        model: str = None,
        optimizer: List[str] = ["BFGS", "LBFGS", "FIRE"],
        fmax: List[float] = [0.001, 0.01],
        steps: int = 1000000,
        optimizer_kwargs: Dict = None,
        relax_kwargs: Dict = None,
        relax_cell: bool = True) -> Atoms:
    """
    Relaxes the provided structure and attaches a calculator and also returns
    trajectory

        Parameters
    ----------
    structure
        structure, in the form an ase.Atoms object
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    model
        a `CHGNet` model or `CHGNetCalculator`; if None, which is the default,
        the pretrained CHGNet model is loaded
    optimizer
        the optimization algorithm, if a list is provided these are used in
        sequence until convergence is reached; defaults to "BFGS" followed by
        "LBFGS" and then "FIRE"
    fmax
        total force tolerance for relaxation convergence, here the sum of force
        and stress forces, if a list is provided these are set in sequence
       until convergence is reached; defaults to 0.001 followed by 0.01
    steps
        maximum number of steps for relaxation; defaults to 1000000
    optimizer_kwargs
        additional kwargs to chgnet.model.StructOptimizer
    relax_kwargs
        additional kwargs to chgnet.model.StructOptimizer.relax()
    relax_cell
        whether to relax the lattice cell
    """

    if isinstance(fmax, Sequence):
        max_forces = fmax
    else:
        max_forces = [fmax]
    if (isinstance(optimizer, Sequence) and not
            isinstance(optimizer, str)):
        optimizers = optimizer
    else:
        optimizers = [optimizer]

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    if relax_kwargs is None:
        relax_kwargs = {}
    relax_kwargs.update({
        'atoms': structure,
        'steps': steps
    })
    if potential.lower() != 'chgnet':
        raise ValueError(f'Only CHGNet is supported, not {potential}')
    from chgnet.model import StructOptimizer

    # Using A100 gpu causes some stability issues with CHGNet
    # Common fix for stability issues on Ampere GPUs
    # This doesnt allow low float precision matrix multiplications
    try:
        # Try new API (PyTorch 2.9+)
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
    except (AttributeError, RuntimeError):
        # Fall back to old API (PyTorch < 2.9)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    optimizer_kwargs['model'] = model
    relax_kwargs['relax_cell'] = relax_cell
    list_optimizer_kwargs = [
        dict(optimizer_class=optimizer_keys[opt.lower()],
             **optimizer_kwargs) for opt in optimizers]

    list_relax_kwargs = [
        dict(fmax=fmax, **relax_kwargs) for fmax in max_forces]

    for r, relax_kwargs in enumerate(list_relax_kwargs):

        for o, optimizer_kwargs in enumerate(list_optimizer_kwargs):

            # Relax structure and add calculator
            relaxer = StructOptimizer(**optimizer_kwargs)
            try:
                relax_results = relaxer.relax(**relax_kwargs)
            except Exception:
                warnings.warn('\n'.join([
                    f"The {optimizers[o]} optimizer failed to relax"
                    f" {structure.get_chemical_formula()} because:",
                    traceback.format_exc()]))
                continue

            if len(relax_results['trajectory'].energies) >= steps:
                warnings.warn(f"{structure.get_chemical_formula()} did not"
                              f" converge to within {max_forces[r]} eV/A when"
                              f" using an {optimizers[o]} optimizer")
                continue

            relaxed_structure = AseAtomsAdaptor.get_atoms(
                relax_results['final_structure'])
            relaxed_structure.calc = relaxer.calculator

            return relaxed_structure, relax_results['trajectory']

    return None, None


def _calculate_forces_sequential(supercells, calculator):
    """Sequential force calculation for all potentials

    Parameters
    ----------
    supercells
        list of PhonopyAtoms supercells
    calculator
        ASE calculator attached to structures

    Returns
    -------
    force_sets
        list of force arrays for each supercell
    """
    force_sets = []
    for supercell in supercells:
        super_structure = Atoms(
            cell=supercell.cell,
            numbers=supercell.numbers,
            pbc=True,
            scaled_positions=supercell.scaled_positions,
            calculator=calculator)
        force_sets.append(super_structure.get_forces())
    return force_sets


def _calculate_forces_batch(supercells, calculator, potential: str,
                            batch_size: int = None):
    """Calculate forces for multiple supercells, optionally using batch processing

    Parameters
    ----------
    supercells
        list of PhonopyAtoms supercells
    calculator
        ASE calculator attached to structures
    potential
        potential name to determine batching strategy
    batch_size
        if None, use sequential calculation (default, more stable);
        if integer, process structures in batches of this size for GPU efficiency
        (CHGNet only); recommended values: 16-64 depending on GPU memory

    Returns
    -------
    force_sets
        list of force arrays for each supercell
    """
    if batch_size is not None and potential.lower() == 'chgnet':
        # CHGNet's predict_structure supports native batch processing
        print(f"Calculating forces for {len(supercells)} supercells using CHGNet batch prediction (batch_size={batch_size})")

        # Get the CHGNet model from calculator
        if hasattr(calculator, 'model'):
            model = calculator.model
        else:
            # Fallback to sequential if we can't access model
            print("Cannot access CHGNet model, using sequential calculation")
            return _calculate_forces_sequential(supercells, calculator)

        # Convert PhonopyAtoms to pymatgen Structure objects for CHGNet
        from pymatgen.core import Structure, Lattice
        pmg_structures = []
        for supercell in supercells:
            lattice = Lattice(supercell.cell)
            structure = Structure(
                lattice=lattice,
                species=supercell.numbers,
                coords=supercell.scaled_positions,
                coords_are_cartesian=False
            )
            pmg_structures.append(structure)

        print("Converting structures to graphs in parallel...")
        # Parallelize graph conversion (CPU-intensive step)
        # The graph_converter is thread-safe and this speeds up preprocessing
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5*batch_size) as executor:
            graphs = list(executor.map(model.graph_converter, pmg_structures))

        print("Calculating forces using CHGNet batch prediction...")
        # Use CHGNet's native batch prediction on pre-converted graphs
        # Note: Cannot use torch.no_grad() because CHGNet computes forces as gradients!
        # Forces are calculated via torch.autograd.grad(energy, positions)
        predictions = model.predict_graph(
            graphs,
            task='ef',  # Only predict energy and forces (faster)
            batch_size=batch_size
        )

        # Extract forces from predictions
        # Handle both torch tensors and numpy arrays
        force_sets = []
        for pred in predictions:
            f = pred['f']
            if hasattr(f, 'cpu'):  # torch tensor
                force_sets.append(f.cpu().numpy())
            else:  # already numpy array
                force_sets.append(f)

        # Clear GPU cache to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return force_sets

    else:
        print("Using sequential force calculations")
        return _calculate_forces_sequential(supercells, calculator)


def _calculate_phonopy_phonons(
        structure: Atoms,
        potential: str = 'CHGNet',
        model: str = None,
        optimizer: List[str] = ["BFGS", "LBFGS", "FIRE"],
        fmax: List[float] = [0.001, 0.01],
        steps: int = 1000000,
        optimizer_kwargs: Dict = None,
        relax_kwargs: Dict = None,
        t_min: float = 0,
        t_max: float = 1500,
        t_step: float = 1,
        temperatures: List[float] = None,
        size: Union[float, int, List[int]] = 12.5,
        displacement: float = 0.02,
        kpts: Union[int, List[int]] = 1000000,
        include_dos: bool = False,
        corrected_energy: bool = False,
        batch_size: int = None,
        logging: object = None) -> Atoms:
    """
    Relaxes the provided structure and adds phonon properties, calculated using
    Phonopy, i.e. the temperature in K, Helmholtz free energy in eV/atom,
    internal energy in eV/atom, entropy in eV/K/atom, and heat capacity at
    constant volume in eV/K/atom as well as the phonon density of states and
     energies in eV/atom, if requested

    Parameters
    ----------
    structure
        structure, in the form an ase.Atoms object
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    model
        a `CHGNet` model or `CHGNetCalculator`; if None, which is the default,
        the pretrained CHGNet model is loaded
    optimizer
        the optimization algorithm, if a list is provided these are used in
        sequence until convergence is reached; defaults to "BFGS" followed by
        "LBFGS" and then "FIRE"
    fmax
        total force tolerance for relaxation convergence, here the sum of force
        and stress forces, if a list is provided these are set in sequence
       until convergence is reached; defaults to 0.001 followed by 0.01
    steps
        maximum number of steps for relaxation; defaults to 1000000
    optimizer_kwargs
        additional kwargs to chgnet.model.StructOptimizer
    relax_kwargs
        additional kwargs to chgnet.model.StructOptimizer.relax()
    t_min
        minimum temperature
    t_max
        maximum temperature
    t_step
        temperature step size
    temperatures
        list of temperatures, overrides t_min, t_max, and t_step
    size
        size of the supercell either in the form of a float that corresponds to
        a minimum lattice parameter, or a single integer, or list thereof,
        which indicates the number of times the primitive cell should be
        repeated in each principle direction
    displacement
        magnitude of displacement in angstrom
    kpts
        size of the k-points grid either in the form of an integer that
        corresponds to the total number of k-points per atom or a list of
        integers, which give the shape of the Monkhorst-Pack grid
    include_dos
        include phonon density of states in output
    batch_size
        if None, use sequential calculation (default, more stable);
        if > 0, use batch prediction for GPU efficiency (CHGNet only);
        if 0, use sequential calculation (default, more stable)
    """
    if logging is not None:
        logging.info(structure.get_chemical_formula())
    else:   
        print(structure.get_chemical_formula())
        
    # Relax structure
    relaxed_structure = _relax_structure(
        structure=structure,
        potential=potential,
        model=model,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        optimizer_kwargs=optimizer_kwargs,
        relax_kwargs=relax_kwargs
    )[0]
    if relaxed_structure is None:
        raise Exception(f"Failed to relax {structure.get_chemical_formula()}")
    calculator = relaxed_structure.calc

    # determine supercell dimensions and k-points grid
    kpts, size = _get_kpts_size(relaxed_structure, size, kpts)

    # Create Phonopy object
    phonopy_structure = PhonopyAtoms(
        symbols=relaxed_structure.get_chemical_symbols(),
        masses=relaxed_structure.get_masses(),
        scaled_positions=relaxed_structure.get_scaled_positions(),
        cell=relaxed_structure.cell)
    # primitive_matrix=identity: keep phonopy's primitive cell equal to the
    # input cell so thermal properties normalize against `natoms` below.
    # phonopy v4 treats None as 'auto', which can reduce to a smaller
    # primitive cell and break that normalization.
    phonon = Phonopy(phonopy_structure, supercell_matrix=np.diag(size),
                     primitive_matrix=np.eye(3))
    
    
    if logging is not None:
        logging.info("Phonopy obj. created")

    # Generate displaced supercells
    phonon.generate_displacements(distance=displacement)
    supercells = phonon.supercells_with_displacements

    # Calculate forces (optionally using batch processing)
    force_sets = _calculate_forces_batch(supercells, calculator, potential,
                                          batch_size)
    
    
    if logging is not None:
        logging.info(f"Calculated forces for {len(force_sets)} displaced supercells")
    else:
        print(f"Calculated forces for {len(force_sets)} displaced supercells")

    # Produce force constants
    disp_dataset = phonon.dataset
    if "natom" in disp_dataset:  # type-1 dataset
        for forces, disp in zip(force_sets, disp_dataset["first_atoms"]):
            disp["forces"] = forces
    elif "displacements" in disp_dataset:  # type-2 dataset
        disp_dataset["forces"] = np.array(force_sets, dtype="double",
                                          order="C")
    phonon.dataset = disp_dataset
    phonon.produce_force_constants()
    
    
    if logging is not None:
        logging.info("Force constants produced")
    else:
        print(f"Force constants produced")

    # Run mesh and calculate thermal properties
    _log = logging.info if logging is not None else print
    _log("run_mesh: start")
    phonon.run_mesh(kpts)
    _log("run_mesh: done; run_thermal_properties: start")
    # phonopy v4's run_thermal_properties() takes the pure-Python summation
    # path (mode_S/mode_F/mode_cv looped millions of times), ~100x slower than
    # the compiled path. Drive ThermalProperties directly with lang="C" to
    # force the fast compiled summation. T=0 makes exp(hw/kT) overflow for
    # every mode, so suppress the resulting numpy RuntimeWarning storm too.
    from phonopy.phonon.thermal_properties import ThermalProperties
    with np.errstate(over="ignore", invalid="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tp = ThermalProperties(phonon.mesh, classical=False)
        if temperatures is not None:
            tp.temperatures = temperatures
        else:
            tp.set_temperature_range(t_min=t_min, t_max=t_max, t_step=t_step)
        tp.run(lang="C")
        phonon._thermal_properties = tp
    _log("run_thermal_properties: done")
    tp_dict = phonon.get_thermal_properties_dict()
    natoms = len(relaxed_structure)
    potential_energy = relaxed_structure.get_potential_energy() / natoms
    info = {
        'temperature': list(tp_dict['temperatures']),
        'free_energy': list(tp_dict['free_energy'] * kJ / mol / natoms +
                            potential_energy),
        'entropy': list(tp_dict['entropy'] * J / mol / natoms),
        'heat_capacity': list(tp_dict['heat_capacity'] * J / mol / natoms),
    }
    info['internal_energy'] = list(cumtrapz(
        np.nan_to_num(info['heat_capacity'], posinf=0.0, neginf=0.0),
        np.array(info['temperature']), initial=0) + potential_energy)

    if include_dos:
        phonon.run_total_dos()
        total_dos_dict = phonon.get_total_dos_dict()
        pdos_data = {
            'phonon_dos': total_dos_dict['total_dos'],
            'phonon_energy': total_dos_dict[
                'frequency_points'] * 1e12 * _hplanck * J}
        info.update(pdos_data)
    relaxed_structure.info = info
    return relaxed_structure


def _get_kpts_size(
        structure: Atoms,
        size: Union[float, int, List[int]],
        kpts: Union[int, List[int]]) -> List[List[int]]:
    """Check how the kpts and size parameters have been specified and produce
    suitable representations, i.e. lists of integers, given the provided
    structure

    Parameters
    ----------
    structure
        structure, in the form an ase.Atoms object
    size
        size of the supercell either in the form of a float that corresponds to
        a minimum lattice parameter, or a single integer, or list thereof,
        which indicates the number of times the primitive cell should be
        repeated in each principle direction
    kpts
        size of the k-points grid either in the form of an integer that
        corresponds to the total number of k-points per atom or a list of
        integers, which give the shape of the Monkhorst-Pack grid
    """

    # determine supercell dimensions
    try:
        if isinstance(size, Sequence):
            assert len(size) == 3
            assert all([isinstance(s, numbers.Integral) for s in size])
        else:
            size = determine_supercell_dimensions(primitive=structure,
                                                  min_size=size)
    except Exception:
        raise Exception(f"The size must be either a sequence of three"
                        f" integers, a float, or an int not {type(size)}")

    # determine k-points grid
    try:
        if isinstance(kpts, Sequence):
            assert len(kpts) == 3
            assert all([isinstance(s, numbers.Integral) for s in kpts])
        else:
            pymatgen_structure = AseAtomsAdaptor.get_structure(structure)
            kpts = Kpoints.automatic_density(pymatgen_structure, kpts).kpts[0]
    except Exception:
        raise Exception(f"The kpts must be either a sequence of three"
                        f" integers or a single integer not {type(kpts)}")

    return kpts, size


def update_db_mlp_calculation(
        function: Callable,
        filenames: Union[str, List[str]],
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        potential: str = 'CHGNet',
        asynchronus: bool = False,
        processes: int = 1,
        threads_per_process: int = None,
        use_gpus: bool = False,
        semi_parallel: bool = False,
        **fun_kwargs) -> None:
    '''Use the supplied function to perform calculations, using an MLP, for
    the the selected rows in the specified, possibly compressed, database(s)

    Parameters
    ----------
    function
        function that should be applied to the database, which must accept a
        dictionary with input arguments as input
    filenames
        path(s) to database file(s); the calculations will be parallelised over
        multiple files if provided, and rows otherwise
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; if the number is less than two these are
        run sequentially
    threads_per_process
        number of threads to be used by each process. If unspecified it will be
        set equal to the environment variable `OMP_NUM_THREADS`, if available,
        otherwise, this will be handled by the MLP. In the latter case,
        multiple processes will typically compete for all available threads.
        `threads_per_process` is ignored when using GPUs.
    use_gpus
        if GPUs should be used; raises exception if unavailable
    semi_parallel
        run the calculations in a semi-parallel fashion; recommended for CHGNet
    '''

    # handle filenames
    if len(filenames) == 0:
        raise Exception('No files have been provided')
    if not isinstance(filenames, str) and len(filenames) == 1:
        filenames = filenames[0]
    if isinstance(filenames, str):
        if not os.path.isfile(filenames):
            raise Exception(f'The database {filenames} does not exist')
        dirname = os.path.dirname(filenames)
        if dirname != '':
            os.makedirs(dirname, exist_ok=True)

    if potential.lower() != 'chgnet':
        raise ValueError(f'Only CHGNet is supported, not {potential}')

    if use_gpus:

        if pytorch_error is not None:
            raise ImportError('\n'.join(['PyTorch is not available because'
                                         ' of', str(pytorch_error)]))

        # Count number of GPUs
        gpus = range(1 if torch.backends.mps.is_available() else
                     torch.cuda.device_count())
        if len(gpus) == 0:
            raise Exception('No GPUs available')

        if processes > len(gpus):
            raise Exception(f'The number of processes ({processes}) cannot'
                            f' exceed the number of GPUs ({len(gpus)})')

    # add potential to function kwargs
    fun_kwargs['potential'] = potential

    with (tempfile.TemporaryDirectory() if (isinstance(filenames, str) and
                                            '.gz' in filenames) else
            nullcontext()) as tmpdir:

        # decompress file if necessary
        if isinstance(filenames, str) and '.gz' in filenames:
            tmpnames = os.path.join(tmpdir, 'database.db')
            if os.path.isfile(filenames):
                decompress_db(filenames, tmpnames, delete=False)
        else:
            tmpnames = filenames

        if processes == 1:
            if use_gpus:
                fun_kwargs['gpu_id'] = -1
            function(dict(filenames=tmpnames, selection=selection,
                          **fun_kwargs))
        else:

            kwargs_list = []

            if semi_parallel:
                if isinstance(filenames, str):
                    row_ids = get_row_ids(tmpnames, selection)
                    processes = min(len(row_ids), processes)
                    for i in range(processes):
                        kwargs = dict(
                            filenames=tmpnames,
                            selection=row_ids[i::processes],
                            **fun_kwargs)
                        if use_gpus:
                            kwargs['gpu_id'] = i
                        else:
                            kwargs['threads_per_process'] = threads_per_process
                        kwargs_list.append(kwargs)
                else:
                    processes = min(len(filenames), processes)
                    for i in range(processes):
                        kwargs = dict(
                            filenames=filenames[i::processes],
                            selection=selection,
                            **fun_kwargs)
                        if use_gpus:
                            kwargs['gpu_id'] = i
                        else:
                            kwargs['threads_per_process'] = threads_per_process
                        kwargs_list.append(kwargs)
            else:
                if isinstance(filenames, str):
                    row_ids = get_row_ids(tmpnames, selection)
                    processes = min(len(row_ids), processes)
                    for row_id in row_ids:
                        kwargs = dict(
                            filenames=tmpnames,
                            selection=row_id,
                            **fun_kwargs)
                        if use_gpus:
                            kwargs['gpu_id'] = -1
                        else:
                            kwargs['threads_per_process'] = threads_per_process
                        kwargs_list.append(kwargs)
                else:
                    processes = min(len(filenames), processes)
                    for filename in tmpnames:
                        kwargs = dict(
                            filenames=[filename],
                            selection=selection,
                            **fun_kwargs)
                        if use_gpus:
                            kwargs['gpu_id'] = -1
                        else:
                            kwargs['threads_per_process'] = threads_per_process
                        kwargs_list.append(kwargs)

            if pytorch_error is not None:
                raise ImportError('\n'.join(['PyTorch is not available'
                                             ' because of',
                                             str(pytorch_error)]))
            Pool = torch.multiprocessing.Pool
            torch.multiprocessing.set_start_method('spawn', force=True)
            with Pool(processes) as pool:
                if asynchronus:
                    results = pool.map_async(
                        function,
                        kwargs_list,
                        chunksize=1
                    )
                    results.get()
                else:
                    results = pool.map(
                        function,
                        kwargs_list,
                    )

        if isinstance(filenames, str) and '.gz' in filenames:
            # Zip database and delete the existing file
            compress_db(tmpnames, filenames, delete=True)


def update_relax_structures_dbs_threaded(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        overwrite: bool = False,
        potential: str = 'CHGNet',
        model: Union[str, object] = None,
        optimizer: str = "BFGS",
        fmax: Union[float, List[float]] = None,
        steps: int = 1000000,
        optimizer_kwargs: Dict = None,
        relax_kwargs: Dict = None,
        relax_cell: bool = True,
        n_threads: int = 2,
        gpu_id: int = None,
        threads_per_process: int = None,
        timeout_minutes: float = 10.0,
        logger: object = None) -> None:
    """
    Relax database structures using threading to maximize GPU utilization.

    This function runs multiple relaxations concurrently using ThreadPoolExecutor,
    which keeps the GPU busy by interleaving forward passes from different
    relaxations. Use this when you have low GPU utilization issues.

    Parameters
    ----------
    filename
        path to database file (single file only)
    selection
        database selection (same format as update_relax_structures_dbs)
    overwrite
        if updated entries should be overwritten
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    model
        model path/name for the potential, or a pre-loaded model object
    optimizer
        optimization algorithm: 'BFGS', 'LBFGS', or 'FIRE'
    fmax
        force convergence tolerance (default: 0.001 eV/Å)
    steps
        maximum optimization steps
    optimizer_kwargs
        additional kwargs to structure optimizer
    relax_kwargs
        additional kwargs to relax function (default: {'verbose': False})
    relax_cell
        whether to relax the lattice cell
    n_threads
        number of concurrent relaxations (2-4 recommended for 40GB GPU)
        Higher values = more GPU utilization but more memory usage
    gpu_id
        GPU id to use
    threads_per_process
        OMP threads per relaxation
    timeout_minutes
        timeout per relaxation task in minutes (default: 45.0)
        If a relaxation exceeds this time, it will be terminated and skipped
    logger
        optional logger object for status messages
    """
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    from .tools import get_row_ids

    # Handle mutable default arguments
    if fmax is None:
        fmax = 0.001
    if relax_kwargs is None:
        relax_kwargs = {'verbose': False}

    def _log(msg):
        """Helper for consistent logging"""
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    _log(f"Running threaded relaxations with {n_threads} concurrent workers...")

    # Get row IDs to process
    row_ids = get_row_ids(filename, selection)
    _log(f"Processing {len(row_ids)} structures")

    if potential.lower() != 'chgnet':
        raise NotImplementedError(
            f"Threaded relaxation only implemented for CHGNet, not {potential}")

    # Check torch is available
    if pytorch_error is not None:
        raise ImportError(f"PyTorch not available: {pytorch_error}")

    from chgnet.model import StructOptimizer, CHGNet
    from chgnet.model.dynamics import CHGNetCalculator

    _log("Loading CHGNet model...")

    # Using A100 gpu causes some stability issues with CHGNet
    # Common fix for stability issues on Ampere GPUs
    try:
        # Try new API (PyTorch 2.9+)
        torch.backends.cuda.matmul.fp32_precision = 'ieee'
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
    except (AttributeError, RuntimeError):
        # Fall back to old API (PyTorch < 2.9)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    use_device = get_torch_device(
        potential=potential,
        gpu_id=gpu_id,
        threads_per_process=threads_per_process
    )

    # Load model once (this is what all threads share)
    if model is None:
        shared_model = CHGNet.load()
    elif isinstance(model, str):
        # model is a path to saved weights
        shared_model = CHGNet.load(model_path=model)
    else:
        # model is already a CHGNet object
        shared_model = model
    shared_model.to(use_device)
    _log(f"Model loaded on {use_device}")

    # Map optimizer string to class
    optimizer_map = {
        'BFGS': 'BFGS',
        'LBFGS': 'LBFGS',
        'FIRE': 'FIRE'
    }
    optimizer_name = optimizer if isinstance(optimizer, str) else optimizer[0]
    if optimizer_name.upper() not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                         f"Choose from {list(optimizer_map.keys())}")

    from ase import optimize as ase_optimize
    optimizer_class = getattr(ase_optimize, optimizer_map[optimizer_name.upper()])

    # Prepare optimizer kwargs (each thread will create its own optimizer)
    base_optimizer_kwargs = {
        'use_device': use_device,
        'model': shared_model,
        'optimizer_class': optimizer_class
    }
    if optimizer_kwargs is not None:
        base_optimizer_kwargs.update(optimizer_kwargs)

    # Thread-safe lock for database access
    db_lock = Lock()

    # Track rows being processed to prevent duplicate work
    rows_in_progress = set()

    def relax_single_row(row_id):
        """Relax a single database row"""
        try:
            # Read structure from database (thread-safe)
            with db_lock:
                # Check if another thread is already working on this row
                if row_id in rows_in_progress:
                    _log(f"  Row {row_id}: Already being processed by another thread")
                    return
                rows_in_progress.add(row_id)

                with connect(filename) as db:
                    row = db.get(id=row_id)
                    if not overwrite and 'relaxed' in row.key_value_pairs:
                        rows_in_progress.discard(row_id)
                        _log(f"  Row {row_id}: Already relaxed, skipping")
                        return
                    structure = row.toatoms()

            # Each thread creates its own optimizer (but shares the model!)
            # This prevents optimizer state conflicts between threads
            thread_optimizer = StructOptimizer(**base_optimizer_kwargs)

            # Convert to pymatgen structure
            from pymatgen.io.ase import AseAtomsAdaptor
            pmg_structure = AseAtomsAdaptor.get_structure(structure)

            # Prepare relax kwargs
            relax_kwargs_local = relax_kwargs.copy()
            relax_kwargs_local['relax_cell'] = relax_cell

            # Relax using thread-local optimizer
            fmax_value = fmax[0] if isinstance(fmax, list) else fmax
            relax_results = thread_optimizer.relax(
                pmg_structure,
                fmax=fmax_value,
                steps=steps,
                **relax_kwargs_local
            )

            # Check if relaxation succeeded
            if relax_results is None or 'final_structure' not in relax_results:
                _log(f"  Row {row_id}: Relaxation failed")
                rows_in_progress.discard(row_id)
                return

            relaxed = AseAtomsAdaptor.get_atoms(relax_results['final_structure'])
            if relaxed is None:
                _log(f"  Row {row_id}: Failed to convert relaxed structure")
                rows_in_progress.discard(row_id)
                return

            _log(f"  Row {row_id}: Relaxed {structure.get_chemical_formula()}")

            # Attach a new CHGNet calculator to get energy
            relaxed.calc = CHGNetCalculator(model=shared_model, use_device=use_device)

            # Get energy and forces, then attach SinglePointCalculator
            # This ensures the energy is stored properly in the database
            from ase.calculators.singlepoint import SinglePointCalculator
            calculator_kwargs = {
                'energy': relaxed.get_potential_energy(),
                'forces': relaxed.get_forces()
            }
            try:
                calculator_kwargs['stress'] = relaxed.get_stress()
            except Exception:
                pass
            relaxed.calc = SinglePointCalculator(relaxed, **calculator_kwargs)

            # Update database (thread-safe)
            with db_lock:
                with connect(filename) as db:
                    # Re-check if already relaxed (edge case: overwrite=False)
                    try:
                        row = db.get(id=row_id)
                        if row is None:
                            _log(f"  Row {row_id}: Row no longer exists")
                            return
                        if 'relaxed' in row.key_value_pairs and not overwrite:
                            _log(f"  Row {row_id}: Already relaxed, skipping update")
                            return
                        db.update(id=row_id, atoms=relaxed, relaxed=True)
                    except (KeyError, TypeError) as e:
                        _log(f"  Row {row_id}: Database error - {str(e)}")
                        return
                    finally:
                        rows_in_progress.discard(row_id)

        except KeyboardInterrupt:
            raise  # Re-raise to allow clean shutdown
        except Exception as e:
            _log(f"  Row {row_id}: Error - {str(e)}")
            traceback.print_exc()
            rows_in_progress.discard(row_id)

    # Run relaxations in parallel using threads with timeout protection
    import sys
    from concurrent.futures import as_completed, TimeoutError

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Timeout per relaxation task (in seconds)
    TIMEOUT_PER_TASK = timeout_minutes * 60

    completed_count = 0
    failed_count = 0
    timeout_count = 0

    try:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit all tasks and get future objects
            future_to_row_id = {executor.submit(relax_single_row, row_id): row_id
                               for row_id in row_ids}

            # Process completed futures with timeout
            for future in as_completed(future_to_row_id, timeout=None):
                row_id = future_to_row_id[future]
                try:
                    # Wait for this specific task with timeout
                    future.result(timeout=TIMEOUT_PER_TASK)
                    completed_count += 1
                    _log(f"  Row {row_id}: Completed ({completed_count}/{len(row_ids)})")

                except TimeoutError:
                    timeout_count += 1
                    _log(f"  Row {row_id}: TIMEOUT after {TIMEOUT_PER_TASK/60:.1f} minutes - skipping")
                    # Cancel the future to free resources
                    future.cancel()

                except Exception as e:
                    failed_count += 1
                    _log(f"  Row {row_id}: Failed with error: {str(e)}")

    except KeyboardInterrupt:
        _log("Interrupted by user, shutting down gracefully...")
        raise

    finally:
        # Restore stdout/stderr in case CHGNet redirected them
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        sys.stdout.flush()
        sys.stderr.flush()

    # Summary report
    summary = (f"Relaxation summary: {completed_count} completed, "
               f"{timeout_count} timed out, {failed_count} failed "
               f"(Total: {len(row_ids)})")
    _log(summary)


def get_corrected_energy(
        structure: Atoms,
        only_correction: bool = False,
        user_potcar_settings: dict = None) -> float:
    """Calculate the corrected energy according to the conventions in the
    2020 version of the Materials Project

    Parameters
    ----------
        structure
            atomic configuration
        only_correction
            only get the correction
        user_potcar_settings
            override POTCAR symbols, e.g. ``{"W": "W_sv"}``
    """

    # create Materials Project entry
    entry = get_computed_entry(structure,
                               user_potcar_settings=user_potcar_settings)

    # correct energy
    mpc = MaterialsProject2020Compatibility(check_potcar_hash=False)
    mpc.process_entry(entry)

    if only_correction:
        return entry.correction
    return entry.energy


def get_computed_entry(
        structure: Atoms,
        include_structure: bool = False,
        user_potcar_settings: dict = None) -> ComputedEntry:
    """Return Hubbard U values from a VASP input set from Pymatgen.

    Parameters
    ----------
        structure
            atomic configuration
        user_potcar_settings
            override POTCAR symbols, e.g. ``{"W": "W_sv"}``
    """

    # create Pymatgen structure
    pymatgen_structure = AseAtomsAdaptor.get_structure(structure)

    # get Materials Project input set
    input_set = MPRelaxSet(pymatgen_structure,
                           user_potcar_functional='PBE_54_W_HASH',
                           user_potcar_settings=user_potcar_settings or {})

    # generate entry
    is_hubbard = input_set.incar.get("LDAU", False)
    entry = ComputedEntry(
        composition=pymatgen_structure.composition,
        energy=structure.get_potential_energy(),
        parameters={'is_hubbard': is_hubbard,
                    'potcar_symbols': input_set.potcar_symbols,
                    # Labels from the POTCAR symbols, so no POTCAR files are needed.
                    # MP2020Compatibility with check_potcar_hash=False reads only
                    # the symbol, the second word of each label.
                    'potcar_spec': [{'titel': f'PAW_PBE {symbol}'}
                                    for symbol in input_set.potcar_symbols],
                    'hubbards': get_hubbards(input_set),
                    'run_type': ''.join(['GGA', '+U' if is_hubbard else ''])})
    return entry


def get_hubbards(
        input_set: VaspInputSet) -> Dict[str, str]:
    """Return Hubbard U values from a VASP input set from Pymatgen.

    Parameters
    ----------
        input_set
            input set for VASP calculations from Pymatgen
    """
    symbols = [re.split(r"_", s)[0] for s in input_set.potcar_symbols]
    if not input_set.incar.get("LDAU", False):
        return {}
    us = input_set.incar.get("LDAUU")
    js = input_set.incar.get("LDAUJ")
    if len(js) != len(us):
        js = [0] * len(us)
    if len(us) == len(symbols):
        return {symbols[idx]: us[idx] - js[idx] for idx in range(len(symbols))}
    if sum(us) == 0 and sum(js) == 0:
        return {}
    raise Exception('Length of U value parameters and atomic symbols are'
                    ' mismatched')


def update_phonopy_phonons_dbs(
        filenames: Union[str, List[str]],
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        dict_path: List[str] = ['phonons'],
        overwrite: bool = False,
        rerun_failed: bool = False,
        potential: str = 'CHGNet',
        model: str = None,
        optimizer: List[str] = ["BFGS", "LBFGS", "FIRE"],
        fmax: List[float] = [0.001, 0.01],
        steps: int = 1000000,
        optimizer_kwargs: Dict = None,
        relax_kwargs: Dict = None,
        t_min: float = 0,
        t_max: float = 1500,
        t_step: float = 1,
        temperatures: List[float] = None,
        size: Union[float, int, List[int]] = 12.5,
        displacement: float = 0.02,
        kpts: Union[int, List[int]] = 1000000,
        include_dos: bool = False,
        corrected_energy: bool = False,
        batch_size: int = None,
        logging: object = None,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        asynchronus: bool = False,
        processes: int = 1,
        threads_per_process: int = None,
        use_gpus: bool = False,
        semi_parallel: bool = False,
        connect_each_row: bool = False,
        use_lock_file: bool = True) -> None:
    """
    Update the selected rows in the specified, possibly compressed, database(s)
    after relaxing the structure and calculating the phonon properties with
    Phonopy, i.e. the temperature in K, Helmholtz free energy in eV/atom,
    internal energy in eV/atom, entropy in eV/K/atom, and heat capacity at
    constant volume in eV/K/atom as well as the phonon density of states and
    energies eV/atom, if requested

    Parameters
    ----------
    filenames
        path(s) to database file(s); the calculations will be parallelised over
        multiple files if provided, and rows otherwise
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    dict_path
        list defining the sequence of keys under which the results should be
        stored, in the data dictionary
    overwrite
        if updated entries should be overwritten
    rerun_failed
        if failed calculations should be rerun
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    model
        a `CHGNet` model or `CHGNetCalculator`; if None, which is the default,
        the pretrained CHGNet model is loaded
    optimizer
        the optimization algorithm, if a list is provided these are used in
        sequence until convergence is reached; defaults to "BFGS" followed by
        "LBFGS" and then "FIRE"
    fmax
        total force tolerance for relaxation convergence, here the sum of force
        and stress forces, if a list is provided these are set in sequence
       until convergence is reached; defaults to 0.001 followed by 0.01
    steps
        maximum number of steps for relaxation; defaults to 1000000
    optimizer_kwargs
        additional kwargs to chgnet.model.StructOptimizer
    relax_kwargs
        additional kwargs to chgnet.model.StructOptimizer.relax()
    t_min
        minimum temperature
    t_max
        maximum temperature
    t_step
        temperature step size
    temperatures
        list of temperatures, overrides t_min, t_max, and t_step
    size
        size of the supercell either in the form of a float that corresponds to
        a minimum lattice parameter, or a single integer, or list thereof,
        which indicates the number of times the primitive cell should be
        repeated in each principle direction
    displacement
        magnitude of displacement in angstrom
    kpts
        size of the k-points grid either in the form of an integer that
        corresponds to the total number of k-points per atom or a list of
        integers, which give the shape of the Monkhorst-Pack grid
    include_dos
        include phonon density of states in output
    corrected_energy
        add corrected energy using the scheme applied in the Materials Project
        database (https://docs.materialsproject.org/methodology/
        materials-methodology/thermodynamic-stability/thermodynamic-stability);
        not applicable for CHGNet since it has been trained with corrected
        energies
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    asynchronous
        run the processes asynchronously; ignored if sequential
    processes
        number of processes to start; if the number is less than two these are
        run sequentially
    threads_per_process
        number of threads to be used by each process. If unspecified it will be
        set equal to the environment variable `OMP_NUM_THREADS`, if available,
        otherwise, this will be handled by the MLP. In the latter case,
        multiple processes will typically compete for all available threads.
        `threads_per_process` is ignored when using GPUs.
    use_gpus
        if GPUs should be used; raises exception if unavailable
    semi_parallel
        run the calculations in a semi-parallel fashion; recommended for CHGNet
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    update_db_mlp_calculation(
        function=_update_phonopy_phonons_dbs_mp,
        filenames=filenames,
        selection=selection,
        potential=potential,
        asynchronus=asynchronus,
        processes=processes,
        threads_per_process=threads_per_process,
        use_gpus=use_gpus,
        semi_parallel=semi_parallel,
        dict_path=dict_path,
        overwrite=overwrite,
        rerun_failed=rerun_failed,
        model=model,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        optimizer_kwargs=optimizer_kwargs,
        relax_kwargs=relax_kwargs,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        temperatures=temperatures,
        size=size,
        displacement=displacement,
        kpts=kpts,
        include_dos=include_dos,
        corrected_energy=corrected_energy,
        batch_size=batch_size,
        logging=logging,
        keyword_filters=keyword_filters,
        exclude_filtered=exclude_filtered,
        property_calculators=property_calculators,
        connect_each_row=connect_each_row,
        use_lock_file=use_lock_file)


def _update_phonopy_phonons_dbs_mp(kwargs):
    """Parallelisable version of _update_phonopy_phonons_dbs"""
    try:
        _update_phonopy_phonons_dbs(**kwargs)
    except Exception:
        warnings.warn(''.join([
            'Failed to perform phonon calculations with Phonopy and update'
            ' rows because of', '\n', traceback.format_exc()]))


def _update_phonopy_phonons_dbs(
        filenames: Union[str, List[str]],
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        gpu_id: int = None,
        threads_per_process: int = None,
        dict_path: List[str] = ['phonons'],
        overwrite: bool = False,
        rerun_failed: bool = False,
        potential: str = 'CHGNet',
        model: str = None,
        optimizer: List[str] = ["BFGS", "LBFGS", "FIRE"],
        fmax: List[float] = [0.001, 0.01],
        steps: int = 1000000,
        optimizer_kwargs: Dict = None,
        relax_kwargs: Dict = None,
        t_min: float = 0,
        t_max: float = 1500,
        t_step: float = 1,
        temperatures: List[float] = None,
        size: Union[float, int, List[int]] = 12.5,
        displacement: float = 0.02,
        kpts: Union[int, List[int]] = 1000000,
        include_dos: bool = False,
        corrected_energy: bool = False,
        batch_size: int = None,
        logging: object = None,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        connect_each_row: bool = False,
        use_lock_file: bool = True) -> None:
    """
    Update the selected rows in the specified database(s) after relaxing the
    structure and calculating the phonon properties with Phonopy, i.e. the
    temperature in K, Helmholtz free energy in eV/atom, internal energy in
    eV/atom, entropy in eV/K/atom, and heat capacity at constant volume in
    eV/K/atom as well as the phonon density of states and energies eV/atom, if
    requested

    Parameters
    ----------
    filenames
        path(s) to database file(s); the calculations will be parallelised over
        multiple files if provided, and rows otherwise
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    gpu_id
        GPU id, selected automatically if negative
    threads_per_process
        number of threads to be used by each process. If unspecified it will be
        set equal to the environment variable `OMP_NUM_THREADS`, if available,
        otherwise, this will be handled by the MLP. In the latter case,
        multiple processes will typically compete for all available threads.
        `threads_per_process` is ignored when using GPUs.
    dict_path
        list defining the sequence of keys under which the results should be
        stored, in the data dictionary
    overwrite
        if updated entries should be overwritten
    rerun_failed
        if failed calculations should be rerun
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    model
        a `CHGNet` model or `CHGNetCalculator`; if None, which is the default,
        the pretrained CHGNet model is loaded
    optimizer
        the optimization algorithm, if a list is provided these are used in
        sequence until convergence is reached; defaults to "BFGS" followed by
        "LBFGS" and then "FIRE"
    fmax
        total force tolerance for relaxation convergence, here the sum of force
        and stress forces, if a list is provided these are set in sequence
       until convergence is reached; defaults to 0.001 followed by 0.01
    steps
        maximum number of steps for relaxation; defaults to 1000000
    optimizer_kwargs
        additional kwargs to chgnet.model.StructOptimizer
    relax_kwargs
        additional kwargs to chgnet.model.StructOptimizer.relax()
    t_min
        minimum temperature
    t_max
        maximum temperature
    t_step
        temperature step size
    temperatures
        list of temperatures, overrides t_min, t_max, and t_step
    size
        size of the supercell either in the form of a float that corresponds to
        a minimum lattice parameter, or a single integer, or list thereof,
        which indicates the number of times the primitive cell should be
        repeated in each principle direction
    displacement
        magnitude of displacement in angstrom
    kpts
        size of the k-points grid either in the form of an integer that
        corresponds to the total number of k-points per atom or a list of
        integers, which give the shape of the Monkhorst-Pack grid
    include_dos
        include phonon density of states in output
    corrected_energy
        add corrected energy using the scheme applied in the Materials Project
        database (https://docs.materialsproject.org/methodology/
        materials-methodology/thermodynamic-stability/thermodynamic-stability);
        not applicable for CHGNet since it has been trained with corrected
        energies
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    if isinstance(filenames, str):
        filenames = [filenames]
    fun_kwargs = {
        'selection': selection,
        'gpu_id': gpu_id,
        'threads_per_process': threads_per_process,
        'dict_path': dict_path,
        'overwrite': overwrite,
        'potential': potential,
        'model': model,
        'optimizer': optimizer,
        'fmax': fmax,
        'steps': steps,
        'optimizer_kwargs': optimizer_kwargs,
        'relax_kwargs': relax_kwargs,
        't_min': t_min,
        't_max': t_max,
        't_step': t_step,
        'temperatures': temperatures,
        'size': size,
        'displacement': displacement,
        'kpts': kpts,
        'include_dos': include_dos,
        'corrected_energy': corrected_energy,
        'batch_size': batch_size,
        'logging': logging,
        'keyword_filters': keyword_filters,
        'exclude_filtered': exclude_filtered,
        'property_calculators': property_calculators,
        'connect_each_row': connect_each_row,
        'use_lock_file': use_lock_file
    }
    for filename in filenames:
        apply_function_db(filename=filename,
                          function=_update_phonopy_phonons_mp,
                          fun_kwargs=fun_kwargs)


def _update_phonopy_phonons_mp(kwargs):
    """Parallelisable version of update_phonopy_phonons"""
    try:
        update_phonopy_phonons(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to perform phonon calculations with Phonopy and update'
            f' rows in the database {kwargs["filename"]} because of',
            '\n', traceback.format_exc()]))


def update_phonopy_phonons(
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        gpu_id: int = None,
        threads_per_process: int = None,
        dict_path: List[str] = ['phonons'],
        overwrite: bool = False,
        rerun_failed: bool = False,
        potential: str = 'CHGNet',
        model: str = None,
        optimizer: List[str] = ["BFGS", "LBFGS", "FIRE"],
        fmax: List[float] = [0.001, 0.01],
        steps: int = 1000000,
        optimizer_kwargs: Dict = None,
        relax_kwargs: Dict = None,
        t_min: float = 0,
        t_max: float = 1500,
        t_step: float = 1,
        temperatures: List[float] = None,
        size: Union[float, int, List[int]] = 12.5,
        displacement: float = 0.02,
        kpts: Union[int, List[int]] = 1000000,
        include_dos: bool = False,
        corrected_energy: bool = False,
        batch_size: int = None,
        logging: object = None,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        connect_each_row: bool = False,
        use_lock_file: bool = True) -> None:
    """
    Update the selected rows in the specified database after relaxing the
    structure and calculating the phonon properties with Phonopy, i.e. the
    temperature in K, Helmholtz free energy in eV/atom, internal energy in
    eV/atom, entropy in eV/K/atom, and heat capacity at constant volume in
    eV/K/atom as well as the phonon density of states and energies eV/atom, if
    requested

    Parameters
    ----------
    filename
        path to database file
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    gpu_id
        GPU id, selected automatically if negative
    threads_per_process
        number of threads to be used by each process. If unspecified it will be
        set equal to the environment variable `OMP_NUM_THREADS`, if available,
        otherwise, this will be handled by the MLP. In the latter case,
        multiple processes will typically compete for all available threads.
        `threads_per_process` is ignored when using GPUs.
    dict_path
        list defining the sequence of keys under which the results should be
        stored, in the data dictionary
    overwrite
        if updated entries should be overwritten
    rerun_failed
        if failed calculations should be rerun
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    model
        a `CHGNet` model or `CHGNetCalculator`; if None, which is the default,
        the pretrained CHGNet model is loaded
    optimizer
        the optimization algorithm, if a list is provided these are used in
        sequence until convergence is reached; defaults to "BFGS" followed by
        "LBFGS" and then "FIRE"
    fmax
        total force tolerance for relaxation convergence, here the sum of force
        and stress forces, if a list is provided these are set in sequence
       until convergence is reached; defaults to 0.001 followed by 0.01
    steps
        maximum number of steps for relaxation; defaults to 1000000
    optimizer_kwargs
        additional kwargs to chgnet.model.StructOptimizer
    relax_kwargs
        additional kwargs to chgnet.model.StructOptimizer.relax()
    t_min
        minimum temperature
    t_max
        maximum temperature
    t_step
        temperature step size
    temperatures
        list of temperatures, overrides t_min, t_max, and t_step
    size
        size of the supercell either in the form of a float that corresponds to
        a minimum lattice parameter, or a single integer, or list thereof,
        which indicates the number of times the primitive cell should be
        repeated in each principle direction
    displacement
        magnitude of displacement in angstrom
    kpts
        size of the k-points grid either in the form of an integer that
        corresponds to the total number of k-points per atom or a list of
        integers, which give the shape of the Monkhorst-Pack grid
    npts
        number of energy points
    broadening
        broadening of Lorentzian line-shape in eV
    include_dos
        include phonon density of states in output
    corrected_energy
        add corrected energy using the scheme applied in the Materials Project
        database (https://docs.materialsproject.org/methodology/
        materials-methodology/thermodynamic-stability/thermodynamic-stability);
        not applicable for CHGNet since it has been trained with corrected
        energies
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    update_phonons(
        function=_calculate_phonopy_phonons,
        filename=filename,
        selection=selection,
        gpu_id=gpu_id,
        dict_path=dict_path,
        overwrite=overwrite,
        rerun_failed=rerun_failed,
        potential=potential,
        model=model,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        optimizer_kwargs=optimizer_kwargs,
        relax_kwargs=relax_kwargs,
        t_min=t_min,
        t_max=t_max,
        t_step=t_step,
        temperatures=temperatures,
        size=size,
        displacement=displacement,
        kpts=kpts,
        include_dos=include_dos,
        corrected_energy=corrected_energy,
        batch_size=batch_size,
        logging=logging,
        keyword_filters=keyword_filters,
        exclude_filtered=exclude_filtered,
        property_calculators=property_calculators,
        connect_each_row=connect_each_row,
        use_lock_file=use_lock_file
    )


def update_phonons(
        function,
        filename: str,
        selection: Union[str, int, List[Union[int, str, Tuple]]] = None,
        gpu_id: int = None,
        threads_per_process: int = None,
        dict_path: List[str] = ['phonons'],
        overwrite: bool = False,
        rerun_failed: bool = False,
        potential: str = 'CHGNet',
        optimizer_kwargs: Dict = None,
        corrected_energy: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        connect_each_row: bool = False,
        use_lock_file: bool = True,
        **fun_kwargs) -> None:
    """
    Update the selected rows in the database with the results obtained by using
    the specified function to relax the provided structure and add phonon
    properties, i.e. the temperature in K, free energy in eV/atom, internal
    energy in eV/atom, entropy in eV/K/atom, and heat capacity at in eV/K/atom
    as well as the phonon density of states and energies in eV/atom, if
    requested

    Parameters
    ----------
    function
        function that must take the structure (structure) asmwell as boolean
        (include_dos), determines whether to include phonon density of states,
        together with a number of additional keyword arguments (fun_kwargs) as
        input and returns a relaxed structure in the form of an ASE Atoms
        object where the Atoms.info dictionary should contain the thermal
        properties, possibly together with the phonon density of states
    filename
        path to database file
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    gpu_id
        GPU id, selected automatically if negative
    threads_per_process
        number of threads to be used by each process. If unspecified it will be
        set equal to the environment variable `OMP_NUM_THREADS`, if available,
        otherwise, this will be handled by the MLP. In the latter case,
        multiple processes will typically compete for all available threads.
        `threads_per_process` is ignored when using GPUs.
    dict_path
        list defining the sequence of keys under which the results should be
        stored, in the data dictionary
    overwrite
        if updated entries should be overwritten
    rerun_failed
        if failed calculations should be rerun
    potential
        the machine-learned interatomic potential; only 'CHGNet' is supported
    optimizer_kwargs
        additional kwargs to chgnet.model.StructOptimizer
    corrected_energy
        add corrected energy using the scheme applied in the Materials Project
        database (https://docs.materialsproject.org/methodology/
        materials-methodology/thermodynamic-stability/thermodynamic-stability);
        not applicable for CHGNet since it has been trained with corrected
        energies
    keyword_filters
        list of keyword filters in the form of dictionaries with the following
        keys:
        ``key``: keyword associated with the ASE Atoms, or AtomsRow, object
        ``symbol``: symbol representing the comparison operator, i.e. '<',
            '<=', '=', '>=', '>', '!=' or '~', which corresponds to
            ``numpy.isclose()``
        ``value``: value with which the keyword should be compared
    exclude_filtered
        exclude the structures that do not pass through the filter; the default
        is to only add the boolean keyword ``filtered`` to the row
    property_calculators
        dictionary with keywords that should be added to the row and functions
        that are used calculate to calculate the value, which must take an
        AtomsRow object as input and return a scalar or dictionary
    connect_each_row
        reconnect to the database before processing each row; this means that
        changes are commited sequentially instead of all at once, which takes
        longer but reduces the memory usage
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    """

    if potential.lower() != 'chgnet':
        raise ValueError(f'Only CHGNet is supported, not {potential}')

    if corrected_energy:
        warnings.warn('CHGNet has been trained with corrected energies')
        corrected_energy = False

    if pytorch_error is not None:
        raise ImportError('\n'.join(['PyTorch is not available because of',
                                     str(pytorch_error)]))

    use_device = get_torch_device(
        potential=potential,
        gpu_id=gpu_id,
        threads_per_process=threads_per_process
    )
    if optimizer_kwargs is None:
        optimizer_kwargs = {'use_device': use_device}
    else:
        optimizer_kwargs['use_device'] = use_device

    # get row ids
    row_ids = get_row_ids(filename, selection)

    with (nullcontext() if connect_each_row else
          connect(filename, use_lock_file=use_lock_file)) as database:

        for row_id in row_ids:

            with (nullcontext() if not connect_each_row else
                  connect(filename, use_lock_file=use_lock_file)) as db:

                # Load structure
                if db is None:
                    db = database
                row = db.get(f"id={row_id}")

                # Check if phonon data already exists
                if not overwrite:
                    data_dict = row.data
                    for data_key in dict_path:
                        if data_key not in data_dict:
                            data_exists = False
                            break
                        data_exists = True
                        data_dict = data_dict[data_key]
                    if data_dict is None and rerun_failed:
                        data_exists = False
                    if data_exists:
                        warnings.warn(f"The data dictionary for row"
                                      f" {row.id} already contains the"
                                      f" sequence of keys "
                                      + '/'.join(dict_path))
                        continue

                # Calculate phonon properties
                try:
                    relaxed_structure = function(
                        structure=row.toatoms(),
                        potential=potential,
                        optimizer_kwargs=optimizer_kwargs,
                        **fun_kwargs)
                except Exception:
                    warnings.warn('\n'.join([
                        f"Failed to calculate phonon properties for row"
                        f" {row.id} with formula {row.formula} using"
                        f" {potential} because of",
                        traceback.format_exc()]))
                    continue
                add_data = relaxed_structure.info
                relaxed_structure.info = {}
                if corrected_energy:
                    corrected_energy = get_corrected_energy(
                        relaxed_structure)
                    correction = (corrected_energy -
                                  relaxed_structure.get_potential_energy()
                                  ) / len(relaxed_structure)
                    for prop in add_data.keys():
                        if 'energy' in prop:
                            add_data[prop] = [v + correction for v in
                                              add_data[prop]]

                # Attach single point calculator
                calculator_kwargs = {
                    'energy': relaxed_structure.get_potential_energy(),
                    'forces': relaxed_structure.get_forces()
                    }
                try:
                    calculator_kwargs[
                        'stress'] = relaxed_structure.get_stress()
                except Exception:
                    pass
                try:
                    calculator_kwargs['magmoms'] =\
                        relaxed_structure.get_magnetic_moments()
                except Exception:
                    pass
                relaxed_structure.calc = SinglePointCalculator(
                    relaxed_structure, **calculator_kwargs)

                # update row
                data = row.data
                for data_key in dict_path[::-1]:
                    add_data = {data_key: add_data}
                data.update(add_data)
                kwargs = {}
                if corrected_energy:
                    kwargs['corrected_energy'] = get_corrected_energy(
                        relaxed_structure)
                _add_structure_db(
                    database=db,
                    structure=relaxed_structure,
                    row_id=row.id,
                    data=data,
                    keyword_filters=keyword_filters,
                    exclude_filtered=exclude_filtered,
                    property_calculators=property_calculators,
                    **kwargs)

    if gpu_id is not None:
        torch.cuda.empty_cache()


def estimate_configurational_entropy(
        cluster_space: ClusterSpace,
        structure: Atoms,
        stirling: bool = True) -> float:
    """Calculate an estimate of the configurational entropy based on the
    provided structure and cluster space

    Parameters
    ----------
    cluster_space
        cluster space
    structure
        atomic configuration
    stirling
        use the Stirling approximation; otherwise the result only
        approaches the macroscopic value for an infitely large supercell
    """

    total_entropy = 0.0
    for sl in cluster_space.get_sublattices(structure):

        # add entropy contribution from sublattice
        symbols = [structure.symbols[i] for i in sl.indices]
        natoms = len(symbols)
        sublattice_entropy = 0.0
        if not stirling:
            sublattice_entropy += np.log(float(factorial(natoms)))
        for symbol in set(symbols):
            count = symbols.count(symbol)
            if stirling:
                sublattice_entropy -= np.log(count / natoms) * count
            else:
                sublattice_entropy -= np.log(float(factorial(count)))
        total_entropy += kB * sublattice_entropy
    return total_entropy

