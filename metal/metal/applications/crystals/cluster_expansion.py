from itertools import product
import numpy as np
from random import sample, choice
import traceback
from typing import Dict, List, Union, Callable
import warnings

from ase import Atoms

from icet import ClusterSpace
from icet.tools.structure_generation import (generate_sqs,
                                             generate_sqs_from_supercells,
                                             generate_sqs_by_enumeration,
                                             occupy_structure_randomly,
                                             _validate_concentrations)

from .tools import _add_structure_db_mp
from ...utils import apply_function_db, run_function_parallel

#### For reproducibility of structure generation for build_training_database.py and possibly other scripts
#
# NOTE: a process-global np.random.seed / random.seed does NOT make icet's SQS
# search reproducible: (1) icet only uses system entropy unless its explicit
# `random_seed` argument is set, and (2) under multiprocessing each structure's
# global-RNG state depends on the pool size / worker scheduling, so results
# change with --samples-per-cycle. Instead we pass an explicit deterministic
# `random_seed` straight into the icet generators (see generate_sqs_wrapper).
SQS_RANDOM_SEED = 3


def build_db_random_configurations(
        primitive: Atoms,
        filename: str,
        chemical_symbols: Union[List[str], List[List[str]]],
        sublattice_concentrations_list: List[Dict[str, Dict[str, float]]],
        dimensions: Union[int, List[int]] = 3,
        n_structures: int = 1,
        symprec: float = 1e-5,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        random_seed: int = None) -> None:
    """Create a database with randomly generated configurations based on a
    specific primitive structure and sublattice concentrations

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    sublattice_concentrations_list
        list of dictionaries with concentration of each species in the target
        structure, per sublattice (for example {'Au': 0.5, 'Pd': 0.5} for a
        single sublattice Au-Pd structure, or {'A': {'Au': 0.5, 'Pd': 0.5},
        'B': {'H': 0.25, 'X': 0.75}} for a system with two sublattices. The
        symbols defining sublattices ('A', 'B' etc) can be found by printing
        the cluster_space
    dimensions
        dimensions of the supercell
    n_structures
        number of structures to generate, for each concentration, before
        filtering
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or dictionary
    """

    # create dummy cluster space
    cs = ClusterSpace(primitive, cutoffs=[1.0],
                      chemical_symbols=chemical_symbols, symprec=symprec)
    supercell = primitive.repeat(dimensions)

    # generate structures
    # occupy_structure_randomly uses its own LOCAL np.random.default_rng(seed) --
    # it does NOT read the global numpy RNG state, so it must be seeded directly
    # via its random_seed argument. Each (concentration, i) draw gets a distinct
    # derived seed so multiple structures per concentration don't collide.
    fun_kwargs = []
    struct_idx = 0
    for sublattice_concentrations in sublattice_concentrations_list:
        for i in range(n_structures):
            if (i + 1) % 10 == 0:
                print('Generating structure', i + 1, 'of', n_structures, end='\r')
            structure = supercell.copy()
            struct_seed = None if random_seed is None else random_seed + struct_idx
            occupy_structure_randomly(structure, cs, sublattice_concentrations,
                                      random_seed=struct_seed)
            struct_idx += 1
            tag = (None if n_structures > 1 else
                   structure.get_chemical_formula())
            fun_kwargs.append(
                dict(structure=structure,
                     vacancies=vacancies,
                     tag=tag,
                     keyword_filters=keyword_filters,
                     exclude_filtered=exclude_filtered,
                     property_calculators=property_calculators))

    # add structures to database
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def build_sqs_db(
        primitive: Atoms,
        filename: str,
        cutoffs: List[float],
        chemical_symbols: Union[List[str], List[List[str]]],
        sublattice_concentrations_list: List[Dict[str, Dict[str, float]]],
        dimensions: Union[int, List[int]] = None,
        max_size: int = 4,
        enumeration: bool = False,
        n_steps: int = None,
        symprec: float = 1e-5,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        overwrite: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None,
        asynchronus: bool = False,
        processes: int = 1,
        random_seed: int = None) -> None:
    """Create a database with randomly generated configurations based on a
    specific primitive structure and sublattice concentrations

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    cutoffs
        cutoff radii per order that define the cluster space

        Cutoffs are specified in units of Angstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. ``cutoffs=[7.0, 4.5]`` thus implies
        that all pairs distanced 7 A or less will be included,
        as well as all triplets among which the longest distance is no
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    sublattice_concentrations_list
        list of dictionaries with concentration of each species in the target
        structure, per sublattice (for example {'Au': 0.5, 'Pd': 0.5} for a
        single sublattice Au-Pd structure, or {'A': {'Au': 0.5, 'Pd': 0.5},
        'B': {'H': 0.25, 'X': 0.75}} for a system with two sublattices. The
        symbols defining sublattices ('A', 'B' etc) can be found by printing
        the cluster_space
    dimensions
        dimensions of the supercell; is superseded by ``max_size``
    max_size
        search among all supercells obtained by repeating the primitive cell,
        in a single direction, between one and ``max_size`` number of times;
        supersedes ``dimensions``
    enumeration
        find SQS via enumeration; ignored if ``dimensions`` is specified
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    overwrite
        overwrite exisiting entries if the formulas match
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or
        dictionary; note that it can be used to overwrite the ``tag``
        associated with each row, which is used to determine the uniqueness
    asynchronus
        run the processes asynchronously, ignored if single process
    processes
        number of processes to start; default is to run sequentially
        (1 process)
    """

    # generate SQS structures
    fun_kwargs = [
        dict(primitive=primitive,
             cutoffs=cutoffs,
             chemical_symbols=chemical_symbols,
             dimensions=dimensions,
             max_size=max_size,
             symprec=symprec,
             n_steps=n_steps,
             enumeration=enumeration,
             random_seed=random_seed,
             sublattice_concentrations=sublattice_concentrations)
        for sublattice_concentrations in sublattice_concentrations_list]
    structures = run_function_parallel(function=_generate_sqs_wrapper_mp,
                                       fun_kwargs=fun_kwargs,
                                       asynchronus=asynchronus,
                                       processes=processes)

    # add SQS structures to database
    fun_kwargs = [
        dict(structure=structure,
             tag=structure.get_chemical_formula(),
             vacancies=vacancies,
             overwrite=overwrite,
             keyword_filters=keyword_filters,
             exclude_filtered=exclude_filtered,
             property_calculators=property_calculators)
        for structure in structures]
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)


def _generate_sqs_wrapper_mp(kwargs):
    """Parallelisable version of generate_sqs"""
    try:
        structure = generate_sqs_wrapper(**kwargs)
    except Exception:
        warnings.warn(''.join([
            'Failed to generate SQS structure with concentrations', '\n',
            kwargs["sublattice_concentrations"], f'based on '
            f'{kwargs["primitive"].get_chemical_formula()} because of',
            '\n', traceback.format_exc()]))
        return
    return structure


def generate_sqs_wrapper(
        primitive: Atoms,
        cutoffs: List[float],
        chemical_symbols: Union[List[str], List[List[str]]],
        sublattice_concentrations: Dict[str, Dict[str, float]],
        dimensions: Union[int, List[int]] = None,
        max_size: int = 4,
        n_steps: int = None,
        symprec: float = 1e-5,
        enumeration: bool = False,
        random_seed: int = None) -> None:
    """Generate a special quasi-random structure based on a specific primitive
    structure and sublattice concentrations

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    cutoffs
        cutoff radii per order that define the cluster space

        Cutoffs are specified in units of Angstrom and refer to the
        longest distance between two atoms in the cluster. The first
        element refers to pairs, the second to triplets, the third
        to quadruplets, and so on. ``cutoffs=[7.0, 4.5]`` thus implies
        that all pairs distanced 7 A or less will be included,
        as well as all triplets among which the longest distance is no
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    sublattice_concentrations
        concentration of each species in the target structure, per sublattice
        (for example {'Au': 0.5, 'Pd': 0.5} for a single sublattice Au-Pd
        structure, or {'A': {'Au': 0.5, 'Pd': 0.5}, 'B': {'H': 0.25,
        'X': 0.75}} for a system with two sublattices. The symbols defining
        sublattices ('A', 'B' etc) can be found by printing the cluster_space
    dimensions
        dimensions of the supercell; is superseded by ``max_size``
    max_size
        search among all supercells obtained by repeating the primitive cell,
        in a single direction, between one and ``max_size`` number of times;
        supersedes ``dimensions``
    symprec
        tolerance that is imposed when using spglib to analyze the symmetry
        of the structure on which the cluster space is based
    enumeration
        find SQS via enumeration; ignored if ``dimensions`` is specified
    """

    # create cluster space
    cs = ClusterSpace(primitive, cutoffs=cutoffs,
                      chemical_symbols=chemical_symbols, symprec=symprec)
    sqs_kwargs = {}
    if n_steps is not None:
        sqs_kwargs['n_steps'] = n_steps

    # find SQS
    # Seed icet's Monte-Carlo annealing explicitly so the decoration is
    # reproducible independent of worker scheduling / pool size. The seed comes
    # from the active-learning --seed flag (threaded down); fall back to the
    # module constant if none is supplied. Enumeration is deterministic and
    # takes no seed.
    seed = SQS_RANDOM_SEED if random_seed is None else random_seed
    if _check_concentrations(sublattice_concentrations, cs):
        if dimensions is not None:
            structure = generate_sqs_from_supercells(
                cluster_space=cs,
                supercells=[primitive.repeat(dimensions)],
                target_concentrations=sublattice_concentrations,
                random_seed=seed,
                **sqs_kwargs)
        else:
            if enumeration:
                structure = generate_sqs_by_enumeration(
                    cluster_space=cs,
                    max_size=max_size,
                    target_concentrations=sublattice_concentrations,
                    include_smaller_cells=True)
            else:
                structure = generate_sqs(
                    cluster_space=cs,
                    max_size=max_size,
                    target_concentrations=sublattice_concentrations,
                    random_seed=seed,
                    include_smaller_cells=True)
    else:

        warnings.warn('No swaps are possible given the specified'
                      ' concentrations {sublattice_concentrations}, a'
                      ' simple supercell of the primitive structure will'
                      ' therefore be created')

        if dimensions is None:
            dims = list([(size, 1, 1) for size in range(1, max_size + 1)])
        else:
            dims = [dimensions]

        for dim in dims:
            structure = primitive.repeat(dim)
            try:
                occupy_structure_randomly(structure, cs,
                                          sublattice_concentrations)
                break
            except ValueError:
                pass

            structure = None

        if structure is None:
            raise Exception('Failed to generate a supercell that matches the'
                            ' specificed concentrions')

    return structure


def _check_concentrations(concentrations: dict,
                          cluster_space: ClusterSpace) -> bool:
    """
    Checks that the concentrations are valide against a cluster space (raises
    `ValueError` if they do not match) and that swaps are possible one or more
    of the sublattices.

    Parameters
    ----------
    concentrations
        concentration specification
    cluster_space
        cluster space to check against
    """
    concentrations = _validate_concentrations(cluster_space=cluster_space,
                                              concentrations=concentrations)

    # check that swaps are possible on at least one sublattice
    for sl_symbol, conc_dict in concentrations.items():
        swap_possible = len(np.nonzero(list(conc_dict.values()))[0]) > 1
        if swap_possible:
            return True
    return False


def build_db_random_concentrations(
        primitive: Atoms,
        filename: str,
        chemical_symbols: Union[List[str], List[List[str]]],
        dimensions: Union[int, List[int]] = 3,
        fixed_symbols: List[str] = None,
        min_total: int = None,
        max_total: int = None,
        min_sublattice: dict = None,
        max_sublattice: dict = None,
        deviation_limit: int = None,
        n_structures: int = 100,
        vacancies: List[str] = None,
        append: bool = False,
        delete_existing: bool = False,
        overwrite: bool = False,
        compress: bool = False,
        keyword_filters: List[Dict[str, Union[int, str, float]]] = None,
        exclude_filtered: bool = False,
        property_calculators: Dict[str, Callable] = None) -> None:
    """Create a database of structures with random configurations and
       concentrations based on a specific primitive structure and set of
       chemical symbols

    Parameters
    ----------
    primitive
        primitive structure from which derivative superstructures should
        be generated
    filename
        path to database where the structures shall be stored
    chemical_symbols
        lists of chemical symbols, each of which must map to an element of the
        periodic table

        If a list of chemical symbols is provided for a given structure, all
        sites on the lattice will have the same allowed occupations as the
        input list.

        If a list of chemical symbols is provided then the outer list must be
        the same length as the primitive structure and ``chemical_symbols[i]``
        will correspond to the allowed species on lattice site ``i``.
    dimensions
        dimensions of the supercell
    fixed_symbols
        symbols for chemical elements in the primitive that should not be
        replaced
    min_total
        minimum total number of distinct elements in each structure
    max_total
        maximum total number of distinct elements in each structure
    min_sublattice
        minimum number of distinct elements on each sublattice
    max_sublattice
        maximum number of distinct elements on each sublattice
    deviation_limit
        maximum deviation in the number of atoms, of a given species, per
        sublattice
    n_structures
        number of structures to generate, before filtering
    vacancies
        list of symbols that correspond to vacancies and should therefore be
        removed from the generated structures
    append
        append structures if the output database already exists, otherwise
        an exception is raised unless ``delete_existing`` is True
    delete_existing
        delete existing database if ``append`` is not True
    overwrite
        overwrite exisiting entries if the formulas match
    compress
        if the database should be compressed; the default is to decide this
        from the filename
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
        AtomsRow or Atoms object as input and return a scalar or
        dictionary; note that it can be used to overwrite the ``tag``
        associated with each row, which is used to determine the uniqueness
    """

    # create cluster space
    cluster_space = ClusterSpace(primitive, cutoffs=[1.0],
                                 chemical_symbols=chemical_symbols)
    supercell = primitive.repeat(dimensions)
    sublattices = cluster_space.get_sublattices(supercell)

    # determine possible combinations
    combos = None
    if not all([lim is None for lim in [min_sublattice, max_sublattice]]):
        combos = []
        for sl in sublattices:
            if min_sublattice is not None:
                min_sl = min_sublattice.get(sl.symbol, 1)
            else:
                min_sl = 1
            if max_sublattice is not None:
                max_sl = max_sublattice.get(sl.symbol,
                                            len(sl.chemical_symbols))
            else:
                max_sl = len(sl.chemical_symbols)
            combos.append(list(range(min_sl, max_sl)))
        combos = np.array(list(product(*combos)))
    if not all([lim is None for lim in [min_total, max_total]]):
        if combos is None:
            combos = np.array(list(product(*[
                list(range(1, len(sl.chemical_symbols))) for sl in
                sublattices])))
        if min_total is not None:
            combos = combos[np.where(np.sum(combos, axis=1) >= min_total)]
        if max_total is not None:
            combos = combos[np.where(np.sum(combos, axis=1) <= max_total)]

    # generate structures
    fun_kwargs = []
    for _ in range(n_structures):
        structure = supercell.copy()
        if combos is not None:
            combo = choice(combos)
        for s, sl in enumerate(sublattices):
            if combos is not None:
                possible_symbols = sample(sl.chemical_symbols, k=combo[s])
            else:
                possible_symbols = sl.chemical_symbols
            indices = [
                index for index in sl.indices if (
                    fixed_symbols is None or structure.symbols[index] not in
                    fixed_symbols)]
            if len(indices) == 0:
                continue
            if (len(possible_symbols) == 1 or deviation_limit is None or
                    sl.symbol not in deviation_limit):
                for index in indices:
                    structure.symbols[index] = choice(possible_symbols)
            else:
                counts = np.array(list(product(
                    list(range(len(indices))), repeat=len(possible_symbols))))
                counts = counts[np.where(np.sum(counts, axis=1) ==
                                         len(indices))]
                max_diff = np.max(np.abs(np.atleast_2d([
                    counts[:, i] - counts[:, j] for i in
                    range(counts.shape[1] - 1) for j in
                    range(i + 1, counts.shape[1])])), axis=0)
                counts = counts[np.where(
                    max_diff <= deviation_limit[sl.symbol])]
                for count, symbol in zip(
                        choice(counts), sample(possible_symbols,
                                               k=len(possible_symbols))):
                    symbol_indices = sample(indices, k=count)
                    indices = list(set(indices) - set(symbol_indices))
                    for index in symbol_indices:
                        structure.symbols[index] = symbol
        tag = structure.get_chemical_formula()
        fun_kwargs.append(
            dict(structure=structure,
                 overwrite=overwrite,
                 vacancies=vacancies,
                 tag=tag,
                 keyword_filters=keyword_filters,
                 exclude_filtered=exclude_filtered,
                 property_calculators=property_calculators))

    # add structures to database
    apply_function_db(filename=filename,
                      function=_add_structure_db_mp,
                      fun_kwargs=fun_kwargs,
                      append=append,
                      delete_existing=delete_existing,
                      compress=compress)

