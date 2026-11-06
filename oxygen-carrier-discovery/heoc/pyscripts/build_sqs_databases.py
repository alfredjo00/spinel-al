#!/usr/bin/env python3
"""Build the SQS databases <directory>/cycle<N>/sqs_<formula>.db.

For each composition, a special quasirandom structure (SQS) of the 84-atom spinel
supercell is generated at each of 0 to 7 oxygen vacancies. Compositions come from
cycle<N>/sample.db of the highest-numbered cycle, or from training.db if only cycle0
exists.
"""

from ase.db import connect
import argparse
from glob import glob
from multiprocessing import Pool
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.getcwd().split('oxygen-carrier-discovery')[0],
                             'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.cluster_expansion import\
    build_sqs_db
from metal.applications.crystals.tools import (get_rows_db, get_sublattice_map,
                                               _atoms_row_to_structure)


def get_spinel_tag(atoms_row):
    '''Generate tag by adding ``_spinel`` to the chemical formula

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    return f'{structure.get_chemical_formula()}_spinel'


def mp_fun(kwarg_list):
    '''Create SQS databases for a list of keyword argument sets

    Parameters
    ----------
    kwarg_list
        list of dictionaries with keyword arguments for build_sqs_db
    '''
    for kwargs in kwarg_list:
        build_sqs_db(**kwargs)


def build_sqs_structures(directory='spinel', selection=None, processes=1, n_steps=None,
                         asynchronus=True, dimensions=(1, 2, 3), n_vacancies_max=8,
                         random_seed=None, max_cations=None):

    '''Create databases with SQS structures of the 84-atom spinel supercell
    with 0 to n_vacancies_max - 1 oxygen vacancies. If only cycle0 exists the
    compositions are taken from the training database, and otherwise from
    sample.db of the latest cycle.

    Parameters
    ----------
    directory
        data directory
    selection
        can be:
        * an integer id
        * a string like 'key=value', where '=' can also be one of
          '<=', '<', '>', '>=' or '!='.
        * a string like 'key'
        * comma separated strings like 'key1<value1,key2=value2,key'
        * list of strings or tuples: [('charge', '=', 1)].
        * list of integer ids: [1, 3, 5].
    processes : int
        number of processes to start; default is to run sequentially
        (1 process)
    asynchronus : bool
        run the processes asynchronously, ignored if single process
    dimensions : tuple
        SQS supercell dimensions (default: (1, 2, 3))
    n_vacancies_max : int
        number of vacancy counts, 0 to n_vacancies_max - 1 (default: 8)
    random_seed : int
        seed of the SQS generation; the active learning loop passes the run seed
    max_cations : int
        skip structures with more distinct cations than this
    '''
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    # check cycle
    if len(glob(f'{parent_dir}/{directory}/cycle*')) > 1:
        cycle = sorted([int(os.path.basename(folder).replace('cycle', ''))
                        for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
        input_file = f'{parent_dir}/{directory}/cycle{cycle}/sample.db'
    else:
        input_file = f'{parent_dir}/{directory}/training.db'
        cycle = 0
    # Try to read crystal space, otherwise change path
    # Based on relative path of this script
    try:
        crystal_space = CrystalSpace.read(f'{parent_dir}/{directory}/crystal_space.spa')
    except:
        try:
            # Add crystal space based on relative location of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            spa_file_path = os.path.join(parent_dir, f'{directory}/crystal_space.spa')
            
            crystal_space = CrystalSpace.read(spa_file_path)
        except:
            print("Path to crystal_space.spa incorrect")   

    print('Input file:', input_file)
        
    primitives = crystal_space.structure_list
    dimensions_list = [dimensions]  # Convert tuple to list of tuples for compatibility
    prim_sl_sym_maps = {}
    for chem_sym in set(primitives[0].get_chemical_symbols()):
        prim_sl_sym_maps[chem_sym] = []
        for i in range(crystal_space.nstructures):
            for sl_sym, chem_syms in get_sublattice_map(
                    structure=primitives[i],
                    cluster_space=crystal_space.cluster_space_list[i]).items():
                if chem_sym in chem_syms:
                    prim_sl_sym_maps[chem_sym].append(sl_sym)
                    break

    # generate kwarg list
    tag_functions = [get_spinel_tag]
    kwarg_lists = []
    for row in get_rows_db(filename=input_file, selection=selection):
        # Candidates are filtered on cation count before selection, but that
        # count is taken on the overall composition while the structure is built
        # by rescaling each concentration onto a sublattice. A species whose
        # overall fraction rounds away can still occupy a site once rescaled, so
        # a realised structure can hold more distinct cations than the
        # composition it came from. Such a structure is not a valid candidate:
        # it is an artefact of that rescaling, and the Materials Project cannot
        # build a phase diagram for it. Drop it before anything is computed.
        if max_cations is not None:
            n_cations = len({symbol for symbol in row.symbols
                             if symbol not in ('O', 'X')})
            if n_cations > max_cations:
                print(f'Skipping {row.formula}: {n_cations} distinct cations'
                      f' exceeds the limit of {max_cations}')
                continue
        output_fname = f'sqs_{row.formula}.db'
        output_file = f'{parent_dir}/{directory}/cycle{cycle}/{output_fname}'
        existing_files = glob(f'{parent_dir}/{directory}/cycle*/{output_fname}')
        if len(existing_files) > 0:
            if output_file not in existing_files:
                print(f'The database {output_fname} already exists')
                continue
            db = connect(output_file)
            file_tags = [r.tag.split('_')[-1] for r in db.select()]
        else:
            file_tags = None
        concentrations = {symbol: count / row.natoms for symbol, count in
                          row.count_atoms().items()}
        sublattice_concentrations =\
            crystal_space.get_sublattice_concentrations(
                concentrations=concentrations,
                structure_index=0,
                dimensions=dimensions)
        kwarg_list = []
        for i in range(crystal_space.nstructures):
            primitive = primitives[i]
            sublattice_concentrations_list = []
            property_calculators = {'tag': get_spinel_tag}
            vacancies = ['X']
            
            # Assuming spinel structure
            n_supercells = len(primitive) * int(np.prod(dimensions)) / 7
            n_oxygen_total = n_supercells * 4
            
            for vacancy in np.arange(0, n_vacancies_max):
                sublattice_concentrations_list.append({
                    **sublattice_concentrations,
                    prim_sl_sym_maps['O'][i]: {'X': vacancy / n_oxygen_total,
                                                'O': 1 - vacancy / n_oxygen_total}})
            if file_tags is not None:
                tag = tag_functions[i](primitive).split('_')[-1]
                if file_tags.count(tag) >= len(sublattice_concentrations_list):
                    continue
            kwarg_list.append(dict(
                primitive=primitive,
                filename=output_file,
                cutoffs=crystal_space.cutoffs_list[i],
                chemical_symbols=crystal_space.chemical_symbols_list[i],
                sublattice_concentrations_list=sublattice_concentrations_list,
                dimensions=dimensions,
                n_steps=n_steps,
                vacancies=vacancies,
                append=True,
                random_seed=random_seed,
                property_calculators=property_calculators))
        kwarg_lists.append(kwarg_list)

    if processes == 1:
        for kwarg_list in kwarg_lists:
            mp_fun(kwarg_list)
    else:
        with Pool(processes) as pool:
            if asynchronus:
                results = pool.map_async(mp_fun, kwarg_lists)
                results.get()
            else:
                pool.map_async(mp_fun, kwarg_lists)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='spinel',
                        help='run directory')
    parser.add_argument('--min_id',
                        type=int,
                        default=None,
                        help='minimum row id to consider')
    parser.add_argument('--max_id',
                        type=int,
                        default=None,
                        help='maximum row id to consider')
    parser.add_argument('--processes',
                        default=None,
                        type=int,
                        help='number of processes; the default is SLURM_NTASKS'
                        ' or 1 if this environment variable is unspecified')
    parser.add_argument('--dimensions', type=int, nargs=3, default=[1, 2, 3],
                        help='SQS supercell dimensions (default: 1 2 3)')
    parser.add_argument('--n-vacancies-max', type=int, default=8,
                        help='number of vacancy counts, 0 to n - 1 (default: 8)')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='seed of the SQS generation; the active learning loop '
                             'passes the run seed')
    parser.add_argument('--max-cations', type=int, default=None,
                        help='skip structures with more distinct cations; the runs used 8')
    args = parser.parse_args()

    # parse selection
    selection = ['filtered=False']
    if args.min_id is not None:
        selection.append(f'id>={args.min_id}')
    if args.max_id is not None:
        selection.append(f'id<={args.max_id}')
    selection = ','.join(selection)

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    # determine number of processes
    processes = args.processes
    if processes is None:
        processes = int(os.getenv('SLURM_NTASKS', 1))

    build_sqs_structures(directory=args.directory, selection=selection,
                         processes=processes, dimensions=tuple(args.dimensions),
                         n_vacancies_max=args.n_vacancies_max,
                         random_seed=args.random_seed,
                         max_cations=args.max_cations)

    os.chdir(cwd)
