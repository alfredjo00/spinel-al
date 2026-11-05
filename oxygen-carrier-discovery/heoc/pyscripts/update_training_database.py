#!/usr/bin/env python3

import argparse
from ase.data import atomic_masses, atomic_numbers
from glob import glob
import multiprocessing
import numpy as np
from operator import itemgetter
import os
import pandas as pd
from scipy.interpolate import CubicSpline
import sys
import traceback
import warnings


sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.tools import (
    add_structure_db,
    get_rows_db,
    get_row_ids,
    get_average_abundance,
    get_price,
    get_mu_oxygen)
from ase.db import connect
from build_training_database import (get_stability_factors,
                                     get_average_metal_abundance,
                                     estimate_configurational_entropies)

# Per-worker crystal space cache (set by pool initializer or sequential path)
_worker_crystal_space = None


def _pool_init(spa_path):
    """Pool initializer: load CrystalSpace once per worker process."""
    global _worker_crystal_space
    import os
    print(f'  [pool_init] Worker {os.getpid()} loading crystal space...', flush=True)
    _worker_crystal_space = CrystalSpace.read(spa_path)
    print(f'  [pool_init] Worker {os.getpid()} ready.', flush=True)


def update_training_database(directory='wang', pattern='sqs_*.db', cycle=None,
                             asynchronus=True, use_lock_file=True,
                             processes=1, dimensions=(1, 2, 3), quadratic_fit=True):
    '''Update the database with training structures with the results from
    phonon calculations based on the SQS structures generated from the
    original training set or samples produced by the network in later cycles

    Parameters
    ----------
    directory
        data directory
    asynchronous
        run the processes asynchronously; ignored if sequential
    use_lock_file
        use a lock file to prevent simultaneous access to database; should be
        set to False when using parallel processing but might cause problems
        since multiple processes can attempt to change to the same row
    processes
        number of processes to start; if the number is less than two these are
        run sequentially
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Define list of keyword arguments
    if cycle is not None:
        overwrite = True
    else:
        cycle = sorted([
            int(os.path.basename(folder).replace('cycle', ''))
            for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
        overwrite = False

    train_file = f'{parent_dir}/{directory}/training.db'
    # sorted() is load-bearing for reproducibility, not cosmetic. These filenames
    # set the order of kwargs_list -> results -> the Phase 2 write loop, and so
    # the row order in training.db. glob() returns os.scandir order, and the SQS
    # files are produced by concurrent phonon jobs, so that order varies between
    # runs of the same seed. A different row order means create_df hands the
    # WAE/surrogate a differently ordered training set, the seeded DataLoader
    # shuffle permutes a different array, and the run diverges from cycle 2 on.
    filenames = sorted(glob(f'{parent_dir}/{directory}/cycle{cycle}/{pattern}'))

    # Resolve crystal space path
    spa_path = f'{parent_dir}/{directory}/crystal_space.spa'
    if not os.path.isfile(spa_path):
        spa_path = os.path.join(parent_dir, f'{directory}/crystal_space.spa')

    # Phase 1: compute OTC / Gibbs data in parallel (no db writes)
    compute_kwargs = {'cycle': cycle,
                      'directory': directory,
                      'overwrite': overwrite,
                      'quadratic_fit': quadratic_fit,
                      'dimensions': tuple(dimensions),
                      'train_file': train_file}
    kwargs_list = [dict(filename=filename, **compute_kwargs)
                   for filename in filenames]

    # Load crystal space once in main process (needed for write phase + sequential)
    import time as _time
    t0 = _time.perf_counter()
    global _worker_crystal_space
    _worker_crystal_space = CrystalSpace.read(spa_path)
    print(f'[update_training_db] Loaded crystal space in {_time.perf_counter()-t0:.2f}s')

    n_files = len(kwargs_list)
    processes = min(n_files, processes)
    print(f'[update_training_db] Phase 1: computing OTC/Gibbs for {n_files} files '
          f'(processes={processes})...')
    t0 = _time.perf_counter()
    if processes <= 1:
        results = [_compute_training_data_mp(kw) for kw in kwargs_list]
    else:
        # Use spawn to avoid deadlocks from forking after CUDA/library init
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes, initializer=_pool_init,
                      initargs=(spa_path,)) as pool:
            results = pool.map(_compute_training_data_mp, kwargs_list)
    t1 = _time.perf_counter()

    # Phase 2: write all results to training.db sequentially (no lock contention)
    results = [r for r in results if r is not None]
    n_ok = len(results)
    n_fail = n_files - n_ok
    print(f'[update_training_db] Phase 1 done in {t1-t0:.2f}s — '
          f'{n_ok} succeeded, {n_fail} failed/skipped')
    if not results:
        return

    crystal_space = _worker_crystal_space

    print(f'[update_training_db] Phase 2: writing {n_ok} results to {train_file}...',
          flush=True)
    t0 = _time.perf_counter()
    for i, result in enumerate(results):
        try:
            structure = result['structure']
            kvp = dict(result['key_value_pairs'])
            data = result['data']
            formula = result['formula']

            # Exact same call as original, but:
            #   - property_calculators=None (already computed in Phase 1)
            #   - tag=None (skip tag scan / get_list_of_matching_entries)
            #   - use_lock_file=False (no lock contention)
            if cycle == 0:
                add_structure_db(
                    filename=train_file,
                    structure=structure,
                    row_id=get_row_ids(filename=train_file,
                                       selection=f'formula={formula}')[0],
                    overwrite=overwrite,
                    use_lock_file=False,
                    data=data,
                    **kvp)
            else:
                add_structure_db(
                    filename=train_file,
                    structure=structure,
                    keyword_filters=crystal_space.filters_list[0],
                    overwrite=overwrite,
                    use_lock_file=False,
                    cycle=cycle,
                    data=data,
                    **kvp)
            print(f'  [{i+1}/{n_ok}] Wrote {formula}', flush=True)
        except Exception:
            warnings.warn(
                f'Failed to write {formula} to {train_file}:\n'
                + traceback.format_exc())
    print(f'[update_training_db] Phase 2 done in {_time.perf_counter()-t0:.2f}s',
          flush=True)
    print(f'[update_training_db] Total: {n_ok} compositions updated in training.db',
          flush=True)


def _compute_training_data_mp(kwargs):
    """Parallelisable wrapper for _compute_training_data."""
    try:
        return _compute_training_data(**kwargs)
    except Exception:
        warnings.warn(''.join([
            f'Failed to compute training data for'
            f' {kwargs["filename"]} because of', '\n',
            traceback.format_exc()]))
        return None


def _compute_training_data(filename, directory,
                           dimensions=(1, 2, 3),
                           quadratic_fit=False,
                           cycle=None, overwrite=True,
                           train_file=None):
    '''Compute OTC and Gibbs landscape data from an SQS database.
    Returns a dict with structure, key_value_pairs, and data ready for db write,
    or None if the file should be skipped.

    Parameters
    ----------
    filename
        path to database with SQS structures
    directory
        data directory
    dimensions
        SQS supercell dimensions
    quadratic_fit
        use quadratic fit for OTC
    cycle
        training cycle (used for overwrite logic)
    overwrite
        overwrite existing entries if the formulas match
    train_file
        path to training.db
    '''

    crystal_space = _worker_crystal_space
    basename = os.path.basename(filename).replace('.db', '').replace('sqs_', '')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    if train_file is None:
        train_file = f'{parent_dir}/{directory}/training.db'

    # Remove stale SQLite journal files that can cause hangs on NFS
    for f in [filename, train_file]:
        for suffix in ['-journal', '-wal', '-shm']:
            stale = f + suffix
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

    # check if row exists or has been updated
    # Use connect directly with use_lock_file=False to avoid NFS lock hangs
    if cycle == 0 or not overwrite:
        formula = os.path.basename(
            filename).replace('.db', '').replace('sqs_', '')
        with connect(train_file, use_lock_file=False) as db:
            rows = list(db.select(f'formula={formula}'))
        if rows is None or len(rows) < 1:
            if cycle == 0:
                print(f'There is no row in {train_file} for {formula}')
                return None
        else:
            if not overwrite and rows[0].get('energy', False) is not False:
                print(f'The row in {train_file} for {formula} has already been'
                      f' updated')
                return None

    # Define conditions in reactors (CLOU)
    conditions = {
        'AR': {'T': 950 + 273.15, 'pO2': 0.2},
        'FR': {'T': 1050 + 273.15, 'pO2': 1e-14}
    }

    # get rows from database (use_lock_file=False to avoid NFS lock hangs)
    with connect(filename, use_lock_file=False) as db:
        rows = list(db.select('energy'))

    # get maximum oxygen content and supercell size
    max_oxygen, max_natoms = max([(row.count_atoms()['O'], row.natoms) for
                                  row in rows], key=itemgetter(0))
    ncells = max_natoms // len(crystal_space.structure_list[0])

    # collect formation energies
    data_dicts = []
    for irow, row in enumerate(rows):

        # get free energy from phonon calculations, remove the 0 K energy
        # and add the formation energy before creating a spline
        try:
            form_energy = row.get('form_energy', None)
            if form_energy is None:
                correction = 0.0
            else:
                correction = form_energy - row.energy
            free_energy_spline = CubicSpline(
                np.asarray(row.data['phonons']['temperature']),
                np.asarray(row.data['phonons']['free_energy']
                           ) * row.natoms + correction)
        except Exception as e:
            print(f'Unable to retrieve free energies for row with tag'
                  f' {row.tag} from the database {filename} because of\n'
                  + str(e))
            data_dicts = None
            break

        # determine number of vacancies
        vacancies = max_oxygen - row.count_atoms()['O']

        # estimate configurational entropy, including oxidation states
        pt = get_stability_factors(row, directory=directory, instance=True)
        entropy = pt.entropy(anion_vacancies=True)

        for reactor, condition in conditions.items():

            # calculate total energy
            temperature = condition['T']
            free_energy = free_energy_spline(temperature)

            # calculate oxygen chemical potential
            muO = (get_mu_oxygen(temperature, condition['pO2'],
                                 add_u_dft=form_energy is None) * vacancies
                   if vacancies > 0 else 0.0)

            # add data
            data_dicts.append({
                'formation_energy': free_energy - entropy * temperature
                + muO,
                'vacancies': vacancies,
                'delta': vacancies / ncells,
                'reactor': reactor,
                'irow': irow
            })

    if data_dicts is None:
        print(f'  [compute] {basename}: no phonon data available')
        return None

    # create dataframe with lowest energy at each temperature
    df = pd.DataFrame(data_dicts).groupby(by=['reactor', 'vacancies'],
                                          as_index=False).min()

    stoich_row = rows[df.loc[(df.vacancies == 0)].irow.iloc[0]]
    stoich_mass = stoich_row.toatoms().get_masses().sum()

    # calculate formation energies relative to the stoichiometric phase
    for reactor in df.reactor.unique():
        df.loc[df.reactor == reactor, 'formation_energy'] -= df.loc[
            (df.reactor == reactor) & (df.vacancies == 0)
            ].formation_energy.iloc[0]

    # find vacancy concentration in each reactor
    vacancies = {}
    for reactor in df.reactor.unique():
        df_rea = df.loc[df.reactor == reactor]
        df_min = df_rea.loc[df_rea.formation_energy ==
                            df_rea.formation_energy.min()]
        vacancies[reactor] = df_min.vacancies.iloc[0]

    # calculate oxygen release capacity (gram O2 per gram OC)
    vacancy_diff = vacancies['FR'] - vacancies['AR']
    otc = (atomic_masses[atomic_numbers['O']] * vacancy_diff / stoich_mass)

    # Extract Gibbs energies for ML target (flattened landscape)
    gibbs_keys = {}
    for index, df_row in df.iterrows():
        # Save formation energy for each reactor and vacancy count
        # Keys format: gibbs_{reactor}_{vacancies} (e.g., gibbs_AR_0, gibbs_FR_7)
        key = f"gibbs_{df_row['reactor']}_{int(df_row['vacancies'])}"
        gibbs_keys[key] = df_row['formation_energy']

    vacancies_array = np.array(sorted(set(df.vacancies)))

    # Optionally compute OTC using quadratic fitting
    if quadratic_fit:
        # Sort Gibbs energies by vacancy number to ensure correct order for fitting
        gibbs_AR = np.array([gibbs_keys[f'gibbs_AR_{v}'] for v in vacancies_array])
        gibbs_FR = np.array([gibbs_keys[f'gibbs_FR_{v}'] for v in vacancies_array])

        # Fit quadratic polynomial: G(v) = a*v^2 + b*v + c
        # np.polyfit returns coefficients in descending order [a, b, c]
        coeffs_AR = np.polyfit(vacancies_array, gibbs_AR, 2)
        coeffs_FR = np.polyfit(vacancies_array, gibbs_FR, 2)

        # Analytical solution: minimum of quadratic ax^2 + bx + c is at x = -b/(2a)
        def find_quadratic_minimum(coeffs, gibbs_values, vacancies, min_val, max_val):
            """Find minimum of quadratic, clamped to [min_val, max_val]."""
            a = coeffs[0]  # Quadratic coefficient
            b = coeffs[1]  # Linear coefficient

            if abs(a) < 1e-10:  # Nearly linear
                return min_val if b > 0 else max_val

            if a < 0:  # Parabola opens downward - minimum at boundary
                # Evaluate at both boundaries to find which is lower
                val_min = np.polyval(coeffs, min_val)
                val_max = np.polyval(coeffs, max_val)
                return min_val if val_min < val_max else max_val
            else:  # Parabola opens upward - minimum at critical point
                v_min = -b / (2 * a)
                return np.clip(v_min, min_val, max_val)

        min_vac, max_vac = vacancies_array.min(), vacancies_array.max()
        v_AR_opt = find_quadratic_minimum(coeffs_AR, gibbs_AR, vacancies_array, min_vac, max_vac)
        v_FR_opt = find_quadratic_minimum(coeffs_FR, gibbs_FR, vacancies_array, min_vac, max_vac)

        vacancy_diff_qd = v_FR_opt - v_AR_opt

        otc_qd = (atomic_masses[atomic_numbers['O']] * vacancy_diff_qd / stoich_mass)
    else:
        otc_qd = None

    structure_index = 0
    concentrations = {s: c / stoich_row.natoms for s, c in
                      stoich_row.count_atoms().items()}
    cluster_space = crystal_space.cluster_space_list[structure_index]
    prototype = crystal_space.map(concentrations, structure_index,
                                  dimensions=dimensions)
    configurational_entropies = estimate_configurational_entropies(
        cluster_space=cluster_space, atoms_row=prototype)

    # Build result dict (no db write here)
    structure = stoich_row.toatoms()
    key_value_pairs = {
        'otc': otc,
        'vacancy_diff': vacancy_diff,
        'delta_diff': vacancy_diff / ncells,
        **configurational_entropies,
        **gibbs_keys,
        **row.key_value_pairs}

    # Optionally add quadratic fit OTC if computed
    if quadratic_fit and otc_qd is not None:
        key_value_pairs['otc_quadratic_fit'] = otc_qd
        key_value_pairs['vacancy_diff_quadratic_fit'] = vacancy_diff_qd
        key_value_pairs['mass_'] = stoich_mass

    # Compute property calculators here (in parallel) instead of during db write
    # These read CSV files from disk, so doing it per-row in the write loop is slow
    if cycle != 0:
        stability = get_stability_factors(atoms_row=structure, directory=directory)
        if isinstance(stability, dict):
            key_value_pairs.update(stability)
        key_value_pairs['abundance'] = get_average_abundance(structure)
        key_value_pairs['metal_abundance'] = get_average_metal_abundance(structure.copy())
        key_value_pairs['price'] = get_price(structure)

    data = {'delta_energies': data_dicts}

    formula = structure.get_chemical_formula()
    print(f'  [compute] {basename}: formula={formula}, natoms={len(structure)}, OTC={otc:.4f}' +
          (f', OTC_qd={otc_qd:.4f}' if otc_qd is not None else ''))
    return {
        'structure': structure,
        'formula': formula,
        'key_value_pairs': key_value_pairs,
        'data': data,
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='spinel',
                        help='data directory')
    parser.add_argument('--cycle',
                        default=None,
                        type=int,
                        help='cycle, default is latest; overwrites existing'
                        ' data if not None')
    parser.add_argument('--pattern',
                        default='sqs_*.db',
                        help='pattern used for selecting files')
    parser.add_argument('--processes',
                        default=None,
                        type=int,
                        help='number of processes; the default is SLURM_NTASKS'
                        ' or 1 if this environment variable is unspecified')
    parser.add_argument('--dimensions', type=int, nargs=3, default=[1, 2, 3],
                        help='SQS supercell dimensions (default: 1 2 3)')
    args = parser.parse_args()

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    # ensure that all warnings are printed
    warnings.simplefilter("always")

    # determine number of processes
    processes = args.processes
    if processes is None:
        processes = int(os.getenv('SLURM_NTASKS', 1))

    # Update the database with training structures with the results from
    # phonon calculations based on the SQS structures generated from the
    # original training set or samples produced by the network in later cycles
    update_training_database(directory=args.directory,
                             pattern=args.pattern,
                             asynchronus=True,
                             cycle=args.cycle,
                             processes=processes,
                             dimensions=args.dimensions)

    os.chdir(cwd)
