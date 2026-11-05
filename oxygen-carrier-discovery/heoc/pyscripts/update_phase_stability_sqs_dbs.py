#!/usr/bin/env python3

import argparse
from glob import glob
import os
from ase.db import connect
from pymatgen.core import SETTINGS
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.stability_factors import\
    update_phase_stability_dbs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='spinel',
                        help='data directory')
    parser.add_argument('--cycle',
                        default=None,
                        type=int,
                        help='cycle, default is latest')
    parser.add_argument('--pattern',
                        default='sqs_*.db',
                        help='pattern used for selecting files')
    parser.add_argument('--processes',
                        default=None,
                        type=int,
                        help='number of processes; the default is SLURM_NTASKS'
                        ' or 1 if this environment variable is unspecified')
    args = parser.parse_args()

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    # determine number of processes
    processes = args.processes
    if processes is None:
        processes = int(os.getenv('SLURM_NTASKS', 1))

    # check cycle
    directory = args.directory
    if args.cycle is None:
        cycle = sorted([
            int(os.path.basename(folder).replace('cycle', ''))
            for folder in glob(f'{directory}/cycle*')])[-1]
    else:
        cycle = args.cycle

    # Update all SQS databases by estimating the phase stability, which
    # involves calculating the formation and decomposition energy as well as
    # the distance to the convex hull
    api_key = SETTINGS['PMG_MAPI_KEY']

    # Filter out databases where all rows already have form_energy
    filenames = []
    for f in glob(f'{directory}/cycle{cycle}/{args.pattern}'):
        db = connect(f)
        n_total = db.count()
        n_with_form_energy = db.count('form_energy')
        if n_with_form_energy < n_total:
            filenames.append(f)
        else:
            print(f'Skipping {os.path.basename(f)}: all {n_total} rows have form_energy')

    if filenames:
        update_phase_stability_dbs(
            filenames=filenames,
            api_key=api_key,            
            asynchronus=True,
            connect_each_row=True,
            processes=processes)
    else:
        print('All databases already complete, nothing to do.')

    os.chdir(cwd)
