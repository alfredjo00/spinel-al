#!/usr/bin/env python3
"""Build <directory>/cycle<N>/sample.db from the selected compositions in cycle<N>/sample.csv.

Each composition is placed on the 84-atom spinel supercell by a random decoration of
the sublattices, and its stability factors, entropy, abundance and price are stored.
N is the highest-numbered cycle directory.
"""

import argparse
from ase.data import chemical_symbols
from glob import glob
import os
import pandas as pd
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.thermodynamics import\
    estimate_configurational_entropy
from metal.applications.crystals.tools import (get_average_abundance,
                                               get_price)
from build_training_database import get_stability_factors


def build_sample_database(directory='spinel', random_seed=None):
    '''Create a database of structures by randomly populating the 84-atom
    spinel supercell according to the compositions in the sample text file of
    the latest cycle folder

    Parameters
    ----------
    directory
        run directory, relative to heoc/
    random_seed
        seed for the random decoration of the sublattices
    '''

    # paths are relative to heoc/, the parent of this script's folder
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    crystal_space = CrystalSpace.read(
        os.path.join(parent_dir, directory, 'crystal_space.spa'))

    # read concentrations
    cycle = sorted([int(os.path.basename(folder).replace('cycle', ''))
                    for folder in glob(f'{parent_dir}/{directory}/cycle*')])[-1]
    df = pd.read_csv(f'{parent_dir}/{directory}/cycle{cycle}/sample.csv', sep='\t')
    df = df[[key for key in df.keys() if key in chemical_symbols]]
    concentrations_list = list(df.to_dict(orient='records'))

    # create database with random configurations
    property_calculators = {
        'stability': lambda x: get_stability_factors(
            atoms_row=x, directory=directory),
        'configurational_entropy': lambda x: estimate_configurational_entropy(
            cluster_space=crystal_space.cluster_space_list[0], structure=x),
        'abundance': get_average_abundance,
        'price': get_price,
        'cycle': lambda x: cycle
    }
    crystal_space.build_db_random_configurations(
            structure_index=0,
            concentrations_list=concentrations_list,
            dimensions=(1, 2, 3),
            filename=f'{parent_dir}/{directory}/cycle{cycle}/sample.db',
            apply_filter=True,
            property_calculators=property_calculators,
            random_seed=random_seed)


if __name__ == '__main__':

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='spinel',
                        help='run directory')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='seed of the sublattice decoration; the active learning '
                             'loop passes the run seed')
    args = parser.parse_args()

    build_sample_database(directory=args.directory, random_seed=args.random_seed)

    os.chdir(cwd)
