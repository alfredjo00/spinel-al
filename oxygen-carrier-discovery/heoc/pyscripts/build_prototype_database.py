#!/usr/bin/env python3
"""Create <directory>/prototypes.db with the cubic MgAl2O4 spinel from the Materials Project.

Needs a Materials Project API key in PMG_MAPI_KEY.
"""

import argparse
import os
from pymatgen.core import SETTINGS
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.tools import build_prototype_db


def build_prototype_database(directory='spinel'):
    '''Create a database with the cubic MgAl2O4 spinel as prototype structure

    Parameters
    ----------
    directory
        data directory
    '''

    # read api key, since pymatgen typically fails to do so
    api_key = SETTINGS['PMG_MAPI_KEY']

    build_prototype_db(
        filename=f'{directory}/prototypes.db',
        formulas=['MgAl2O4'],
        crystal_systems=['Cubic'],
        properties=['energy', 'energy_per_atom'],
        make_ideal=True,
        api_key=api_key)


if __name__ == '__main__':

    cwd = os.getcwd()
    parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(parent_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        type=str,
                        default='spinel',
                        help='data directory')
    args = parser.parse_args()

    build_prototype_database(directory=args.directory)

    os.chdir(cwd)
