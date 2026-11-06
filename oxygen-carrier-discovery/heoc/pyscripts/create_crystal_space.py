#!/usr/bin/env python3
"""Create the crystal space of the spinel compositions, <directory>/crystal_space.spa.

The space is built on the MgAl2O4 spinel prototype in <directory>/prototypes.db
(build_prototype_database.py). A sites take Mg, Zn, Cu, Co, Ni, Ca or Li, B sites take
Al, Cr, Ga, Ti, Fe, Mn, Zr, V, Sn or In, and oxygen sites take O or a vacancy (X).
Compositions outside 0.77 <= spinel tolerance <= 1.06, or without a charge-neutral
assignment of oxidation states, are filtered out.
"""

import argparse
import os
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.tools import get_structures_db

A_ELEMENTS = ['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca', 'Li']
B_ELEMENTS = ['Al', 'Cr', 'Ga', 'Ti', 'Fe', 'Mn', 'Zr', 'V', 'Sn', 'In']


def create_crystal_space(directory='spinel'):
    '''Create the spinel crystal space and write it to <directory>/crystal_space.spa

    Parameters
    ----------
    directory
        data directory, which holds prototypes.db
    '''

    replacements = {'Mg': A_ELEMENTS, 'Al': B_ELEMENTS, 'O': ['O', 'X']}

    # read prototype structures
    structure_list = list(get_structures_db(f'{directory}/prototypes.db'))

    # define chemical symbols
    chemical_symbols_list = [[replacements[s] for s in structure.symbols] for
                             structure in structure_list]

    # define cutoffs
    cutoffs_list = [[6.0, 4.5]]

    # filters: compositions matching any of these are excluded
    filters_list = [[
        {'key': 'spinel_tolerance', 'symbol': '>', 'value': 1.06},
        {'key': 'spinel_tolerance', 'symbol': '<', 'value': 0.77},
        {'key': 'delta_charge_neutrality', 'symbol': '=', 'value': False}
        ]]

    # create and write a crystal space
    crystal_space = CrystalSpace(db_path='.',
                                 structure_list=structure_list,
                                 chemical_symbols_list=chemical_symbols_list,
                                 cutoffs_list=cutoffs_list,
                                 filters_list=filters_list)
    crystal_space.write(f'{directory}/crystal_space.spa')


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

    create_crystal_space(directory=args.directory)

    os.chdir(cwd)
