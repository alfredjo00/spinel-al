#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.tools import get_structures_db


def create_crystal_space(directory='wang'):
    '''Create a crystal space based on  prototype structures represented by the
    cubic and orthorhombic form of the perovskite CaMnO3 and the brownmillerite
    Ca2Mn2O5

    Parameters
    ----------
    directory
        data directory, must contain either 'wang' or 'inaki'
    '''

    if 'spinel' in directory:
        replacements = {'Mg': ['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca'],
                        'Al': ['Al', 'Cr', 'Ga', 'Ce', 'Ti', 'Fe', 'Mn', 'Zr', 'V'],
                        'O': ['O', 'X']}
    else:
        raise ValueError(f'The directory must be "spinel", not'
                         f' {directory}')

    # read prototype structures
    structure_list = list(get_structures_db(f'{directory}/prototypes.db'))
    print(structure_list)

    # define chemical symbols
    chemical_symbols_list = [[replacements[s] for s in structure.symbols] for
                             structure in structure_list]

    # define cutoffs
    cutoffs_list = [[6.0, 4.5]]

    # define filters
    filters_list = [[
        {'key': 'spinel_tolerance', 'symbol': '>', 'value': 1},
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
                        choices=['spinel'],
                        help='data directory')

    args = parser.parse_args()

    # Create a crystal space based on  prototype structures represented by the
    # cubic and orthorhombic form of the perovskite CaMnO3 and the
    # brownmillerite Ca2Mn2O5
    create_crystal_space(directory=args.directory)

    os.chdir(cwd)
