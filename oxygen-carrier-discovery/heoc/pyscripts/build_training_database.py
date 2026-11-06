#!/usr/bin/env python3
"""Create the initial dataset, <directory>/training.db.

Draws 1,000 random spinel compositions with five to eight cations from the crystal
space (create_crystal_space.py) and stores them with their stability factors,
abundance, price and configurational entropy as cycle 0. The draw is not seeded, so
a rerun gives a different set. The initial dataset of the article is in the Zenodo
record.
"""

from ase.formula import non_metals
from ase.units import kB
import argparse
import os
import sys
import time    
sys.path.append(os.path.join(
    os.getcwd().split('oxygen-carrier-discovery')[0], 'metal'))
from metal.applications.crystals.crystal_space import CrystalSpace
from metal.applications.crystals.stability_factors import SpinelTolerance
from metal.applications.crystals.tools import (get_average_abundance,
                                               _atoms_row_to_structure,
                                               get_price)
from metal.applications.crystals.thermodynamics import\
    estimate_configurational_entropy


def get_average_metal_abundance(atoms_row):
    '''Estimate the average abundance of the metals a structure

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    '''

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # delete non-metals
    del structure[[s.index for s in structure if s.symbol in non_metals]]

    return get_average_abundance(structure)


def estimate_configurational_entropies(cluster_space, atoms_row,
                                       stirling=True):
    """Calculate an estimate of the configurational entropy based on the
    provided structure and cluster space

    Parameters
    ----------
    cluster_space
        cluster space
    atoms_row
        ASE Atoms or AtomsRow object
    stirling
        use the Stirling approximation; otherwise the result only
        approaches the macroscopic value for an infitely large supercell
    """

    # check atoms/row
    structure = _atoms_row_to_structure(atoms_row)

    # calculate configurational entropy
    configurational_entropy = estimate_configurational_entropy(
                cluster_space=cluster_space, structure=structure)

    # Sconf_R per AB2O4 formula unit (7 atoms), in units of R
    return {'configurational_entropy': configurational_entropy,
            'Sconf_R': configurational_entropy / kB / len(structure) * 7}


def get_stability_factors(atoms_row, directory='spinel', instance=False):
    """Calculate the spinel tolerance factor, the total charge, the charge
    neutrality and the oxidation-state entropy

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    directory
        data directory
    instance
        return the SpinelTolerance instance instead of a dictionary
    """
    st = SpinelTolerance(
            atoms_row=atoms_row,
            possible_a_elements=['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca', 'Li'],
            possible_b_elements=['Al', 'Cr', 'Ga', 'Ti', 'Fe', 'Mn', 'Zr', 'V', 'Sn', 'In'],)

    if instance:
        return st
    return {
        'spinel_tolerance': st.spinel_tolerance,
        'total_charge': st.total_charge,
        'delta_charge_neutrality': st.delta_charge_neutrality(),
        'oxidation_state_entropy': st.entropy()}

def build_training_database(directory='spinel'):
    '''Create a database with 1,000 random spinel compositions in 84-atom
    supercells with five to eight distinct cations

    Parameters
    ----------
    directory
        data directory
    '''

    max_total = 8

    # read crystal space
    crystal_space = CrystalSpace.read(f'{directory}/crystal_space.spa')

    # create database with all possible replacements
    property_calculators = {
        'stability': lambda x: get_stability_factors(
            atoms_row=x, directory=directory),
        'abundance': get_average_abundance,
        'metal_abundance': get_average_metal_abundance,
        'price': get_price,
        'entropies': lambda x: estimate_configurational_entropies(
            cluster_space=crystal_space.cluster_space_list[0], atoms_row=x),
        'cycle': lambda x: 0
    } 

    start = time.time()

    crystal_space.build_db_random_concentrations(
            structure_index=0,
            overwrite=False,
            fixed_symbols=['O'],
            max_total=max_total,
            min_total=5,
            n_structures=1000,
            dimensions=(1, 2, 3),
            filename=f'{directory}/training.db',
            apply_filter=True,
            exclude_filtered=True,
            property_calculators=property_calculators,
            append=True)
    # code block
    end = time.time()
    print(f"Elapsed time: {end - start:.2f} seconds")


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

    build_training_database(directory=args.directory)

    os.chdir(cwd)
