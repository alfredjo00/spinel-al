#!/usr/bin/env python3

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

    # Changed here: return Sconf_R with factor 7 (for spinels) instead of 5 (for perovskites)
    return {'configurational_entropy': configurational_entropy,
            'Sconf_R': configurational_entropy / kB / len(structure) * 7}


def get_stability_factors(atoms_row, directory='spinel', instance=False):
    """Calculate various stability factors, including the Bartel and
    Goldschmidt tolerance as well as the charge neutrality and configurational
    entropy

    Parameters
    ----------
    atoms_row
        ASE Atoms or AtomsRow object
    directory
        data directory
    instance
        return instance of PerovskiteTolerance instead of dictionrary
    """
    if 'spinel' in directory:
        st = SpinelTolerance(
                atoms_row=atoms_row,
                possible_a_elements=['Mg', 'Zn', 'Cu', 'Co', 'Ni', 'Ca', 'Li'],
                possible_b_elements=['Al', 'Cr', 'Ga', 'Ti', 'Fe', 'Mn', 'Zr', 'V', 'Sn', 'In'],)
        

    else:
        raise ValueError(f'The directory must be either "wang" or "inaki", not'
                         f' {directory}')

    if instance:
        return st
    return {
        'spinel_tolerance': st.spinel_tolerance,
        'total_charge': st.total_charge,
        'delta_charge_neutrality': st.delta_charge_neutrality(),
        'oxidation_state_entropy': st.entropy()}

def build_training_database(directory='spinel'):
    '''Create a database with training structures in the form of ABO3
    perovskites with 80 atoms per supercells and up to six distinct metallic
    elements

    Parameters
    ----------
    directory
        data directory
    '''

    if 'spinel' in directory:
        max_total = 8
    else:
        raise ValueError(f'The directory must be either "spinel", not'
                         f' {directory}')

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

    # Create a database with training structures in the form of ABO3
    # perovskites with 80 atoms per supercells and up to six distinct metallic
    # elements
    build_training_database(directory=args.directory)

    os.chdir(cwd)
