#!/usr/bin/env python3
"""Residual of the fuel-reactor reconstruction G_FR(v) = G_AR(v) - kappa*v by vacancy count.

The residual G_FR(v) - G_AR(v) + kappa*v holds the parts of the two relative Gibbs
curves that the oxygen chemical potential does not account for, mainly the change of
vibrational free energy and of the configurational entropy term between the two
reactor temperatures. The databases store only the total relative Gibbs energies, so
the two contributions cannot be separated here.

Input: the initial dataset (cycle 0) in spinel_rev_de_s1/training.db.
Output: reactor_residual_by_vacancy.csv, with the mean, the standard deviation across
compositions and the RMS of the residual in eV per supercell for v = 1 to 7.
KAPPA is the same constant as in audit_selections.py.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

from paths import DATA, HEOC

KAPPA = 1.8836231615976065


def main():
    path = HEOC / 'spinel_rev_de_s1' / 'training.db'
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        rows = [json.loads(kv) for (kv,) in con.execute('SELECT key_value_pairs FROM systems')]
    residual = np.array([[kv[f'gibbs_FR_{v}'] - kv[f'gibbs_AR_{v}'] + KAPPA * v for v in range(8)]
                         for kv in rows if kv.get('cycle', 0) == 0])
    table = pd.DataFrame({
        'vacancies': range(1, 8),
        'mean_eV': residual[:, 1:].mean(axis=0),
        'sd_eV': residual[:, 1:].std(axis=0),
        'rms_eV': np.sqrt((residual[:, 1:] ** 2).mean(axis=0)),
    })
    table['n_compositions'] = len(residual)
    table.to_csv(DATA / 'reactor_residual_by_vacancy.csv', index=False)
    print(table.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
