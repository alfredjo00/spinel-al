#!/usr/bin/env python3
"""Measure how narrowly acquired compositions sit at the edges of the modeled vacancy interval.

A quadratic is fitted by least squares to the eight evaluated relative Gibbs energies
of each reactor, as in the OTC extraction. The AR margin is the fitted G_AR(7) - G_AR(0)
and the FR margin the fitted G_FR(0) - G_FR(7), both in meV per atom of the 84-atom
supercell. A small positive AR margin means the fully oxygenated state is only just
favored over the seven-vacancy state in the AR, and a small positive FR margin that the
seven-vacancy state is only just favored in the FR.

Inputs: acquisitions.csv and the training databases of the runs. Outputs: ar_margin.csv,
with one row per group and margin, giving the share of compositions below each threshold,
and realization_noise.csv. The noise table uses formulas
acquired more than once, whose records come from independent structural realizations,
and gives the median within-formula standard deviation of each relative AR Gibbs energy
and of the fitted AR margin.
"""
import json
import sqlite3

import numpy as np
import pandas as pd

from paths import DATA, HEOC

THRESHOLDS = (0.0, 5.0, 10.0, 20.0)  # meV per atom
N_ATOMS = 84
VACANCIES = np.arange(8)


def read_margins(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        rows = con.execute('SELECT id,key_value_pairs FROM systems').fetchall()
    margins = {}
    for ident, kv in rows:
        kv = json.loads(kv)
        if 'gibbs_AR_7' in kv:
            ar = fitted_span(kv, 'AR')
            fr = -fitted_span(kv, 'FR')
            margins[ident] = (kv.get('cycle', 0), kv['otc_quadratic_fit'] * 100, ar, fr,
                              [kv[f'gibbs_AR_{v}'] for v in VACANCIES])
    return margins


def fitted_span(kv, reactor):
    energies = [kv[f'gibbs_{reactor}_{v}'] for v in VACANCIES]
    fit = np.polyval(np.polyfit(VACANCIES, energies, 2), [0, 7])
    return (fit[1] - fit[0]) / N_ATOMS * 1000


def summarize(name, margins, label):
    row = {'group': name, 'margin': label, 'n': len(margins), 'median_margin_meV_per_atom': margins.median()}
    for threshold in THRESHOLDS:
        row[f'share_below_{threshold:g}_meV'] = (margins < threshold).mean()
    return row


def main():
    acquisitions = pd.read_csv(DATA / 'acquisitions.csv')
    parts = []
    for (arm, replicate), group in acquisitions.groupby(['arm', 'replicate']):
        margins = read_margins(HEOC / f'spinel_rev_{arm}_s{replicate}' / 'training.db')
        group = group.copy()
        missing = (None, None, float('nan'), float('nan'), [float('nan')] * len(VACANCIES))
        group['ar'] = group['id'].map(lambda ident: margins.get(ident, missing)[2])
        group['fr'] = group['id'].map(lambda ident: margins.get(ident, missing)[3])
        for v in VACANCIES:
            group[f'gibbs_AR_{v}'] = group['id'].map(lambda ident, v=v: margins.get(ident, missing)[4][v])
        parts.append(group)
    acquired = pd.concat(parts)
    initial = read_margins(HEOC / 'spinel_rev_de_s1' / 'training.db')
    initial = pd.DataFrame([(otc, ar, fr) for cycle, otc, ar, fr, _ in initial.values() if cycle == 0],
                           columns=['otc', 'ar', 'fr'])
    top = acquired.nlargest(200, 'otc')
    groups = {
        'initial dataset': initial,
        'initial dataset, OTC >= 3 wt.%': initial[initial['otc'] >= 3],
        'all acquisitions': acquired.dropna(subset=['ar', 'fr']),
        '200 highest-OTC acquisitions': top.dropna(subset=['ar', 'fr']),
    }
    rows = []
    for name, group in groups.items():
        rows.append(summarize(name, group['ar'], 'AR'))
        rows.append(summarize(name, group['fr'], 'FR'))
        rows.append(summarize(name, group[['ar', 'fr']].min(axis=1), 'smaller of AR and FR'))
    table = pd.DataFrame(rows)
    table.to_csv(DATA / 'ar_margin.csv', index=False)
    print(table.round(3).to_string(index=False))

    repeated = acquired.dropna(subset=['ar']).groupby('formula').filter(lambda group: len(group) > 1)
    columns = [f'gibbs_AR_{v}' for v in VACANCIES]
    spread = repeated.groupby('formula')[columns + ['ar']].std().median()
    noise = pd.DataFrame({
        'quantity': [f'G_AR({v}) - G_AR(0), eV per supercell' for v in VACANCIES] + ['fitted AR margin, meV per atom'],
        'median_within_formula_sd': spread.values,
    })
    noise['n_formulas'] = repeated['formula'].nunique()
    noise['n_records'] = len(repeated)
    noise.to_csv(DATA / 'realization_noise.csv', index=False)
    print(noise.round(3).to_string(index=False))
    print('top 200 minimum AR margin, meV per atom:', round(top['ar'].min(), 1))


if __name__ == '__main__':
    main()
