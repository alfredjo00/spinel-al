#!/usr/bin/env python3
"""Compare the OTC spread from discrete minima and from quadratic fits across realizations.

Each evaluation uses one SQS and one vacancy arrangement for each vacancy count, so
repeated evaluations of the same composition give different Gibbs curves. For every
curve, v* is taken either as the vacancy count with the lowest evaluated energy
(discrete) or from the least-squares quadratic of the OTC extraction (quadratic), and the
within-composition standard deviation of the resulting OTC is compared between the two.

Two sources are used:

1. Repeated acquisitions: formulas acquired more than once across the 50 runs, read
   from the run training databases through acquisitions.csv. The quadratic OTC
   reproduces the stored otc_quadratic_fit.
2. The controlled SQS study (sqs_validation.zip, Supplementary Section S4): 14
   compositions with 20 independent SQS at each vacancy count. It was run with the same
   SQS, relaxation and phonon settings but without stored formation energies, so G(v)
   used E_corr = 0 and the DFT O2 reference in mu_O. Converting to the reference of the
   runs adds the exact linear term (E_O_MP - E_O_DFT) * v = +1.0127 eV * v, with
   E_O_DFT = -5.960650525 eV (half of O2.json in the metal package) and
   E_O_MP = -4.9479610 eV (Supplementary Section S3). The 20 structures at each vacancy
   count are independent, so they are combined into 20 curves in generation order.

Outputs: quadratic_fit_noise_repeats.csv (one row per repeated formula),
quadratic_fit_noise_validation.csv (one row per composition of the controlled study,
both references), quadratic_fit_noise_summary.csv and the table
quadratic_fit_validation.tex.
"""
import argparse
import json
from pathlib import Path
import sqlite3

from ase.data import atomic_masses, atomic_numbers
from ase.formula import Formula
import numpy as np
import pandas as pd

from paths import DATA, HEOC, SQS_VALIDATION as VALIDATION, TABLES

TABLE = TABLES / 'quadratic_fit_validation.tex'
VACANCIES = np.arange(8)
M_O = atomic_masses[atomic_numbers['O']]
E_O_DFT = -11.92130105 / 2  # eV per O atom, metal/.../data/O2.json
E_O_MP = -4.9479610  # eV per O atom, slope of E_corr in the run databases
N_TRIALS = 20
TIE = 1e-9
A_SITE = ['Li', 'Mg', 'Co', 'Ni', 'Cu', 'Zn', 'Ca']


def quadratic_minimum(energies):
    """Minimizing vacancy count of the least-squares quadratic, as in the OTC extraction."""
    a, b, _ = np.polyfit(VACANCIES, energies, 2)
    if abs(a) < 1e-10:
        return 0.0 if b > 0 else 7.0
    if a < 0:
        return 0.0 if np.polyval([a, b, 0], 0) < np.polyval([a, b, 0], 7) else 7.0
    return float(np.clip(-b / (2 * a), 0, 7))


def otc_pair(g_ar, g_fr, mass):
    """OTC in wt.% from the discrete minima and from the quadratic fits."""
    discrete = VACANCIES[np.argmin(g_fr)] - VACANCIES[np.argmin(g_ar)]
    quadratic = quadratic_minimum(g_fr) - quadratic_minimum(g_ar)
    return 100 * M_O * discrete / mass, 100 * M_O * quadratic / mass


def compare(sd_discrete, sd_quadratic):
    lower = int((sd_quadratic < sd_discrete - TIE).sum())
    higher = int((sd_quadratic > sd_discrete + TIE).sum())
    return lower, len(sd_discrete) - lower - higher, higher


def revision_repeats(root):
    acquisitions = pd.read_csv(DATA / 'acquisitions.csv')
    records = []
    for (arm, replicate), group in acquisitions.groupby(['arm', 'replicate']):
        path = root / f'spinel_rev_{arm}_s{replicate}' / 'training.db'
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
            rows = dict(con.execute('SELECT id,key_value_pairs FROM systems').fetchall())
        for ident, formula, mass in group[['id', 'formula', 'mass']].itertuples(index=False):
            kv = json.loads(rows[ident])
            if 'gibbs_AR_7' not in kv:
                continue
            g_ar = np.array([kv[f'gibbs_AR_{v}'] for v in VACANCIES])
            g_fr = np.array([kv[f'gibbs_FR_{v}'] for v in VACANCIES])
            discrete, quadratic = otc_pair(g_ar, g_fr, mass)
            assert abs(quadratic - 100 * kv['otc_quadratic_fit']) < 1e-9
            # OTC change from the AR minimum alone, discrete minus fitted v*_AR.
            ar_shift = 100 * M_O * (VACANCIES[np.argmin(g_ar)] - quadratic_minimum(g_ar)) / mass
            records.append((formula, discrete, quadratic, VACANCIES[np.argmin(g_ar)], ar_shift))
    records = pd.DataFrame(records, columns=['formula', 'otc_discrete', 'otc_quadratic',
                                             'v_ar_discrete', 'otc_shift_from_ar'])
    repeated = records.groupby('formula').filter(lambda group: len(group) > 1)
    table = repeated.groupby('formula').agg(
        n=('otc_quadratic', 'size'),
        mean_otc_discrete=('otc_discrete', 'mean'), mean_otc_quadratic=('otc_quadratic', 'mean'),
        sd_otc_discrete=('otc_discrete', 'std'), sd_otc_quadratic=('otc_quadratic', 'std'))
    table.to_csv(DATA / 'quadratic_fit_noise_repeats.csv')
    return table, repeated


def validation_curves(directory):
    raw = pd.read_csv(directory / 'raw_results.csv')
    assert (raw['form_energy_correction_eV'] == 0).all()
    curves = {}
    for reactor, part in raw.groupby('reactor'):
        part = part.copy()
        part['trial'] = part.groupby('n_vacancies').cumcount()
        energy = part.pivot(index='trial', columns='n_vacancies', values='gibbs_formation_energy_eV')
        assert energy.shape == (N_TRIALS, len(VACANCIES))
        curves[reactor] = energy[list(VACANCIES)].to_numpy()
    return curves


def validation(source):
    rows = []
    for directory in sorted(p for p in source.iterdir() if p.is_dir()):
        label = directory.name
        formula = label.split('_', 1)[1]
        mass = sum(atomic_masses[atomic_numbers[element]] * count
                   for element, count in Formula(formula).count().items())
        curves = validation_curves(directory)
        row = {'label': label, 'formula': formula, 'mass': mass}
        # The fallback reference puts v * E_O_DFT into G, the revision puts v * E_O_MP.
        for reference, tilt in (('original', 0.0), ('revision', E_O_MP - E_O_DFT)):
            otc = np.array([otc_pair(curves['AR'][t] + tilt * VACANCIES,
                                     curves['FR'][t] + tilt * VACANCIES, mass)
                            for t in range(N_TRIALS)])
            row[f'mean_otc_discrete_{reference}'] = otc[:, 0].mean()
            row[f'mean_otc_quadratic_{reference}'] = otc[:, 1].mean()
            row[f'sd_otc_discrete_{reference}'] = otc[:, 0].std(ddof=1)
            row[f'sd_otc_quadratic_{reference}'] = otc[:, 1].std(ddof=1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DATA / 'quadratic_fit_noise_validation.csv', index=False)
    return table


def shift_summary(records):
    """How the fitted OTC differs from the discrete one, per evaluation."""
    difference = records['otc_quadratic'] - records['otc_discrete']
    return {
        'share_same_otc': (difference.abs() < TIE).mean(),
        'share_quadratic_higher_otc': (difference > TIE).mean(),
        'share_quadratic_lower_otc': (difference < -TIE).mean(),
        'median_otc_difference': difference.median(),
        'mean_otc_difference_from_ar': records['otc_shift_from_ar'].mean(),
        'share_ar_discrete_1_or_2': records['v_ar_discrete'].isin([1, 2]).mean(),
    }


def summarize(repeats, repeated, valid):
    rows = []

    def add(source, sd_d, sd_q, mean_d, mean_q, n_records, **extra):
        lower, equal, higher = compare(sd_d, sd_q)
        rows.append({
            'source': source, 'n_compositions': len(sd_d), 'n_records': n_records,
            'mean_sd_discrete': sd_d.mean(), 'mean_sd_quadratic': sd_q.mean(),
            'median_sd_discrete': sd_d.median(), 'median_sd_quadratic': sd_q.median(),
            'rms_sd_discrete': np.sqrt((sd_d ** 2).mean()), 'rms_sd_quadratic': np.sqrt((sd_q ** 2).mean()),
            'n_quadratic_lower': lower, 'n_equal': equal, 'n_quadratic_higher': higher,
            'n_equal_both_zero': int(((sd_d < TIE) & (sd_q < TIE)).sum()),
            'mean_otc_discrete': mean_d, 'mean_otc_quadratic': mean_q, **extra,
        })

    add('revision repeats', repeats['sd_otc_discrete'], repeats['sd_otc_quadratic'],
        repeated['otc_discrete'].mean(), repeated['otc_quadratic'].mean(), len(repeated),
        **shift_summary(repeated))
    for reference in ('original', 'revision'):
        add(f'validation, {reference} reference',
            valid[f'sd_otc_discrete_{reference}'], valid[f'sd_otc_quadratic_{reference}'],
            valid[f'mean_otc_discrete_{reference}'].mean(), valid[f'mean_otc_quadratic_{reference}'].mean(),
            len(valid) * N_TRIALS)
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA / 'quadratic_fit_noise_summary.csv', index=False)
    return summary


def latex_formula(formula):
    counts = Formula(formula).count()
    order = A_SITE + ['Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ga', 'Zr', 'In', 'Sn']

    def site(elements):
        return ''.join(e if counts[e] == 1 else f'{e}$_{{{counts[e]}}}$' for e in order
                       if e in counts and e in elements)
    b_site = [e for e in counts if e not in A_SITE and e != 'O']
    return f'({site(A_SITE)})({site(b_site)})O$_{{48}}$'


def write_table(valid):
    valid = valid.sort_values('mean_otc_quadratic_revision')
    lines = [
        r'\begin{table}[htbp]\centering\small',
        r'\caption{OTC of the 14 compositions with 20 independent SQS at each vacancy count, '
        r'converted to the oxygen reference of the active learning runs and ordered by mean quadratic-fit OTC. '
        r'Mean and sample standard deviation over the 20 curves, in wt.\%.}\label{tab:quadfit}',
        r'\begin{tabular}{lrrrr}\toprule',
        r' & \multicolumn{2}{c}{Discrete minimum} & \multicolumn{2}{c}{Quadratic fit}\\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'Composition & Mean & SD & Mean & SD\\\midrule',
    ]
    for row in valid.itertuples():
        lines.append(f'{latex_formula(row.formula)} & {row.mean_otc_discrete_revision:.2f} & '
                     f'{row.sd_otc_discrete_revision:.2f} & {row.mean_otc_quadratic_revision:.2f} & '
                     f'{row.sd_otc_quadratic_revision:.2f}\\\\')
    lines += [
        r'\midrule',
        f'Mean & {valid.mean_otc_discrete_revision.mean():.2f} & {valid.sd_otc_discrete_revision.mean():.2f} & '
        f'{valid.mean_otc_quadratic_revision.mean():.2f} & {valid.sd_otc_quadratic_revision.mean():.2f}\\\\',
        r'\bottomrule\end{tabular}',
        r'\end{table}',
    ]
    TABLE.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', type=Path, default=HEOC,
                        help='directory holding spinel_rev_<arm>_s<seed>/training.db')
    parser.add_argument('--validation', type=Path, default=VALIDATION,
                        help='directory holding <label>/raw_results.csv of the controlled study')
    args = parser.parse_args()
    repeats, repeated = revision_repeats(args.root)
    valid = validation(args.validation)
    summary = summarize(repeats, repeated, valid)
    write_table(valid)
    with pd.option_context('display.width', 200, 'display.max_columns', None):
        print(summary.round(4).T.to_string())
        print(valid.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
