#!/usr/bin/env python3
"""LaTeX tables of the run outcomes, the primary contrasts, all pairwise comparisons and the charge scenarios.

Inputs: run_metrics.csv, planned_contrasts.csv, pairwise_tukey_games_howell.csv and
charge_scenario_summary.csv. Outputs: individual_runs.tex, planned_contrasts.tex,
all_pairs_comparisons.tex and charge_scenarios.tex. No statistics are recomputed.
"""
import pandas as pd

from paths import DATA, TABLES as OUT

ARMS = ['de', 'de_clip', 'de_land', 'dde', 'gp', 'gp_clip', 'mogp', 'dgp', 'mc', 'mc_clip']
LABELS = dict(zip(ARMS, ['DE scalar', 'DE clipped', 'DE Gibbs vector', 'DE vacancy input',
                       'GP scalar', 'GP clipped', 'GP Gibbs vector', 'GP vacancy input',
                       'MC scalar', 'MC clipped']))


def pvalue(x):
    if x >= 0.001:
        return f'{x:.3g}'
    if x < 1e-12:
        return '$<10^{-12}$'
    mantissa, exponent = f'{x:.2e}'.split('e')
    return f'${mantissa}' + r'\times10^{' + str(int(exponent)) + '}$'


def longtable(name, caption, label, spec, heading, rows):
    tex = '\n'.join([r'\begingroup\small', r'\setlength{\tabcolsep}{4pt}',
        r'\begin{longtable}{' + spec + '}',
        r'\caption{' + caption + r'}\label{' + label + r'}\\',
        r'\toprule', heading + r'\\\midrule', r'\endfirsthead',
        r'\multicolumn{' + str(len(spec)) + r'}{l}{\tablename\ \thetable\ continued}\\',
        r'\toprule', heading + r'\\\midrule', r'\endhead',
        r'\bottomrule\endfoot', *rows, r'\end{longtable}', r'\endgroup', ''])
    (OUT / name).write_text(tex)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    assert len(runs) == 50 and runs.groupby('arm').size().eq(5).all()
    rows = []
    for arm in ARMS:
        for _, row in runs[runs.arm.eq(arm)].sort_values('replicate').iterrows():
            rows.append(f'{LABELS[arm]} & {int(row.replicate)} & {int(row.n)} & '
                        f'{row.mean_otc:.3f} & {row.max_otc:.3f} & {row.top10_mean:.3f} & '
                        f'{int(row.n_ge_6)}' + r' \\')
    longtable('individual_runs.tex', r'Individual run outcomes. OTC values are in wt.\% and exclude the initial dataset. Yield counts new evaluations at or above 6\,wt.\% per nominal budget of 200 selections.',
              'tab:runs', 'lrrrrrr', r'Strategy & Run & New records & Mean & Maximum & Top ten & Yield $\geq6$', rows)

    endpoint_titles = [('mean_otc', 'mean acquired OTC'), ('max_otc', 'run maximum OTC'), ('top10_mean', 'top-ten mean OTC')]
    planned = pd.read_csv(DATA / 'planned_contrasts.csv')
    assert len(planned) == 45
    rows = []
    for endpoint, title in endpoint_titles:
        rows.append(r'\multicolumn{7}{l}{\emph{' + title[0].upper() + title[1:] + r'}} \\')
        for row in planned[planned.endpoint.eq(endpoint)].itertuples():
            arm_a, arm_b = row.contrast.split(' - ')
            rows.append(f'{LABELS[arm_a]} & {LABELS[arm_b]} & {row.estimate:+.3f} & '
                        f'[{row.ci95_low:+.3f}, {row.ci95_high:+.3f}] & {pvalue(row.p_holm)} & '
                        f'{pvalue(row.p_welch_unpaired)} & {pvalue(row.p_games_howell_45)}' + r' \\')
    longtable('planned_contrasts.tex',
              r'Primary contrasts between strategies. Each row compares the strategy in the first column with the one in the second. Differences are the first minus the second in OTC percentage points, with 95\% intervals that are not adjusted for multiplicity. The three clipping contrasts, listed first under each measure, are paired by seed and use paired $t$ tests with four degrees of freedom, and all other contrasts use Welch $t$ tests. $p_{\rm Holm}$ adjusts over the 15 contrasts of each measure. The sensitivity columns give the unadjusted, unpaired Welch probability ($p_{\rm Welch}$), which differs in design from the primary test only for the clipping contrasts, and the Games--Howell probability over all 45 pairs of the ten strategies ($p_{\rm GH}$).',
              'tab:planned', 'llrrrrr',
              r'Strategy & Compared with & Difference & 95\% interval & $p_{\rm Holm}$ & $p_{\rm Welch}$ & $p_{\rm GH}$', rows)

    comparisons = pd.read_csv(DATA / 'pairwise_tukey_games_howell.csv')
    assert len(comparisons) == 135
    by_pair = {endpoint: {(r.arm_a, r.arm_b): r for r in comparisons[comparisons.endpoint.eq(endpoint)].itertuples()}
               for endpoint, _ in endpoint_titles}
    pairs = list(by_pair['mean_otc'])
    assert len(pairs) == 45 and all(set(pairs) == set(by_pair[e]) for e, _ in endpoint_titles)
    rows = []
    for pair in pairs:
        cells = [f'{by_pair[e][pair].difference:+.3f} & {pvalue(by_pair[e][pair].p_games_howell_45)}'
                 for e, _ in endpoint_titles]
        rows.append(f'{LABELS[pair[0]]} & {LABELS[pair[1]]} & ' + ' & '.join(cells) + r' \\')
    longtable('all_pairs_comparisons.tex',
              r'All 45 pairwise comparisons for the three run-level measures, as a sensitivity analysis for the primary contrasts of Table~\ref{tab:planned}. Each row compares the strategy in the first column with the one in the second, and differences are the first minus the second in OTC percentage points. $p_{\rm GH}$ is the Games--Howell probability adjusted across the 45 pairs within each measure, not across measures. Simultaneous intervals and Tukey probabilities for selected contrasts are given in Table~\ref{tab:procedures}.',
              'tab:stats-all', 'llrrrrrr',
              r'& & \multicolumn{2}{c}{Mean acquired OTC} & \multicolumn{2}{c}{Run maximum OTC} & \multicolumn{2}{c}{Top-ten mean OTC}\\'
              r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}'
              r'Strategy & Compared with & Diff. & $p_{\rm GH}$ & Diff. & $p_{\rm GH}$ & Diff. & $p_{\rm GH}$', rows)

    screened = pd.read_csv(DATA / 'charge_scenario_summary.csv')
    rows = []
    for arm in ARMS:
        entries = [screened[(screened.arm == arm) & (screened.scenario == scenario)].iloc[0]
                   for scenario in ['A', 'B']]
        rows.append(LABELS[arm] + ' & ' + ' & '.join(
            f'{100*r.fraction_not_excluded:.1f} & {r.conditional_mean_otc:.2f} & {r.n_ge_6_not_excluded:.1f}' for r in entries) + r' \\')
    longtable('charge_scenarios.tex', r'Acquisitions retained under the stricter charge scenarios A and B (Section~\ref{si:screening}). Retained fractions are the percentages of new evaluations that pass, averaged over runs, mean OTC is the mean acquired OTC of the passing evaluations in wt.\% averaged over runs, and retained yields count passing evaluations at or above 6\,wt.\% per nominal budget of 200 selections.',
              'tab:charge', 'lrrrrrr', r'Strategy & A retained (\%) & A mean OTC & A yield & B retained (\%) & B mean OTC & B yield', rows)


if __name__ == '__main__':
    main()
