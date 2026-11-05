#!/usr/bin/env python3
"""All-pairs Tukey and Games-Howell tests, and the acquisitions that pass charge scenarios A and B.

Run after run_level_statistics.py. For each run-level measure, every pair of the ten
strategies is compared with Tukey's test (pooled variance) and the Games-Howell test
(Welch standard errors and degrees of freedom), both adjusted over the 45 pairs, with
their simultaneous 95% intervals. The charge scenarios cap the cation charges as in
Supplementary Section S2 (OX_CONSERVATIVE is scenario A, OX_PERMISSIVE scenario B), and a
composition passes when the capped charges can reach +96.

Inputs: run_metrics.csv, acquisitions.csv, initial_dataset.csv and
reanalysis_pairwise_parametric.csv (to check the Tukey probabilities).
Outputs: pairwise_tukey_games_howell.csv, charge_scenario_screen.csv (share of each set
that fails), charge_scenario_runs.csv (per run) and charge_scenario_summary.csv (per
strategy, means over runs).
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA
from run_level_statistics import ARMS, ENDPOINTS, OX_CONSERVATIVE, OX_PERMISSIVE

SCENARIOS = [('A', OX_CONSERVATIVE), ('B', OX_PERMISSIVE)]


def games_howell(groups):
    """Games-Howell probabilities and simultaneous 95% intervals for all pairs."""
    k = len(groups)
    rows = {}
    for i, j in itertools.combinations(range(k), 2):
        x, y = groups[i], groups[j]
        vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
        se = np.sqrt((vx + vy) / 2)
        dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
        diff = x.mean() - y.mean()
        half_width = stats.studentized_range.ppf(0.95, k, dof) * se
        rows[i, j] = (stats.studentized_range.sf(abs(diff) / se, k, dof),
                      diff - half_width, diff + half_width)
    return rows


def main():
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    acquired = pd.read_csv(DATA / 'acquisitions.csv')
    initial = pd.read_csv(DATA / 'initial_dataset.csv')
    earlier = pd.read_csv(DATA / 'reanalysis_pairwise_parametric.csv')
    tests = []
    for endpoint in ENDPOINTS:
        groups = [runs.loc[runs.arm == a, endpoint].to_numpy() for a in ARMS]
        tukey = stats.tukey_hsd(*groups)
        tukey_ci = tukey.confidence_interval()
        games = games_howell(groups)
        reference = earlier[earlier.endpoint == endpoint].set_index(['arm_a', 'arm_b'])
        for i, j in itertools.combinations(range(len(ARMS)), 2):
            a, b = ARMS[i], ARMS[j]
            assert np.isclose(tukey.pvalue[i, j], reference.loc[(a, b), 'p_tukey'], atol=1e-9)
            p_gh, gh_low, gh_high = games[i, j]
            tests.append(dict(endpoint=endpoint, arm_a=a, arm_b=b,
                              difference=tukey.statistic[i, j],
                              p_tukey_45=tukey.pvalue[i, j],
                              tukey_simultaneous_ci_low=tukey_ci.low[i, j],
                              tukey_simultaneous_ci_high=tukey_ci.high[i, j],
                              p_games_howell_45=p_gh,
                              games_howell_ci_low=gh_low,
                              games_howell_ci_high=gh_high))
    table = pd.DataFrame(tests)
    table.to_csv(DATA / 'pairwise_tukey_games_howell.csv', index=False)

    screens, screened_runs = [], []
    unique = acquired.sort_values('otc', ascending=False).drop_duplicates('formula')
    sets = {'initial': initial, 'acquisition_entries': acquired,
            'top200_acquisition_entries': acquired.nlargest(200, 'otc'),
            'top200_unique_by_best_observation': unique.head(200)}
    for label, charges in SCENARIOS:
        def deficit(frame):
            upper = sum(frame.get('count_' + e, pd.Series(0, index=frame.index)).fillna(0) * q
                        for e, q in charges.items())
            return np.maximum(96 - upper, 0)
        for name, frame in sets.items():
            d = deficit(frame)
            screens.append(dict(scenario=label, subset=name, n=len(frame),
                                n_charge_deficit=int((d > 0).sum()), charge_deficit_fraction=(d > 0).mean()))
        for (arm, replicate), frame in acquired.groupby(['arm', 'replicate']):
            passed = frame[deficit(frame) == 0]
            screened_runs.append(dict(scenario=label, arm=arm, replicate=replicate, n_acquired=len(frame),
                                      n_not_excluded=len(passed), fraction_not_excluded=len(passed) / len(frame),
                                      conditional_mean_otc=passed.otc.mean(), conditional_max_otc=passed.otc.max(),
                                      n_ge_5_5_not_excluded=int((passed.otc >= 5.5).sum()),
                                      n_ge_6_not_excluded=int((passed.otc >= 6).sum())))
    pd.DataFrame(screens).to_csv(DATA / 'charge_scenario_screen.csv', index=False)
    per_run = pd.DataFrame(screened_runs)
    per_run.to_csv(DATA / 'charge_scenario_runs.csv', index=False)
    per_run.groupby(['scenario', 'arm']).agg(
        fraction_not_excluded=('fraction_not_excluded', 'mean'),
        conditional_mean_otc=('conditional_mean_otc', 'mean'),
        conditional_max_otc=('conditional_max_otc', 'mean'),
        n_ge_5_5_not_excluded=('n_ge_5_5_not_excluded', 'mean'),
        n_ge_6_not_excluded=('n_ge_6_not_excluded', 'mean')
    ).to_csv(DATA / 'charge_scenario_summary.csv')
    print(table.groupby('endpoint')[['p_tukey_45', 'p_games_howell_45']].agg(lambda x: int((x < .05).sum())))
    print(pd.DataFrame(screens).to_string(index=False))


if __name__ == '__main__':
    main()
