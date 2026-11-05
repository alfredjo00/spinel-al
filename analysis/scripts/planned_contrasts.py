#!/usr/bin/env python3
"""The 15 primary run-level contrasts of the article, for each of the three run-level measures:

  clipping (3)     clipped minus unclipped scalar, per surrogate, paired by seed
  gibbs (8)        each Gibbs-target strategy minus the unclipped and the clipped
                   scalar strategy of the same surrogate
  vacancy (2)      vacancy input minus Gibbs vector, per surrogate
  surrogate (2)    GP minus DE for the clipped scalar and the Gibbs-vector targets

Unpaired contrasts use Welch t tests. The clipping contrasts use paired t tests
on seed-matched runs: at the same seed the two strategies of a clipping pair
start from the same fitted generator and first-cycle candidate pool, and their
run outcomes stay correlated, whereas other same-seed pairings are not. Holm
adjustment covers the 15 contrasts within each endpoint. The 95% intervals are
per-contrast, not simultaneous.

Sensitivity columns: the unpaired Welch p for every contrast and the Games-Howell
p over all 45 pairs of the ten strategies. For the clipping contrasts, the exact
sign-flip p is also given. With five pairs, it cannot fall below 2/32 = 0.0625.

Inputs: run_metrics.csv, acquisitions.csv and run_configs.json (used only to check that
the run numbers equal the configured random seeds).
Outputs: planned_contrasts.csv, and seed_correlation.csv with the same-seed correlation of
run outcomes for every pair of strategies, over cycles 1-20 from run_metrics.csv and over
cycles 11-20 from acquisitions.csv.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA

ARMS = ['de', 'de_clip', 'de_land', 'dde', 'gp', 'gp_clip', 'mogp', 'dgp', 'mc', 'mc_clip']
ENDPOINTS = ['mean_otc', 'max_otc', 'top10_mean']
CLIP_PAIRS = [('de_clip', 'de'), ('gp_clip', 'gp'), ('mc_clip', 'mc')]
CONTRASTS = (
    [('clipping', a, b, True) for a, b in CLIP_PAIRS]
    + [('gibbs', a, b, False) for a, b in [
        ('de_land', 'de'), ('dde', 'de'), ('de_land', 'de_clip'), ('dde', 'de_clip'),
        ('mogp', 'gp'), ('dgp', 'gp'), ('mogp', 'gp_clip'), ('dgp', 'gp_clip')]]
    + [('vacancy', 'dde', 'de_land', False), ('vacancy', 'dgp', 'mogp', False)]
    + [('surrogate', 'gp_clip', 'de_clip', False), ('surrogate', 'mogp', 'de_land', False)]
)


def holm(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    k = len(p_values)
    adjusted = np.minimum(1, np.maximum.accumulate(p_values[order] * (k - np.arange(k))))
    out = np.empty(k)
    out[order] = adjusted
    return out


def welch(x: np.ndarray, y: np.ndarray):
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return x.mean() - y.mean(), se, dof


def games_howell_p(x: np.ndarray, y: np.ndarray, n_groups: int = len(ARMS)) -> float:
    diff, se, dof = welch(x, y)
    return float(stats.studentized_range.sf(abs(diff) / se * np.sqrt(2), n_groups, dof))


def sign_flip_p(d: np.ndarray) -> float:
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))


def check_seeds(runs: pd.DataFrame):
    configs = json.loads((DATA / 'run_configs.json').read_text())
    seeds = {(c['arm'], c['replicate']): c['config']['_command_line_args']['seed'] for c in configs}
    mismatched = [key for key, seed in seeds.items() if key[1] != seed]
    if mismatched or len(seeds) != len(runs):
        raise ValueError(f'replicate labels do not match configured seeds: {mismatched}')


def main():
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    check_seeds(runs)

    rows = []
    for endpoint in ENDPOINTS:
        wide = runs.pivot(index='replicate', columns='arm', values=endpoint)
        block = []
        for group, a, b, paired in CONTRASTS:
            x, y = wide[a].values, wide[b].values
            diff_u, se_u, dof_u = welch(x, y)
            p_welch = 2 * stats.t.sf(abs(diff_u / se_u), dof_u)
            if paired:
                d = x - y
                diff, se, dof = d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d) - 1
                p_flip = sign_flip_p(d)
            else:
                diff, se, dof = diff_u, se_u, dof_u
                p_flip = np.nan
            half = stats.t.ppf(0.975, dof) * se
            block.append(dict(
                endpoint=endpoint, group=group, contrast=f'{a} - {b}', paired=paired,
                estimate=diff, se=se, dof=dof, ci95_low=diff - half, ci95_high=diff + half,
                t=diff / se, p=2 * stats.t.sf(abs(diff / se), dof),
                p_welch_unpaired=p_welch, p_games_howell_45=games_howell_p(x, y),
                p_sign_flip_exact=p_flip))
        block = pd.DataFrame(block)
        block.insert(block.columns.get_loc('p') + 1, 'p_holm', holm(block.p.values))
        rows.append(block)
    contrasts = pd.concat(rows, ignore_index=True)
    contrasts.to_csv(DATA / 'planned_contrasts.csv', index=False)

    acquisitions = pd.read_csv(DATA / 'acquisitions.csv')
    late = acquisitions[acquisitions.cycle >= 11].groupby(['replicate', 'arm']).otc.mean().unstack()
    corr_rows = []
    for label, wide in [('mean_otc', runs.pivot(index='replicate', columns='arm', values='mean_otc')),
                        ('max_otc', runs.pivot(index='replicate', columns='arm', values='max_otc')),
                        ('top10_mean', runs.pivot(index='replicate', columns='arm', values='top10_mean')),
                        ('mean_otc_cycles_11_20', late)]:
        for a, b in itertools.combinations(ARMS, 2):
            corr_rows.append(dict(measure=label, arm_a=a, arm_b=b,
                                  clipping_pair=(b, a) in CLIP_PAIRS or (a, b) in CLIP_PAIRS,
                                  r=np.corrcoef(wide[a], wide[b])[0, 1]))
    correlation = pd.DataFrame(corr_rows)
    correlation.to_csv(DATA / 'seed_correlation.csv', index=False)

    pd.set_option('display.width', 200)
    print(contrasts.drop(columns=['se', 'dof', 't']).round(4).to_string(index=False))
    print(correlation.groupby(['measure', 'clipping_pair']).r.agg(['mean', 'min', 'max', 'size']).round(3))


if __name__ == '__main__':
    main()
