#!/usr/bin/env python3
"""Measure how far selected Thompson scores lie above the surrogate mean.

Run after audit_selections.py and audit_mc_selected_scores.py. Only
the unclipped scalar strategies (DE, GP, MC) are used, because clipped scores are
capped and the Gibbs-target strategies log no OTC uncertainty. The logged uncertainty is
the same standard deviation that Thompson sampling draws from (member spread
for DE, latent posterior SD for GP, dropout SD for MC).

Inputs: selected_scores_non_mc.csv, selected_scores_mc.csv and each run's
prediction_comparison.csv. Output: selected_score_offsets.csv
(one row per selection) and a per-arm summary printed to stdout.
"""
import numpy as np
import pandas as pd

from paths import DATA, HEOC

ARMS = ['de', 'gp', 'mc']


def main():
    selected = pd.concat([pd.read_csv(DATA/'selected_scores_non_mc.csv'),
                          pd.read_csv(DATA/'selected_scores_mc.csv')])
    records = []
    for arm in ARMS:
        for seed in range(1, 6):
            config_path = next((HEOC/f'spinel_rev_{arm}_s{seed}').glob('*/config.json'))
            logged = pd.read_csv(config_path.parent/'prediction_comparison.csv')
            logged = logged[logged.property == 'otc_quadratic_fit']
            run = selected[(selected.arm == arm) & (selected.replicate == seed)].merge(
                logged[['cycle', 'sample_rank', 'predicted_mean', 'predicted_uncertainty']],
                left_on=['cycle', 'rank'], right_on=['cycle', 'sample_rank'], validate='one_to_one')
            assert np.allclose(run.predicted_mean*100, run.logged_prediction, atol=1e-6)
            assert (run.predicted_uncertainty > 0).all()
            run['uncertainty'] = run.predicted_uncertainty*100
            run['draw_offset'] = run.score - run.logged_prediction
            run['z'] = run.draw_offset/run.uncertainty
            run['mean_above_cap'] = run.logged_prediction > run.cap
            records.append(run[['arm', 'replicate', 'cycle', 'rank', 'score', 'logged_prediction',
                                'uncertainty', 'draw_offset', 'z', 'cap', 'above_cap', 'mean_above_cap']])
    data = pd.concat(records, ignore_index=True)
    data.to_csv(DATA/'selected_score_offsets.csv', index=False)
    summary = data.groupby('arm').agg(
        n=('z', 'size'), median_z=('z', 'median'), max_z=('z', 'max'),
        fraction_z_above_2=('z', lambda z: (z > 2).mean()),
        fraction_z_above_3=('z', lambda z: (z > 3).mean()),
        median_uncertainty_pp=('uncertainty', 'median'),
        median_draw_offset_pp=('draw_offset', 'median'),
        fraction_mean_above_cap=('mean_above_cap', 'mean'),
        fraction_score_above_cap=('above_cap', 'mean'))
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
