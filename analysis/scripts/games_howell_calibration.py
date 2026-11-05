#!/usr/bin/env python3
"""Family-wise error rate of the Games-Howell test for ten strategies with five runs each.

Simulates run outcomes with no true differences between strategies, drawn from normal
distributions with the observed standard deviation of each strategy's mean acquired OTC,
and reports how often at least one of the 45 pairwise comparisons falls below p = 0.05
(Supplementary Section S5). Input: run_metrics.csv. The result is printed.

Result with the defaults (20,000 simulations, seed 0): a family-wise error rate of 0.082 with a standard error of 0.002.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--simulations', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    sds = runs.groupby('arm').mean_otc.std().to_numpy()
    n_strategies, n_runs, n = len(sds), 5, args.simulations
    rng = np.random.default_rng(args.seed)
    outcomes = rng.normal(0, 1, (n, n_strategies, n_runs)) * sds[None, :, None]
    means, variances = outcomes.mean(2), outcomes.var(2, ddof=1)
    i, j = np.triu_indices(n_strategies, 1)
    se2 = variances[:, i] / n_runs + variances[:, j] / n_runs
    q = np.abs(means[:, i] - means[:, j]) / np.sqrt(se2 / 2)
    dof = se2 ** 2 / ((variances[:, i] / n_runs) ** 2 / (n_runs - 1)
                      + (variances[:, j] / n_runs) ** 2 / (n_runs - 1))
    p = stats.studentized_range.sf(q, n_strategies, dof)
    rate = (p < 0.05).any(1).mean()
    print(f'family-wise error rate {rate:.4f}, standard error {np.sqrt(rate * (1 - rate) / n):.4f}, '
          f'{n} simulations')


if __name__ == '__main__':
    main()
