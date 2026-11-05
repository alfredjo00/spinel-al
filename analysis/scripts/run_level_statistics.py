#!/usr/bin/env python3
"""Parametric run-level tests, clipping dynamics, cycle 1, ceilings, charge screen and run provenance.

Run after analyze_runs.py and audit_selections.py. Writes:

1. Tests on the run-level measures that are not limited by the exact rank tests, whose
   smallest two-sided p at five runs per strategy is 2/252 = 0.0079: Tukey HSD and a
   Westfall-Young step-down max-|t| permutation over all 45 pairs, with their
   assumption checks (reanalysis_pairwise_parametric.csv, reanalysis_assumptions.csv,
   reanalysis_focused_family.csv). Supplementary Section S5.
2. The effect of clipping by measure and by five-cycle block, and the share of the gap
   between scalar and Gibbs-vector prediction that clipping recovers
   (reanalysis_recovery.csv, reanalysis_clip_interaction.csv,
   reanalysis_clip_dynamics.csv, reanalysis_terminal_slopes.csv).
3. The first cycle, where all strategies train on the same initial dataset
   (reanalysis_cycle1_runs.csv, reanalysis_cycle1_tukey.csv, reanalysis_cycle1_overlap.csv).
4. Ceiling and saturation by block (reanalysis_ceiling_by_block.csv), ranking the
   initial dataset by ceiling alone (reanalysis_ceiling_only_reference.csv), the share
   of compositions that fail charge scenarios A (OX_CONSERVATIVE) and B (OX_PERMISSIVE)
   of Supplementary Section S2 (reanalysis_oxidation_screen.csv), and the stored energy
   above the hull (reanalysis_hull_summary.csv).
5. Subsets of 200 drawn from the initial dataset (reanalysis_random_reference.csv), the
   settings of each run (reanalysis_run_provenance.csv) and the scale of the OTC input
   of the WAE under the two normalizations (reanalysis_wae_normalisation.csv).
"""
from __future__ import annotations

import itertools
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats

from paths import DATA, HEOC

ARMS = ['de', 'de_clip', 'de_land', 'dde', 'gp', 'gp_clip', 'mogp', 'dgp', 'mc', 'mc_clip']
LABELS = {
    'de': 'DE scalar', 'de_clip': 'DE clipped scalar',
    'de_land': 'DE Gibbs vector', 'dde': 'DE vacancy input',
    'gp': 'GP scalar', 'gp_clip': 'GP clipped scalar',
    'mogp': 'GP Gibbs vector', 'dgp': 'GP vacancy input',
    'mc': 'MC scalar', 'mc_clip': 'MC clipped scalar',
}
ENDPOINTS = ['mean_otc', 'max_otc', 'top10_mean']

# Contrasts fixed by the arm structure, not by the observed ranking: the clip
# flag within each surrogate family, the representation step within each family,
# and the comparisons among the four Gibbs-target strategies.
FOCUSED = [('de_clip', 'de'), ('gp_clip', 'gp'), ('mc_clip', 'mc'),
           ('de_land', 'de_clip'), ('mogp', 'gp_clip'),
           ('dde', 'de_land'), ('dgp', 'mogp'), ('mogp', 'de_land'), ('dde', 'dgp')]

# Highest oxidation state each cation reaches in a bulk oxide. `conservative`
# is what is actually accessible on a spinel site (Mn(IV) as in LiMn2O4);
# `permissive` additionally allows states that exist in oxides but not normally
# on a spinel B site, and brackets the judgement call.
OX_CONSERVATIVE = {'Li': 1, 'Mg': 2, 'Zn': 2, 'Ca': 2, 'Cu': 2, 'Co': 3, 'Ni': 3,
                   'Al': 3, 'Cr': 3, 'Ga': 3, 'Ti': 4, 'Fe': 3, 'Mn': 4, 'Zr': 4,
                   'V': 5, 'Sn': 4, 'In': 3}
OX_PERMISSIVE = {**OX_CONSERVATIVE, 'Ni': 4, 'Fe': 4, 'Cu': 3, 'Co': 4, 'Cr': 6}
ANION_CHARGE = 96  # 48 O(-II) per cell


def tukey_hsd(values: dict[str, np.ndarray]) -> pd.DataFrame:
    """All pairwise Tukey HSD contrasts for a balanced one-way layout."""
    arms = list(values)
    k = len(arms)
    n = len(values[arms[0]])
    dfw = k * (n - 1)
    mse = sum(((values[a] - values[a].mean()) ** 2).sum() for a in arms) / dfw
    se = np.sqrt(mse / n)
    rows = []
    for a, b in itertools.combinations(arms, 2):
        diff = values[a].mean() - values[b].mean()
        q = abs(diff) / se
        rows.append(dict(arm_a=a, arm_b=b, difference=diff, q=q,
                         p_tukey=float(stats.studentized_range.sf(q, k, dfw))))
    return pd.DataFrame(rows)


def welch_t(x: np.ndarray, y: np.ndarray) -> float:
    return (x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))


def westfall_young(values: dict[str, np.ndarray], pairs, n_permutations=100_000, seed=20260907):
    """Step-down max-|t| adjusted p-values over the joint permutation null.

    Labels are reallocated across all 50 runs, so the reference distribution is
    over 50!/(5!^10) allocations rather than the 252 available to a single
    5-vs-5 contrast. That is what lifts the resolution floor. Family-wise error
    control follows Westfall and Young under subset pivotality; the studentized
    statistic keeps it usable with unequal group variances.
    """
    arms = list(values)
    index = {a: i for i, a in enumerate(arms)}
    pooled = np.concatenate([values[a] for a in arms])
    n = len(values[arms[0]])
    ia = np.array([index[a] for a, _ in pairs])
    ib = np.array([index[b] for _, b in pairs])
    observed = np.array([welch_t(values[a], values[b]) for a, b in pairs])

    rng = np.random.default_rng(seed)
    null = np.empty((n_permutations, len(pairs)))
    for start in range(0, n_permutations, 2000):
        size = min(2000, n_permutations - start)
        draw = pooled[np.argsort(rng.random((size, len(pooled))), axis=1)]
        groups = draw.reshape(size, len(arms), n)
        mean = groups.mean(axis=2)
        var = groups.var(axis=2, ddof=1) / n
        null[start:start + size] = ((mean[:, ia] - mean[:, ib])
                                    / np.sqrt(var[:, ia] + var[:, ib]))
    null = np.abs(null)

    magnitude = np.abs(observed)
    order = np.argsort(-magnitude)
    adjusted = np.empty(len(pairs))
    running = 0.0
    for position, index_ in enumerate(order):
        remaining = order[position:]
        p = ((null[:, remaining].max(axis=1) >= magnitude[index_] - 1e-12).sum() + 1) \
            / (n_permutations + 1)
        running = max(running, p)
        adjusted[index_] = running
    return observed, adjusted


def max_cation_charge(frame: pd.DataFrame, table: dict[str, int]) -> np.ndarray:
    total = np.zeros(len(frame))
    for element, charge in table.items():
        column = f'count_{element}'
        if column in frame:
            total += frame[column].fillna(0).to_numpy() * charge
    return total


def read_hull_energies() -> pd.DataFrame:
    """Pull e_above_hull, stored by the pipeline but never used in the analysis."""
    records = []
    for arm in ARMS:
        for replicate in range(1, 6):
            path = HEOC / f'spinel_rev_{arm}_s{replicate}' / 'training.db'
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
                rows = connection.execute(
                    'SELECT key_value_pairs FROM systems ORDER BY id').fetchall()
            for (raw,) in rows:
                kv = json.loads(raw)
                records.append(dict(arm=arm, replicate=replicate, cycle=kv['cycle'],
                                    otc=kv['otc_quadratic_fit'] * 100,
                                    e_above_hull=kv.get('e_above_hull'),
                                    form_energy=kv.get('form_energy')))
    return pd.DataFrame(records)


def main():
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    cycles = pd.read_csv(DATA / 'cycle_metrics.csv')
    acquired = pd.read_csv(DATA / 'acquisitions.csv')
    initial = pd.read_csv(DATA / 'initial_dataset.csv')
    scores = pd.read_csv(DATA / 'selected_scores_non_mc.csv')
    configs = json.loads((DATA / 'run_configs.json').read_text())

    # ---------------------------------------------------------------- 1. tests
    assumptions, pairwise, focused = [], [], []
    for endpoint in ENDPOINTS:
        values = {a: runs.loc[runs.arm == a, endpoint].to_numpy(float) for a in ARMS}
        groups = [values[a] for a in ARMS]
        residuals = np.concatenate([values[a] - values[a].mean() for a in ARMS])
        f_statistic, p_anova = stats.f_oneway(*groups)
        assumptions.append(dict(
            endpoint=endpoint, anova_F=f_statistic, anova_p=p_anova,
            levene_p=stats.levene(*groups).pvalue,
            shapiro_p_pooled_residuals=stats.shapiro(residuals).pvalue,
            sd_min=min(v.std(ddof=1) for v in groups),
            sd_max=max(v.std(ddof=1) for v in groups),
            rank_test_two_sided_floor=2 / 252))

        table = tukey_hsd(values)
        all_pairs = list(zip(table.arm_a, table.arm_b))
        _, wy_all = westfall_young(values, all_pairs)
        table['p_westfall_young_45'] = wy_all
        table['endpoint'] = endpoint
        pairwise.append(table)

        observed, wy_focused = westfall_young(values, FOCUSED)
        lookup = {(a, b): p for a, b, p in zip(table.arm_a, table.arm_b, table.p_tukey)}
        for (a, b), t, p in zip(FOCUSED, observed, wy_focused):
            focused.append(dict(
                endpoint=endpoint, arm_a=a, arm_b=b,
                difference=values[a].mean() - values[b].mean(), welch_t=t,
                p_westfall_young_9=p,
                p_tukey_45=lookup.get((a, b), lookup.get((b, a)))))

    pd.DataFrame(assumptions).to_csv(DATA / 'reanalysis_assumptions.csv', index=False)
    pairwise = pd.concat(pairwise, ignore_index=True)
    pairwise.to_csv(DATA / 'reanalysis_pairwise_parametric.csv', index=False)
    focused = pd.DataFrame(focused)
    focused.to_csv(DATA / 'reanalysis_focused_family.csv', index=False)

    # ----------------------------------------------- 2. clip effect and recovery
    recovery = []
    for family, (scalar, clipped, landscape) in {
            'DE': ('de', 'de_clip', 'de_land'),
            'GP': ('gp', 'gp_clip', 'mogp')}.items():
        for endpoint in ['mean_otc', 'max_otc', 'top10_mean', 'n_ge_5', 'n_ge_5.5',
                         'n_ge_6', 'mean_cap', 'mean_utilization']:
            s, c, l = (runs.loc[runs.arm == a, endpoint].mean()
                       for a in (scalar, clipped, landscape))
            recovery.append(dict(family=family, endpoint=endpoint, scalar=s,
                                 clipped=c, landscape=l, gap=l - s,
                                 clip_gain=c - s,
                                 fraction_recovered=(c - s) / (l - s) if abs(l - s) > 1e-9 else np.nan))
    pd.DataFrame(recovery).to_csv(DATA / 'reanalysis_recovery.csv', index=False)

    # Difference in differences: is the clip gain itself architecture dependent?
    interaction = []
    for (a, b), (c, d) in itertools.combinations(
            [('de_clip', 'de'), ('gp_clip', 'gp'), ('mc_clip', 'mc')], 2):
        cells = [runs.loc[runs.arm == x, 'mean_otc'].to_numpy() for x in (a, b, c, d)]
        estimate = (cells[0].mean() - cells[1].mean()) - (cells[2].mean() - cells[3].mean())
        se = np.sqrt(sum(v.var(ddof=1) / len(v) for v in cells))
        interaction.append(dict(gain_a=f'{a}-{b}', gain_b=f'{c}-{d}', estimate=estimate,
                                se=se, t=estimate / se))
    pd.DataFrame(interaction).to_csv(DATA / 'reanalysis_clip_interaction.csv', index=False)

    per_cycle = cycles.pivot_table(index='cycle', columns='arm', values='cycle_mean',
                                   aggfunc='mean')
    blocks = {'1': (1, 1), '1-5': (1, 5), '6-10': (6, 10),
              '11-15': (11, 15), '16-20': (16, 20)}

    def block_mean(arm, lo, hi):
        return per_cycle[arm].loc[lo:hi].mean()

    dynamics = [dict(block=name,
                     DE=block_mean('de_clip', *b) - block_mean('de', *b),
                     GP=block_mean('gp_clip', *b) - block_mean('gp', *b),
                     MC=block_mean('mc_clip', *b) - block_mean('mc', *b),
                     DE_vector_vs_clipped=block_mean('de_land', *b) - block_mean('de_clip', *b),
                     GP_vector_vs_clipped=block_mean('mogp', *b) - block_mean('gp_clip', *b))
                for name, b in blocks.items()]
    pd.DataFrame(dynamics).to_csv(DATA / 'reanalysis_clip_dynamics.csv', index=False)

    # Are the arms converged at the budget, or still climbing?
    slopes = [dict(arm=a, label=LABELS[a],
                   slope_cycles_13_20=np.polyfit(np.arange(8), per_cycle[a].loc[13:20], 1)[0],
                   cycle_1_5=per_cycle[a].loc[1:5].mean(),
                   cycle_16_20=per_cycle[a].loc[16:20].mean()) for a in ARMS]
    pd.DataFrame(slopes).to_csv(DATA / 'reanalysis_terminal_slopes.csv', index=False)

    # ------------------------------------------------------------- 3. cycle one
    first = acquired[acquired.cycle == 1]
    cycle_one = first.groupby(['arm', 'replicate']).otc.mean().rename('cycle1_mean').reset_index()
    cycle_one.to_csv(DATA / 'reanalysis_cycle1_runs.csv', index=False)
    values = {a: cycle_one.loc[cycle_one.arm == a, 'cycle1_mean'].to_numpy() for a in ARMS}
    cycle_one_tests = tukey_hsd(values)
    cycle_one_tests.to_csv(DATA / 'reanalysis_cycle1_tukey.csv', index=False)

    # How much of the candidate pool is shared when only the ranking rule differs.
    overlap = []
    for a, b in [('de', 'de_clip'), ('gp', 'gp_clip'), ('mc', 'mc_clip'),
                 ('de', 'gp'), ('de_land', 'mogp'), ('de', 'de_land')]:
        for replicate in range(1, 6):
            left = set(first[(first.arm == a) & (first.replicate == replicate)].formula)
            right = set(first[(first.arm == b) & (first.replicate == replicate)].formula)
            overlap.append(dict(arm_a=a, arm_b=b, replicate=replicate,
                                shared_of_10=len(left & right)))
    pd.DataFrame(overlap).to_csv(DATA / 'reanalysis_cycle1_overlap.csv', index=False)

    # -------------------------------------------- 4. ceiling, censoring, chemistry
    acquired = acquired.copy()
    acquired['saturated'] = np.isclose(acquired.vacancy_diff, 7, atol=1e-5)
    acquired['block'] = pd.cut(acquired.cycle, [0, 5, 10, 15, 20],
                               labels=['1-5', '6-10', '11-15', '16-20'])
    acquired['utilization'] = acquired.otc / acquired.cap
    ceiling = acquired.groupby(['arm', 'block'], observed=True).agg(
        mean_cap=('cap', 'mean'), mean_otc=('otc', 'mean'),
        mean_utilization=('utilization', 'mean'),
        saturated_fraction=('saturated', 'mean')).reset_index()
    ceiling.to_csv(DATA / 'reanalysis_ceiling_by_block.csv', index=False)

    # The ceiling is only informative once a composition is known to saturate.
    # On the initial pool, ranking by ceiling alone is close to useless.
    ceiling_only = []
    for k in [10, 50, 100, 200]:
        ceiling_only.append(dict(
            k=k,
            rank_by_ceiling_mean_otc=initial.nlargest(k, 'cap').otc.mean(),
            rank_by_ceiling_max_otc=initial.nlargest(k, 'cap').otc.max(),
            oracle_mean_otc=initial.nlargest(k, 'otc').otc.mean(),
            oracle_max_otc=initial.nlargest(k, 'otc').otc.max()))
    ceiling_only = pd.DataFrame(ceiling_only)
    ceiling_only['pearson_cap_otc_initial'] = stats.pearsonr(initial.cap, initial.otc)[0]
    ceiling_only['initial_saturated_fraction'] = np.isclose(initial.vacancy_diff, 7, atol=1e-5).mean()
    ceiling_only['initial_zero_release_fraction'] = (initial.vacancy_diff < 1e-6).mean()
    ceiling_only.to_csv(DATA / 'reanalysis_ceiling_only_reference.csv', index=False)

    screen = []
    for name, frame in [('initial', initial), ('acquired', acquired),
                        ('acquired_top200', acquired.nlargest(200, 'otc'))]:
        for label, table in [('conservative', OX_CONSERVATIVE), ('permissive', OX_PERMISSIVE)]:
            infeasible = max_cation_charge(frame, table) < ANION_CHARGE
            screen.append(dict(subset=name, arm='all', oxidation_set=label, n=len(frame),
                               infeasible_fraction=infeasible.mean()))
    for arm, group in acquired.groupby('arm'):
        upper = group[group.otc >= 5.5]
        for label, table in [('conservative', OX_CONSERVATIVE), ('permissive', OX_PERMISSIVE)]:
            screen.append(dict(subset='acquired', arm=arm, oxidation_set=label, n=len(group),
                               infeasible_fraction=(max_cation_charge(group, table) < ANION_CHARGE).mean()))
            if len(upper):
                screen.append(dict(subset='acquired_ge_5.5', arm=arm, oxidation_set=label,
                                   n=len(upper),
                                   infeasible_fraction=(max_cation_charge(upper, table) < ANION_CHARGE).mean()))
    pd.DataFrame(screen).to_csv(DATA / 'reanalysis_oxidation_screen.csv', index=False)

    hull = read_hull_energies()
    # Cycle 0 is the same 1000 compositions in every run, so count it once.
    hull = hull[(hull.cycle > 0) | ((hull.arm == 'de') & (hull.replicate == 1))]
    hull_summary = hull.assign(stage=np.where(hull.cycle == 0, 'initial', 'acquired')) \
        .groupby('stage').agg(n=('e_above_hull', 'size'),
                              mean_e_above_hull_cell=('e_above_hull', 'mean'),
                              median_e_above_hull_cell=('e_above_hull', 'median'))
    hull_summary['mean_e_above_hull_per_atom'] = hull_summary.mean_e_above_hull_cell / 84
    hull_summary.to_csv(DATA / 'reanalysis_hull_summary.csv')

    # ------------------------------------- 5. random reference and run provenance
    rng = np.random.default_rng(20260907)
    pool = initial.otc.to_numpy()
    draws = np.array([rng.choice(pool, 200, replace=False) for _ in range(20_000)])
    random_reference = pd.DataFrame([dict(
        budget=200, n_resamples=len(draws),
        mean_otc=draws.mean(axis=1).mean(),
        mean_otc_lo=np.percentile(draws.mean(axis=1), 2.5),
        mean_otc_hi=np.percentile(draws.mean(axis=1), 97.5),
        best_otc=draws.max(axis=1).mean(),
        best_otc_lo=np.percentile(draws.max(axis=1), 2.5),
        best_otc_hi=np.percentile(draws.max(axis=1), 97.5),
        n_ge_5=(draws >= 5).sum(axis=1).mean())])
    random_reference.to_csv(DATA / 'reanalysis_random_reference.csv', index=False)

    # `delta_gp_iters` exists only in configurations written by code versions that
    # include the vacancy-input models, so it shows which version started each run.
    provenance = pd.DataFrame([dict(
        arm=entry['arm'], replicate=entry['replicate'],
        source_revision='post_delta' if 'delta_gp_iters' in entry['config'] else 'pre_delta',
        clip_otc_physical=entry['config'].get('clip_otc_physical'),
        predict_landscape=entry['config'].get('predict_landscape'),
        model=entry['config'].get('model'),
        inverse_transform=entry['config'].get('inverse_transform'))
        for entry in configs])
    provenance.to_csv(DATA / 'reanalysis_run_provenance.csv', index=False)

    # The WAE takes the normalized OTC as an input next to the composition, and the
    # two normalizations put that input on different scales across strategies.
    normalisation = []
    for cycle in [1, 6, 11, 21]:
        for arm in ARMS:
            history = np.concatenate([
                initial.otc.to_numpy(),
                acquired.loc[(acquired.arm == arm) & (acquired.cycle < cycle), 'otc'].to_numpy()])
            normalisation.append(dict(arm=arm, before_cycle=cycle, n=len(history),
                                      minmax_channel_sd=history.std() / np.ptp(history),
                                      zscore_over_minmax=np.ptp(history) / history.std()))
    pd.DataFrame(normalisation).to_csv(DATA / 'reanalysis_wae_normalisation.csv', index=False)

    # ------------------------------------------------------------------ summary
    pd.set_option('display.width', 200)
    print('Assumption checks')
    print(pd.DataFrame(assumptions).round(4).to_string(index=False))
    print('\nDesign family, mean acquired OTC')
    print(focused[focused.endpoint == 'mean_otc'].round(4).to_string(index=False))
    print('\nTukey HSD significant pairs (of 45) per endpoint')
    print(pairwise.groupby('endpoint').p_tukey.apply(lambda s: int((s < 0.05).sum())).to_string())
    print('\nClip gain by cycle block')
    print(pd.DataFrame(dynamics).round(3).to_string(index=False))
    print('\nFraction of the scalar-to-Gibbs-vector gap recovered by clipping')
    print(pd.DataFrame(recovery).round(3).to_string(index=False))
    print('\nOxidation-state screen')
    print(pd.DataFrame(screen).query("arm == 'all'").round(4).to_string(index=False))
    print('\nHull distance')
    print(hull_summary.round(3).to_string())
    print('\nSource revision per run')
    print(provenance.pivot_table(index='arm', columns='source_revision',
                                 values='replicate', aggfunc='count',
                                 fill_value=0).to_string())
    print('\nSelected acquisition scores relative to the ceiling')
    print(scores.groupby('arm')[['above_cap', 'at_cap']].mean().round(3).to_string())
    print('\nSource databases were opened read-only; only reanalysis_* files were written.')


if __name__ == '__main__':
    main()
