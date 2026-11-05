#!/usr/bin/env python3
"""Run-level measures of the ceiling-only reference runs (spinel_rev_ceil_s1 to s5).

These runs rank the candidates of each cycle by the ceiling C(x) = 7 m_O / m_cell alone,
without a surrogate, and are otherwise set up like the ten strategies (Supplementary
Section S7). The measures follow analyze_runs.py, so the rows can be read
next to run_metrics.csv. The reference is outside the 15 primary contrasts, so the
comparisons written here are descriptive (unadjusted Welch t, mean and observed range).

Inputs: the five run directories, and run_metrics.csv, acquisitions.csv and
initial_dataset.csv from analyze_runs.py.
Outputs: ceiling_run_metrics.csv, ceiling_cycle_metrics.csv, ceiling_acquisitions.csv,
ceiling_comparison.csv and ceiling_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analyze_runs import ARMS, O_MASS, read_db
from paths import DATA, HEOC
from run_level_statistics import ANION_CHARGE, OX_CONSERVATIVE, OX_PERMISSIVE, max_cation_charge


def run_measures(acq: pd.DataFrame, base: pd.DataFrame) -> dict:
    run = dict(n=len(acq), mean_otc=acq.otc.mean(), max_otc=acq.otc.max(),
               max_with_initial=max(acq.otc.max(), base.otc.max()),
               top10_mean=acq.otc.nlargest(10).mean(), median_otc=acq.otc.median(),
               mean_first5=acq.loc[acq.cycle <= 5, "otc"].mean(),
               mean_last5=acq.loc[acq.cycle >= 16, "otc"].mean(),
               mean_cap=acq.cap.mean(), max_cap=acq.cap.max(),
               mean_utilization=(acq.otc / acq.cap).mean(),
               saturated_fraction=np.isclose(acq.vacancy_diff, 7, atol=1e-5).mean(),
               zero_fraction=(acq.vacancy_diff < 1e-6).mean(),
               mean_entropy=acq.entropy.mean(), mean_cation_count=acq.n_cations.mean(),
               mean_li_count=acq.get("count_Li", pd.Series(0, index=acq.index)).mean(),
               mean_mn_count=acq.get("count_Mn", pd.Series(0, index=acq.index)).mean(),
               pearson_cap_otc=stats.pearsonr(acq.cap, acq.otc)[0],
               infeasible_conservative=(max_cation_charge(acq, OX_CONSERVATIVE) < ANION_CHARGE).mean(),
               infeasible_permissive=(max_cation_charge(acq, OX_PERMISSIVE) < ANION_CHARGE).mean())
    for threshold in [5, 5.5, 6]:
        hit = acq.otc >= threshold
        run[f"n_ge_{threshold}"] = int(hit.sum())
        run[f"first_cycle_ge_{threshold}"] = acq.loc[hit, "cycle"].min() if hit.any() else np.nan
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HEOC)
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()

    initial = pd.read_csv(args.out / "initial_dataset.csv")
    base_ref = initial[["formula", "otc", "mass", "vacancy_diff"]].sort_values("formula").reset_index(drop=True)
    runs, curves, records = [], [], []
    for seed in range(1, 6):
        directory = args.root / f"spinel_rev_ceil_s{seed}"
        cfg_paths = list(directory.glob("*/config.json"))
        assert len(cfg_paths) == 1, (directory, cfg_paths)
        cfg = json.loads(cfg_paths[0].read_text())
        assert cfg["_command_line_args"]["seed"] == seed
        assert cfg["quadratic_fit_otc"] and cfg["default_cycles"] == 21
        log = (cfg_paths[0].parent / "active_learning.log").read_text(errors="replace")
        assert "Active learning pipeline complete!" in log, directory
        df = read_db(directory / "training.db")
        df["arm"], df["replicate"] = "ceil", seed
        base = df[df.cycle == 0]
        acq = df[df.cycle.between(1, 20)].copy()
        assert len(base) == 1000 and set(df.cycle) == set(range(21))
        assert (acq.groupby("cycle").size() == 10).all()
        assert (df.oxygen_count == 48).all() and np.isfinite(df.otc).all()
        assert np.allclose(df.otc, df.vacancy_diff * O_MASS / df.mass * 100, atol=1e-10)
        signature = base[["formula", "otc", "mass", "vacancy_diff"]].sort_values("formula").reset_index(drop=True)
        assert signature.formula.equals(base_ref.formula)
        assert np.allclose(signature[["otc", "mass", "vacancy_diff"]], base_ref[["otc", "mass", "vacancy_diff"]], atol=1e-9)
        assert not acq.formula.isin(base.formula).any() and not acq.formula.duplicated().any()
        run = dict(arm="ceil", label="Ceiling only", replicate=seed,
                   config_model=cfg["_command_line_args"].get("model"),
                   **run_measures(acq, base))
        runs.append(run)
        records.append(acq)
        for cycle in range(1, 21):
            upto, current = acq[acq.cycle <= cycle], acq[acq.cycle == cycle]
            curves.append(dict(arm="ceil", replicate=seed, cycle=cycle,
                               cycle_mean=current.otc.mean(), cycle_mean_cap=current.cap.mean(),
                               cumulative_mean=upto.otc.mean(), cumulative_max=upto.otc.max(),
                               cycle_saturated_fraction=np.isclose(current.vacancy_diff, 7, atol=1e-5).mean(),
                               cycle_zero_fraction=(current.vacancy_diff < 1e-6).mean()))
    run_df = pd.DataFrame(runs)
    acq_df = pd.concat(records, ignore_index=True)
    acq_df = acq_df.fillna({c: 0 for c in acq_df if c.startswith("count_")})

    # Descriptive comparison with each paper arm, same per-run measures.
    paper = pd.read_csv(args.out / "run_metrics.csv")
    paper_acq = pd.read_csv(args.out / "acquisitions.csv")
    comparison = []
    for arm in ARMS:
        other = paper[paper.arm == arm]
        for measure in ["mean_otc", "max_otc", "top10_mean", "n_ge_5.5", "n_ge_6",
                        "mean_cap", "saturated_fraction"]:
            x, y = run_df[measure].to_numpy(float), other[measure].to_numpy(float)
            p = stats.ttest_ind(x, y, equal_var=False).pvalue if np.ptp(np.r_[x, y]) > 0 else np.nan
            comparison.append(dict(arm=arm, label=ARMS[arm], measure=measure,
                                   ceil_mean=x.mean(), ceil_min=x.min(), ceil_max=x.max(),
                                   arm_mean=y.mean(), arm_min=y.min(), arm_max=y.max(),
                                   diff_ceil_minus_arm=x.mean() - y.mean(), welch_p_unadjusted=p))
    comparison = pd.DataFrame(comparison)

    unique = acq_df.formula.nunique()
    shared_seeds = acq_df.groupby("formula").replicate.nunique()
    paper_formulas = set(paper_acq.formula)
    overlap_paper = acq_df.formula.isin(paper_formulas)
    initial_top = initial.assign(cap=7 * O_MASS / initial.mass * 100).nlargest(10, "cap")
    summary = dict(
        n_runs=len(run_df), n_acquisitions=len(acq_df), unique_compositions=int(unique),
        compositions_picked_by_more_than_one_seed=int((shared_seeds > 1).sum()),
        acquisitions_also_acquired_by_paper_arms=int(overlap_paper.sum()),
        mean_otc_of_those_overlaps=float(acq_df.loc[overlap_paper, "otc"].mean()) if overlap_paper.any() else None,
        pooled_pearson_cap_otc=float(stats.pearsonr(acq_df.cap, acq_df.otc)[0]),
        initial_top10_by_cap_mean_otc=float(initial_top.otc.mean()),
        pooled_saturated_fraction=float(np.isclose(acq_df.vacancy_diff, 7, atol=1e-5).mean()),
        pooled_zero_fraction=float((acq_df.vacancy_diff < 1e-6).mean()),
        top_compositions=acq_df.nlargest(10, "otc")[["replicate", "cycle", "formula", "otc", "cap"]].to_dict("records"),
    )
    run_df.to_csv(args.out / "ceiling_run_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(args.out / "ceiling_cycle_metrics.csv", index=False)
    acq_df.to_csv(args.out / "ceiling_acquisitions.csv", index=False)
    comparison.to_csv(args.out / "ceiling_comparison.csv", index=False)
    (args.out / "ceiling_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    pd.set_option("display.width", 200)
    print(run_df[["replicate", "mean_otc", "max_otc", "top10_mean", "n_ge_5.5", "n_ge_6", "mean_cap",
                  "mean_utilization", "saturated_fraction", "zero_fraction", "mean_entropy",
                  "infeasible_conservative"]].round(3).to_string(index=False))
    print(comparison[comparison.measure.isin(["mean_otc", "max_otc", "top10_mean"])]
          .pivot(index="label", columns="measure", values="diff_ceil_minus_arm").round(2))
    print(json.dumps({k: v for k, v in summary.items() if k != "top_compositions"}, indent=2, default=float))


if __name__ == "__main__":
    main()
