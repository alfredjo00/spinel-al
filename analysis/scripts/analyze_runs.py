#!/usr/bin/env python3
"""Read the 50 runs, check them, and write the per-run measures, acquisitions and cycle curves.

Run first. Reads training.db, config.json, active_learning.log and
prediction_comparison.csv of every run (spinel_rev_<arm>_s<seed>) and checks that all
runs share the initial dataset, finished all 20 cycles and store OTC values that follow
from the fitted vacancy counts. OTC is the quadratic-fit OTC in wt.%, and the
statistical unit is a run.

Outputs: initial_dataset.csv, acquisitions.csv, run_metrics.csv, cycle_metrics.csv,
arm_summary.csv, data_audit.csv, clipping_log_counts.csv,
logged_prediction_diagnostics.csv, run_configs.json and top30_observed_compositions.csv.
Also writes the rank-based sensitivity analyses of Supplementary Section S5: exact
Mann-Whitney tests for all 45 pairs (pairwise_exploratory.csv) and for nine selected
contrasts (focused_exploratory.csv), and a Kruskal-Wallis permutation test
(omnibus_exploratory.csv).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from scipy import stats

from paths import DATA, HEOC

ARMS = {
    "de": "DE scalar", "de_clip": "DE clipped scalar",
    "de_land": "DE Gibbs vector", "dde": "DE vacancy input",
    "gp": "GP scalar", "gp_clip": "GP clipped scalar",
    "mogp": "GP Gibbs vector", "dgp": "GP vacancy input",
    "mc": "MC scalar", "mc_clip": "MC clipped scalar",
}
O_MASS = atomic_masses[atomic_numbers["O"]]


def read_db(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT id, numbers, key_value_pairs FROM systems ORDER BY id").fetchall()
    result = []
    for ident, blob, raw in rows:
        kv = json.loads(raw)
        z = np.frombuffer(blob, dtype=np.int32)
        counts = np.bincount(z, minlength=len(atomic_masses))
        cations = {chemical_symbols[i]: int(n) for i, n in enumerate(counts) if n and i != 8}
        formula = "".join(e + (str(n) if n != 1 else "") for e, n in sorted(cations.items())) + "O48"
        mass = float(kv["mass_"])
        result.append(dict(
            id=ident, cycle=kv["cycle"], formula=formula,
            otc=kv["otc_quadratic_fit"] * 100,
            vacancy_diff=kv["vacancy_diff_quadratic_fit"],
            mass=mass, cap=7 * O_MASS / mass * 100,
            atom_mass=float(atomic_masses[z].sum()), oxygen_count=int(counts[8]),
            entropy=kv.get("Sconf_R", np.nan), n_cations=len(cations),
            **{f"count_{e}": n for e, n in cations.items()},
        ))
    return pd.DataFrame(result).fillna({f"count_{e}": 0 for e in chemical_symbols})


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    corrected = np.empty(len(p))
    corrected[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected


def paired_signflip(a, b):
    d = a - b
    values = [abs(np.mean(d * signs)) for signs in itertools.product([-1, 1], repeat=len(d))]
    return np.mean(np.asarray(values) >= abs(d.mean()) - 1e-12)


def exact_rank_test(x, y):
    """Enumerate all 252 label allocations; average ranks handle tied outcomes."""
    n, m = len(x), len(y)
    ranks = stats.rankdata(np.r_[x, y])
    u = ranks[:n].sum() - n * (n + 1) / 2
    null = np.array([ranks[list(ix)].sum() - n * (n + 1) / 2
                     for ix in itertools.combinations(range(n + m), n)])
    p = np.mean(np.abs(null - n*m/2) >= abs(u - n*m/2) - 1e-12)
    return u, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HEOC)
    parser.add_argument("--out", type=Path, default=DATA)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records, audits, runs, curves, predictions, configs, clipping = [], [], [], [], [], [], []
    base_ref = None
    for arm in ARMS:
        for seed in range(1, 6):
            directory = args.root / f"spinel_rev_{arm}_s{seed}"
            cfg_paths = list(directory.glob("*/config.json"))
            assert len(cfg_paths) == 1, (directory, cfg_paths)
            cfg_path = cfg_paths[0]
            cfg = json.loads(cfg_path.read_text())
            assert cfg["_command_line_args"]["seed"] == seed
            assert cfg["quadratic_fit_otc"] and cfg["default_cycles"] == 21
            configs.append({"arm": arm, "replicate": seed, "config_path": str(cfg_path.relative_to(args.root)), "config": cfg})
            df = read_db(directory / "training.db")
            df["arm"], df["replicate"] = arm, seed
            base = df[df.cycle == 0].copy()
            acq = df[df.cycle.between(1, 20)].copy()
            assert len(base) == 1000 and set(df.cycle) == set(range(21))
            assert np.isfinite(df.otc).all()
            assert np.allclose(df.mass, df.atom_mass, rtol=1e-10)
            assert (df.oxygen_count == 48).all()
            assert np.allclose(df.otc, df.vacancy_diff * O_MASS / df.mass * 100, atol=1e-10)
            assert ((df.otc >= -1e-8) & (df.otc <= df.cap + 1e-8)).all()
            base_signature = base[["formula", "otc", "mass", "vacancy_diff"]].sort_values("formula").to_csv(index=False)
            if base_ref is None:
                base_ref = base_signature
                base.to_csv(args.out / "initial_dataset.csv", index=False)
            assert base_signature == base_ref, directory
            log = (cfg_path.parent / "active_learning.log").read_text(errors="replace")
            completed = "Active learning pipeline complete!" in log
            assert completed, directory
            matches = re.findall(r"Physical OTC clip: (\d+)/(\d+) candidates exceeded", log)
            clipping.append(dict(arm=arm, replicate=seed, log_entries=len(matches),
                                 pool_clipped=sum(int(a) for a, _ in matches),
                                 pool_scored=sum(int(b) for _, b in matches)))
            pc = pd.read_csv(cfg_path.parent / "prediction_comparison.csv")
            pc = pc[pc.property == "otc_quadratic_fit"].copy()
            assert not pc.duplicated(["cycle", "sample_rank"]).any()
            mismatched_cycles = []
            for cycle in range(1, 21):
                logged = np.sort(pc.loc[pc.cycle == cycle, "actual_value"] * 100)
                stored = np.sort(acq.loc[acq.cycle == cycle, "otc"])
                if len(logged) != len(stored) or not np.allclose(logged, stored, atol=1e-9):
                    mismatched_cycles.append(cycle)
            pc["actual_values_match_database_by_cycle"] = pc.cycle.map(lambda c: c not in mismatched_cycles)
            pc["arm"], pc["replicate"] = arm, seed
            elements = [col.removeprefix("comp_") for col in pc if col.startswith("comp_")]
            composition = pc[["comp_" + e for e in elements]].to_numpy()
            mass_weights = np.array([atomic_masses[atomic_numbers[e]] for e in elements])
            pc["logged_composition_mass"] = composition @ mass_weights * 84
            pc["logged_composition_cap"] = 7 * O_MASS / pc.logged_composition_mass
            # Logging differs across model families. These are NOT uniformly
            # the actual (possibly clipped) Thompson scores used for ranking.
            pc["logged_prediction_above_cap"] = pc.predicted_mean > pc.logged_composition_cap + 1e-7
            pc["logged_prediction_above_training_max"] = False
            for cycle in range(1, 21):
                train_max = df.loc[df.cycle < cycle, "otc"].max() / 100
                pc.loc[pc.cycle == cycle, "logged_prediction_above_training_max"] = pc.loc[pc.cycle == cycle, "predicted_mean"] > train_max + 1e-7
            pc["composition_noninteger_max"] = np.abs(composition * 84 - np.round(composition * 84)).max(axis=1)
            predictions.append(pc)
            records.append(acq)
            count_by_cycle = acq.groupby("cycle").size()
            audits.append(dict(arm=arm, replicate=seed, acquired=len(acq), missing_from_200=200-len(acq),
                               short_cycles=json.dumps({int(k): int(v) for k,v in count_by_cycle.items() if v != 10}),
                               initial_sha256=hashlib.sha256(base_signature.encode()).hexdigest(),
                               completed=completed, duplicate_formula_rows=int(acq.formula.duplicated().sum()),
                               prediction_rows=len(pc), prediction_actual_mismatch_cycles=json.dumps(mismatched_cycles),
                               overlap_initial=int(acq.formula.isin(base.formula).sum()),
                               db=str((directory / "training.db").relative_to(args.root))))
            run = dict(arm=arm, label=ARMS[arm], replicate=seed, n=len(acq),
                       mean_otc=acq.otc.mean(), max_otc=acq.otc.max(),
                       max_with_initial=max(acq.otc.max(), base.otc.max()),
                       top10_mean=acq.otc.nlargest(10).mean(), median_otc=acq.otc.median(),
                       mean_cycle_otc=acq.groupby("cycle").otc.mean().mean(),
                       mean_first197=acq.sort_values(["cycle", "id"]).iloc[:197].otc.mean(),
                       max_first197=acq.sort_values(["cycle", "id"]).iloc[:197].otc.max(),
                       mean_first5=acq.loc[acq.cycle <= 5,"otc"].mean(),
                       mean_last5=acq.loc[acq.cycle >= 16,"otc"].mean(),
                       mean_cap=acq.cap.mean(), mean_utilization=(acq.otc/acq.cap).mean(),
                       saturated_fraction=np.isclose(acq.vacancy_diff,7,atol=1e-5).mean(),
                       mean_entropy=acq.entropy.mean(), mean_cation_count=acq.n_cations.mean(),
                       mean_li_count=acq.get("count_Li",pd.Series(0,index=acq.index)).mean(),
                       mean_mn_count=acq.get("count_Mn",pd.Series(0,index=acq.index)).mean(),
                       logged_above_cap_fraction=pc.logged_prediction_above_cap.mean(),
                       logged_mae=(pc.predicted_mean-pc.actual_value).abs().mean()*100,
                       logged_bias=(pc.predicted_mean-pc.actual_value).mean()*100,
                       logged_composition_noninteger_max=pc.composition_noninteger_max.max())
            for threshold in [5, 5.5, 6]:
                hit = acq.otc >= threshold
                run[f"n_ge_{threshold}"] = int(hit.sum())
                run[f"first_cycle_ge_{threshold}"] = acq.loc[hit,"cycle"].min() if hit.any() else np.nan
            runs.append(run)
            for cycle in range(1, 21):
                upto = acq[acq.cycle <= cycle]
                current = acq[acq.cycle == cycle]
                curves.append(dict(arm=arm, replicate=seed, cycle=cycle, n_evaluated=len(upto),
                                   cycle_mean=current.otc.mean(), cumulative_mean=upto.otc.mean(),
                                   cumulative_max=upto.otc.max(),
                                   cumulative_max_with_initial=max(upto.otc.max(),base.otc.max()),
                                   n_ge_5_5=int((upto.otc >= 5.5).sum())))
    run_df = pd.DataFrame(runs)
    acq_df = pd.concat(records,ignore_index=True).fillna({f"count_{e}":0 for e in chemical_symbols})
    curve_df = pd.DataFrame(curves)
    pd.DataFrame(audits).to_csv(args.out / "data_audit.csv",index=False)
    pd.DataFrame(clipping).to_csv(args.out / "clipping_log_counts.csv",index=False)
    run_df.to_csv(args.out / "run_metrics.csv",index=False)
    acq_df.to_csv(args.out / "acquisitions.csv",index=False)
    curve_df.to_csv(args.out / "cycle_metrics.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(args.out / "logged_prediction_diagnostics.csv",index=False)
    (args.out / "run_configs.json").write_text(json.dumps(configs,indent=2)+"\n")
    metrics = [c for c in run_df if c not in ["arm","label","replicate"]]
    summary = run_df.groupby("arm")[metrics].agg(["mean","std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(args.out / "arm_summary.csv")
    tests = []
    for endpoint in ["mean_otc", "max_otc", "top10_mean"]:
        for a,b in itertools.combinations(ARMS,2):
            x=run_df.loc[run_df.arm==a].sort_values("replicate")[endpoint].to_numpy()
            y=run_df.loc[run_df.arm==b].sort_values("replicate")[endpoint].to_numpy()
            u,p=exact_rank_test(x,y)
            tests.append(dict(endpoint=endpoint,arm_a=a,arm_b=b,difference=x.mean()-y.mean(),
                              U=u,cliffs_delta=2*u/25-1,p_two_sided=p,
                              p_paired_signflip=paired_signflip(x,y),
                              paired_difference_sd=(x-y).std(ddof=1)))
    test_df=pd.DataFrame(tests)
    for _,ix in test_df.groupby("endpoint").groups.items():
        test_df.loc[ix,"p_holm_45"]=holm(test_df.loc[ix,"p_two_sided"])
    test_df.to_csv(args.out / "pairwise_exploratory.csv",index=False)
    # An omnibus test asks whether all ten arm distributions are exchangeable;
    # it does not identify which individual comparisons differ. Use a fixed
    # Monte Carlo stream and the add-one correction for sampled permutations.
    rng = np.random.default_rng(20260907)
    omnibus = []
    for endpoint in ["mean_otc", "max_otc", "top10_mean"]:
        groups = [run_df.loc[run_df.arm == a, endpoint].to_numpy() for a in ARMS]
        ranks = stats.rankdata(np.concatenate(groups))
        sums = ranks.reshape(10, 5).sum(axis=1)
        observed = np.sum(sums**2)
        count, n_permutations = 0, 99999
        for start in range(0, n_permutations, 1000):
            n_batch = min(1000, n_permutations-start)
            indices = np.argsort(rng.random((n_batch, 50)), axis=1)
            totals = ranks[indices].reshape(n_batch, 10, 5).sum(axis=2)
            count += int((np.sum(totals**2, axis=1) >= observed-1e-10).sum())
        h, asymptotic = stats.kruskal(*groups)
        omnibus.append(dict(endpoint=endpoint,H=h,p_asymptotic=asymptotic,
                             p_permutation=(count+1)/(n_permutations+1),
                             exceedances=count,n_permutations=n_permutations))
    omnibus_df = pd.DataFrame(omnibus)
    omnibus_df["p_holm_3_endpoints"] = holm(omnibus_df.p_permutation)
    omnibus_df.to_csv(args.out / "omnibus_exploratory.csv", index=False)
    # Nine selected contrasts on mean OTC, Holm-adjusted over the nine.
    focused=[]
    for a,b in [("de_clip","de"),("gp_clip","gp"),("mc_clip","mc"),
                ("de_land","de_clip"),("mogp","gp_clip"),
                ("dde","de_land"),("dgp","mogp"),("mogp","de_land"),("dde","dgp")]:
        x=run_df.loc[run_df.arm==a,"mean_otc"].to_numpy()
        y=run_df.loc[run_df.arm==b,"mean_otc"].to_numpy()
        u,p=exact_rank_test(x,y)
        focused.append(dict(arm_a=a,arm_b=b,difference=x.mean()-y.mean(),
                            relative_percent=100*(x.mean()/y.mean()-1),cliffs_delta=2*u/25-1,
                            p_two_sided=p,p_paired_signflip=paired_signflip(x,y)))
    focus=pd.DataFrame(focused)
    focus["p_holm_9"]=holm(focus.p_two_sided)
    focus.to_csv(args.out / "focused_exploratory.csv",index=False)
    acq_df.sort_values("otc",ascending=False).drop_duplicates("formula").head(30).to_csv(args.out / "top30_observed_compositions.csv",index=False)
    shown=["mean_otc_mean","mean_otc_std","max_otc_mean","max_otc_std","n_ge_5.5_mean","mean_cap_mean","mean_utilization_mean"]
    print(summary[shown].round(4).to_string())
    print("\nFocused exploratory comparisons:\n",focus.round(5).to_string(index=False))
    print("\nInitial mean / max:",base.otc.mean(),base.otc.max())
    print("Total acquired:",len(acq_df))


if __name__ == "__main__":
    main()
