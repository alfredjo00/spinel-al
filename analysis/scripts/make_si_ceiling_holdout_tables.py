#!/usr/bin/env python3
"""Supplementary Tables of the ceiling-only runs and the held-out test.

Inputs: ceiling_run_metrics.csv (analyze_ceiling_baseline.py) and
holdout_test/holdout_censoring.csv, the held-out predictions made with the active
learning code (not public, see the README).
Outputs: ceiling_runs.tex and holdout_test.tex.
"""
import pandas as pd

from paths import DATA as SRC, TABLES as OUT

MODELS = [("de", "DE scalar"), ("de_cap", "DE clipped"), ("de_land", "DE Gibbs vector"),
          ("dde", "DE vacancy input"), ("gp", "GP scalar"), ("gp_cap", "GP clipped"),
          ("mogp", "GP Gibbs vector"), ("dgp", "GP vacancy input")]


def ceiling_table():
    runs = pd.read_csv(SRC / "ceiling_run_metrics.csv").sort_values("replicate")
    assert len(runs) == 5
    cols = ["mean_otc", "max_otc", "top10_mean", "n_ge_6", "mean_cap", "saturated_fraction",
            "zero_fraction", "infeasible_conservative"]

    def row(label, r):
        return (f"{label} & {r.mean_otc:.2f} & {r.max_otc:.2f} & {r.top10_mean:.2f} & "
                f"{r.n_ge_6:.0f} & {r.mean_cap:.2f} & {100 * r.saturated_fraction:.1f} & "
                f"{100 * r.zero_fraction:.1f} & {100 * r.infeasible_conservative:.1f}\\\\")

    lines = [row(f"Seed {int(r.replicate)}", r) for _, r in runs.iterrows()]
    mean = runs[cols].mean()
    mean_row = row("Mean", mean).replace(f"& {mean.n_ge_6:.0f} &", f"& {mean.n_ge_6:.1f} &")
    text = "\n".join([
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Ceiling-only reference runs. OTC and ceiling $C$ in wt.\%, over the 200 new "
        r"evaluations of each run. $N_{\geq6}$ counts evaluations at or above 6\wt{}. The last three "
        r"columns give the percentage of evaluations that release all seven vacancies, release none, "
        r"and fail the conservative charge scenario~A of Section~\ref{si:screening}.}\label{tab:ceiling}",
        r"\begin{tabular}{lrrrrrrrr}\toprule",
        r"Run & Mean OTC & Max OTC & Top-10 mean & $N_{\geq6}$ & Mean $C$ & $\Delta v=7$ (\%) & "
        r"$\Delta v=0$ (\%) & Scenario A (\%)\\\midrule",
        *lines, r"\midrule", mean_row,
        r"\bottomrule\end{tabular}", r"\end{table}", ""])
    (OUT / "ceiling_runs.tex").write_text(text)


def num(value):
    text = f"{value:.2f}"
    return f"${text}$" if value < -0.005 else text.replace("-0.00", "0.00")


def holdout_table():
    d = pd.read_csv(SRC / "holdout_test" / "holdout_censoring.csv")
    m = d.groupby(["model", "set"]).mean(numeric_only=True)
    lines = []
    for key, label in MODELS:
        r = lambda s, c: m.loc[(key, s), c]  # noqa: E731
        lines.append(
            f"{label} & {num(r('random', 'spearman_all'))} & {num(r('acq_mc', 'spearman_all'))} & "
            f"{num(r('acq', 'spearman_all'))} & {num(r('acq_loo', 'spearman_all'))} & "
            f"{r('random', 'mae_zero'):.2f} & {r('acq_mc', 'mae_ceil'):.2f} & "
            f"{r('acq_mc', 'mae_mid'):.2f} & {r('acq_mc', 'top10_mean_true'):.2f}\\\\")
        if key == "dde":
            lines.append(r"\midrule")
    best = m.xs("acq_mc", level="set").top10_best_possible.iloc[0]
    n = {s: int(m.xs(s, level="set").n_all.iloc[0]) for s in ("random", "acq_mc", "acq")}
    text = "\n".join([
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Held-out test of surrogates trained on the initial dataset, means over five "
        r"training seeds. Spearman correlations between predicted and evaluated OTC are given for "
        rf"the {n['random']} random-reference compositions, the {n['acq_mc']:,} compositions acquired by "
        rf"the MC strategies, all {n['acq']:,} compositions acquired by the ten strategies, and the "
        r"same set without the acquisitions of the model's own strategy family (Excl.\ own). Mean "
        r"absolute errors (MAE) in wt.\% are for compositions whose evaluated OTC is zero (random "
        r"set), equals the ceiling, or lies between zero and the ceiling (both MC-acquired set). "
        r"The last column is the mean evaluated OTC of the ten "
        rf"MC-acquired compositions with the highest predictions, against {best:.2f}\wt{{}} for the "
        r"ten best. Clipped denotes the scalar prediction capped at $C$.}\label{tab:holdout}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrr}\toprule",
        r"& \multicolumn{4}{c}{Spearman} & \multicolumn{3}{c}{MAE (wt.\%)} & Top-10\\"
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}",
        r"Surrogate & Random & MC-acq. & All acq. & Excl.\ own & Zero & At $C$ & Below $C$ & "
        r"mean\\\midrule",
        *lines,
        r"\bottomrule\end{tabular}", r"\end{table}", ""])
    (OUT / "holdout_test.tex").write_text(text)


if __name__ == "__main__":
    ceiling_table()
    holdout_table()
