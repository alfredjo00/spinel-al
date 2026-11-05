"""Supplementary Tables of the multiplicity procedures, prediction errors and ceiling-only blocks.

Reads CSVs from derived_data/ and writes procedure_comparison.tex,
prediction_diagnostics.tex and ceiling_blocks.tex. No statistics are recomputed.
"""
import csv
from collections import defaultdict

from paths import DATA as SRC, TABLES as OUT

LABEL = {
    "de": "DE scalar", "de_clip": "DE clipped", "de_land": "DE Gibbs vector",
    "dde": "DE vacancy input", "gp": "GP scalar", "gp_clip": "GP clipped",
    "mogp": "GP Gibbs vector", "dgp": "GP vacancy input",
    "mc": "MC scalar", "mc_clip": "MC clipped",
}
ORDER = ["de", "de_clip", "de_land", "dde", "gp", "gp_clip", "mogp", "dgp", "mc", "mc_clip"]


def read(name):
    with open(SRC / name) as fh:
        return list(csv.DictReader(fh))


def procedures_table():
    """Every multiplicity procedure applied to the nine focused contrasts."""
    focused = {(r["arm_a"], r["arm_b"]): r for r in read("focused_exploratory.csv")}
    wy = {(r["arm_a"], r["arm_b"]): r for r in read("reanalysis_pairwise_parametric.csv")
          if r["endpoint"] == "mean_otc"}
    gh = {(r["arm_a"], r["arm_b"]): r for r in read("pairwise_tukey_games_howell.csv")
          if r["endpoint"] == "mean_otc"}
    planned = {tuple(r["contrast"].split(" - ")): r for r in read("planned_contrasts.csv")
               if r["endpoint"] == "mean_otc"}

    def lookup(table, a, b):
        """Contrast direction is not fixed across the source files."""
        if (a, b) in table:
            return table[(a, b)], 1.0
        if (b, a) in table:
            return table[(b, a)], -1.0
        return None, 0.0

    rows = []
    for a, b in focused:
        g, sign = lookup(gh, a, b)
        w, _ = lookup(wy, a, b)
        p, _ = lookup(planned, a, b)
        f = focused[(a, b)]
        diff = float(f["difference"])
        lo, hi = float(g["games_howell_ci_low"]), float(g["games_howell_ci_high"])
        if sign < 0:
            lo, hi = -hi, -lo
        rows.append((
            LABEL[a], LABEL[b], diff, lo, hi,
            f"{float(p['p_holm']):.4f}" if p else "--",
            float(g["p_games_howell_45"]), float(g["p_tukey_45"]),
            float(w["p_westfall_young_45"]), float(f["p_holm_9"]),
        ))
    rows.sort(key=lambda r: -abs(r[2]))

    body = "".join(
        f"{a} & {b} & ${d:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ & {p_plan} & "
        f"{p_gh:.4f} & {p_tk:.4f} & {p_wy:.4f} & {p_h:.4f}\\\\\n"
        for a, b, d, lo, hi, p_plan, p_gh, p_tk, p_wy, p_h in rows
    )
    return (
        "\\begingroup\\small\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{table}[htbp]\\centering\n"
        "\\caption{Mean acquired OTC contrasts under each multiplicity procedure. "
        "Each row compares the strategy in the first column with the one in the second. "
        "Differences are the first minus the second in OTC percentage points, and the interval is the Games--Howell "
        "simultaneous 95\\% interval. $p_{\\rm prim}$ is the Holm-adjusted probability of the primary "
        "analysis (Table~\\ref{tab:planned}), paired by seed for the clipping contrasts, and a dash "
        "marks a contrast outside the primary set. "
        "$p_{\\rm GH}$ and $p_{\\rm Tukey}$ adjust over all 45 pairs of "
        "the ten strategies. $p_{\\rm WY}$ is a Westfall--Young permutation adjustment over the same "
        "45 pairs. $p_{\\rm Holm9}$ restricts the family to these nine selected contrasts and "
        "uses exact two-sided rank tests, which cannot reach 0.05 at five runs per strategy (Section~\\ref{si:stats}). "
        "No contrast is significant under every procedure.}"
        "\\label{tab:procedures}\n"
        "\\begin{tabular}{llrrrrrrr}\\toprule\n"
        "Strategy & Compared with & Difference & GH interval & $p_{\\rm prim}$ & $p_{\\rm GH}$ & $p_{\\rm Tukey}$ & "
        "$p_{\\rm WY}$ & $p_{\\rm Holm9}$\\\\\\midrule\n"
        + body +
        "\\bottomrule\\end{tabular}\n\\end{table}\n\\endgroup\n"
    )


def prediction_table():
    """Logged surrogate error, as recorded along each search's own trajectory."""
    arms = {r["arm"]: r for r in read("arm_summary.csv")}
    body = "".join(
        "{} & ${:.3f} \\pm {:.3f}$ & ${:+.3f} \\pm {:.3f}$ & {:.1f}\\\\\n".format(
            LABEL[a],
            float(arms[a]["logged_mae_mean"]), float(arms[a]["logged_mae_std"]),
            float(arms[a]["logged_bias_mean"]), float(arms[a]["logged_bias_std"]),
            100 * float(arms[a]["logged_above_cap_fraction_mean"]),
        )
        for a in ORDER
    )
    return (
        "\\begin{table}[htbp]\\centering\\small\n"
        "\\caption{Error of the logged surrogate prediction relative to the evaluated OTC of the same "
        "record, in OTC percentage points, as mean $\\pm$ sample standard deviation over five runs. "
        "The last column is the percentage of logged predictive means exceeding the selected "
        "structure's ceiling (Section~\\ref{si:prediction}).}"
        "\\label{tab:prediction}\n"
        "\\begin{tabular}{lrrr}\\toprule\n"
        "Strategy & Mean absolute error & Bias & Above ceiling (\\%)\\\\\\midrule\n"
        + body +
        "\\bottomrule\\end{tabular}\n\\end{table}\n"
    )


def blocks_table():
    """Attainable ceiling and its utilization, by five-cycle block."""
    by_arm = defaultdict(dict)
    for r in read("reanalysis_ceiling_by_block.csv"):
        by_arm[r["arm"]][r["block"]] = r
    blocks = ["1-5", "6-10", "11-15", "16-20"]
    body = ""
    for a in ORDER:
        cells = []
        for b in blocks:
            r = by_arm[a][b]
            cells.append("{:.2f} / {:.3f}".format(
                float(r["mean_cap"]), float(r["mean_utilization"])))
        body += LABEL[a] + " & " + " & ".join(cells) + "\\\\\n"
    return (
        "\\begin{table}[htbp]\\centering\\small\n"
        "\\caption{Mean attainable ceiling $C(x)$ in wt.\\%, and the mean fraction of that ceiling "
        "attained, for the new evaluations of each five-cycle block, pooled over the five runs of each "
        "strategy. Entries are ceiling / fraction.}\\label{tab:blocks}\n"
        "\\begin{tabular}{lrrrr}\\toprule\n"
        "Strategy & Cycles 1--5 & Cycles 6--10 & Cycles 11--15 & Cycles 16--20\\\\\\midrule\n"
        + body +
        "\\bottomrule\\end{tabular}\n\\end{table}\n"
    )


def main():
    for name, text in [("procedure_comparison.tex", procedures_table()),
                       ("prediction_diagnostics.tex", prediction_table()),
                       ("ceiling_blocks.tex", blocks_table())]:
        (OUT / name).write_text(text)
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()
