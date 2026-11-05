"""Wall-clock stage timings per cycle from the logs of the 50 runs (run_timings.csv).

Parses <timestamp>/active_learning.log in each run directory and reports, per run, the
median duration of four stages:

  fit_s          "Loading data from" -> "Train WAE"         data loading and DE/GP surrogate fit
  sample_s       "Train WAE" -> "Screened N candidates"     WAE training, candidate sampling and scoring
  eval_s         "Performing phonon" -> "Phonon ... complete"   CHGNet/Phonopy evaluation of the cycle
  local_chunk_s  "Running phonon chunk" -> "Local phonon chunk"  the evaluation chunk run on the main GPU

For MC, fit_s covers data loading only, because the MC surrogate is trained together with
the WAE and its training falls in sample_s. The evaluation of each cycle was split into
five chunks, one run locally and four as separate cluster jobs on one GPU each, while
every other stage used one GPU. Evaluation GPU time is estimated as 5 x local_chunk_s.
Each stage start is paired with the first following stage end, so restarts that leave an
unmatched start are dropped.

These are wall-clock times on a shared GPU cluster, not isolated benchmarks. eval_s
includes queueing of the four cluster jobs, and the strategies were run on different dates.
"""
import csv
import glob
import re
import statistics as st
from datetime import datetime

from paths import DATA, HEOC

ARMS = [("de", "DE scalar"), ("de_clip", "DE clipped"), ("de_land", "DE Gibbs vector"),
        ("dde", "DE vacancy input"), ("gp", "GP scalar"), ("gp_clip", "GP clipped"),
        ("mogp", "GP Gibbs vector"), ("dgp", "GP vacancy input"), ("mc", "MC scalar"),
        ("mc_clip", "MC clipped")]
TIMESTAMP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) - INFO - (.*)$")
MARKERS = (("load", "Loading data from"), ("wae", "Train WAE"), ("scr", "Screened "),
           ("ps", "Performing phonon"), ("pe", "Phonon calculations complete"),
           ("ls", "Running phonon chunk"), ("le", "Local phonon chunk"))


def events(path):
    for line in open(path, errors="replace"):
        m = TIMESTAMP.match(line.rstrip("\n"))
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000
        for key, prefix in MARKERS:
            if m.group(3).startswith(prefix):
                yield key, ts
                break


def spans(events, start, end):
    """Durations from each start to the first following end."""
    out, opened = [], None
    for key, ts in events:
        if key == start:
            opened = ts
        elif key == end and opened is not None:
            out.append(ts - opened)
            opened = None
    return out


def median(values):
    return st.median(values) if values else None


rows = []
for arm, label in ARMS:
    for seed in range(1, 6):
        logs = sorted(glob.glob(f"{HEOC}/spinel_rev_{arm}_s{seed}/2*/active_learning.log"))
        if not logs:
            rows.append([label, seed, None, None, None, None, "missing"])
            continue
        e = list(events(logs[-1]))
        fit, sample, evaluation = spans(e, "load", "wae"), spans(e, "wae", "scr"), spans(e, "ps", "pe")
        local = spans(e, "ls", "le")
        rows.append([label, seed, median(fit), median(sample), median(evaluation), median(local),
                     f"n_fit={len(fit)} n_eval={len(evaluation)}"])
with open(DATA / "run_timings.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["label", "replicate", "fit_s", "sample_s", "eval_s", "local_chunk_s", "note"])
    writer.writerows(rows)

print(f"{'strategy':18s} {'fit s/cycle':>18s} {'sample s':>12s} {'eval h/cycle':>14s} {'fit % of GPU time':>18s}")
for arm, label in ARMS:
    r = [x for x in rows if x[0] == label and x[2] is not None]
    if not r:
        print(f"{label:18s}  NO DATA")
        continue
    fit, sample = [x[2] for x in r], [x[3] for x in r]
    evaluation, local = [x[4] for x in r if x[4]], [x[5] for x in r if x[5]]
    gpu = st.median(fit) + st.median(sample) + 5 * st.median(local)
    print(f"{label:18s} {st.median(fit):7.0f} ({min(fit):.0f}-{max(fit):.0f}) {st.median(sample):11.0f} "
          f"{st.median(evaluation) / 3600:13.2f} {100 * st.median(fit) / gpu:17.2f}   n={len(r)}")
