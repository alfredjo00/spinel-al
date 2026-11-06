#!/usr/bin/env bash
# Recompute every number, table and figure of the article from the run databases.
#
#   analysis/run_all.sh [--with-calibration]
#
# Needs the Zenodo data linked into oxygen-carrier-discovery/heoc/ by
# analysis/scripts/restore_original_layout.py, and Python 3.10 or later with numpy, pandas,
# scipy, matplotlib and ase (max_otc_ceiling.py also needs the dependencies of metal, for
# example the run container). Outputs go to analysis/derived_data, analysis/tables and
# analysis/figures. --with-calibration also runs games_howell_calibration.py (about 1.5 h).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/scripts"
PY=${PYTHON:-python3}

run() { echo "== $1"; "$PY" "$@"; }

# Run-level data from the databases and run records
run analyze_runs.py
run audit_selections.py
run audit_mc_selected_scores.py
run run_level_statistics.py

# Statistics and audits
run planned_contrasts.py
run pairwise_tests_and_charge_scenarios.py
if [ "${1:-}" = --with-calibration ]; then run games_howell_calibration.py; fi
run audit_selected_score_offsets.py
run audit_air_reactor_margin.py
run audit_reactor_residual.py
run audit_quadratic_fit_noise.py
run analyze_random_baseline.py
run analyze_ceiling_baseline.py
run extract_run_timings.py
if "$PY" -c 'import icet, pymatgen' 2>/dev/null; then
    echo "== max_otc_ceiling.py"
    PYTHONWARNINGS=ignore "$PY" max_otc_ceiling.py > ../derived_data/max_otc_ceiling.txt
else
    echo "== max_otc_ceiling.py skipped: it needs the dependencies of metal (see container/)"
fi

# The held-out test (active-learning/src/holdout_censoring.py) trains every surrogate and
# takes hours on a GPU, so its results are taken from the Zenodo outputs unless recomputed.
holdout=../derived_data/holdout_test
if [ ! -f "$holdout/holdout_censoring.csv" ]; then
    if [ -d ../../data/analysis_outputs/derived_data/holdout_test ]; then
        echo "== held-out test results copied from data/analysis_outputs"
        mkdir -p "$holdout"
        cp ../../data/analysis_outputs/derived_data/holdout_test/*.csv "$holdout/"
    else
        echo "Held-out test results missing: unzip analysis_outputs.zip into data/, or run" >&2
        echo "active-learning/src/holdout_censoring.py with OUT and PRED_OUT in $holdout" >&2
        exit 1
    fi
fi

# Tables and figures (gibbs_pipeline.tex was written by hand)
cp ../gibbs_pipeline.tex ../tables/
run make_tables.py
run make_si_tables.py
run make_si_ceiling_holdout_tables.py
run make_figures.py
run make_si_figures.py
