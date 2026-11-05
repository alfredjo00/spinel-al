#!/usr/bin/env python3
"""Random reference: the 500 additional random compositions and subsets of 200 drawn from them.

Checks that the 1,000 compositions shared with the initial dataset agree with the run
databases and that the stored OTC follows from the Gibbs energies, then draws subsets of
200 without replacement (--draws, --seed). The subsets overlap between draws, so their
range describes subsets of these 500 compositions, not independent random searches.

Inputs: spinel_Li_random/training.db and spinel_rev_de_s1/training.db.
Outputs: random_baseline_additional.csv and random_baseline_summary.json.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from ase.data import chemical_symbols

from paths import DATA, HEOC


def records(path):
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as con:
        rows = con.execute("SELECT id,numbers,key_value_pairs FROM systems ORDER BY id").fetchall()
    result = {}
    for ident, blob, raw in rows:
        counts = Counter(chemical_symbols[z] for z in np.frombuffer(blob, dtype=np.int32))
        oxygen_count = counts.pop('O', 0)
        formula = ''.join(e + (str(n) if n != 1 else '') for e, n in sorted(counts.items())) + 'O48'
        assert formula not in result, f'Duplicate composition: {formula}'
        result[formula] = dict(id=ident, oxygen_count=oxygen_count, **json.loads(raw))
    return result


def minimum(values):
    a, b, c = np.polynomial.polynomial.polyfit(np.arange(8), values, 2)[::-1]
    if abs(a) < 1e-10:
        return 0.0 if b > 0 else 7.0
    if a > 0:
        return float(np.clip(-b / (2 * a), 0, 7))
    return 0.0 if c <= 49 * a + 7 * b + c else 7.0


def summarize(values):
    return dict(mean=float(np.mean(values)), q025=float(np.quantile(values, .025)),
                q975=float(np.quantile(values, .975)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HEOC)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260908)
    args = parser.parse_args()
    source = args.root / 'spinel_Li_random/training.db'
    reference = args.root / 'spinel_rev_de_s1/training.db'
    random_records = records(source)
    revision = records(reference)
    initial = {f: r for f, r in revision.items() if r['cycle'] == 0}
    assert len(initial) == 1000 and initial.keys() <= random_records.keys()
    extra = {f: r for f, r in random_records.items() if f not in initial}
    assert len(extra) == 500
    fields = [f'gibbs_{reactor}_{v}' for reactor in ('AR', 'FR') for v in range(8)]
    fields += ['mass_', 'otc_quadratic_fit', 'vacancy_diff_quadratic_fit']
    shared_errors = {k: max(abs(random_records[f][k] - initial[f][k]) for f in initial) for k in fields}
    assert max(shared_errors.values()) < 1e-10
    errors = []
    rows = []
    for formula, r in extra.items():
        assert r['oxygen_count'] == 48
        release = minimum([r[f'gibbs_FR_{v}'] for v in range(8)]) - minimum([r[f'gibbs_AR_{v}'] for v in range(8)])
        otc = r['otc_quadratic_fit'] * 100
        errors.append(abs(100 * 15.999 * release / r['mass_'] - otc))
        rows.append(dict(source_id=r['id'], formula=formula, cycle=r['cycle'], otc=otc,
                         vacancy_diff=r['vacancy_diff_quadratic_fit'], mass=r['mass_']))
    assert max(errors) < 1e-8
    values = np.array([r['otc'] for r in rows])
    initial_max = max(r['otc_quadratic_fit'] * 100 for r in initial.values())
    rng = np.random.default_rng(args.seed)
    means, maxima = [], []
    for _ in range(args.draws):
        sample = rng.choice(values, size=200, replace=False)
        means.append(float(sample.mean()))
        maxima.append(float(sample.max()))
    result = dict(source=str(source.relative_to(args.root)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  reference=str(reference.relative_to(args.root)), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                  n_shared=len(initial), n_additional=len(extra), shared_max_field_errors=shared_errors,
                  max_otc_refit_error_percentage_points=max(errors),
                  additional_mean_otc=float(values.mean()), additional_max_otc=float(values.max()),
                  additional_count_ge6=int((values >= 6).sum()), initial_max_otc=initial_max,
                  budget=200, draws=args.draws, seed=args.seed,
                  mean_acquired_otc=summarize(means), maximum_new_otc=summarize(maxima),
                  maximum_including_initial=summarize(np.maximum(maxima, initial_max)),
                  interpretation='Central 95% subset-resampling ranges conditional on the observed 500 compositions, not confidence intervals from independent search runs.',
                  provenance='The 500 compositions were generated with the 1,000 of the initial dataset. The shared records and the OTC extraction are checked here.')
    out = DATA
    with (out / 'random_baseline_additional.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'random_baseline_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
