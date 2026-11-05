#!/usr/bin/env python3
"""Selected acquisition scores of the MC strategies, converted to OTC and mapped to the evaluated structures.

Run after audit_selections.py. MC selects compositions as fractions, so each
selection is rounded to 84-atom counts as the composition code does and matched to its
saved structure exactly. The normalization of each cycle is recovered from the logged
predictions. Both the ceiling used during selection (from the separately rounded
composition, Supplementary Section S9) and the ceiling of the saved structure are kept,
and `cap` always means the latter.

Inputs: the MC run directories and selected_scores_non_mc.csv.
Outputs: selected_scores_mc.csv, mc_normalization_audit.csv, selected_scores_all.csv,
selected_score_summary_mc.csv, and mc_score_audit_manifest.json with the SHA-256 of
every input file.
"""
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
from ase.data import atomic_masses, atomic_numbers, chemical_symbols

from paths import DATA, HEOC

TOL = 1e-7  # mass fraction, as in audit_selections.py
MASSES = np.pad(atomic_masses, (0, 119-len(atomic_masses)))


def read_db(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        rows = con.execute('SELECT id,numbers,key_value_pairs FROM systems ORDER BY id').fetchall()
    return [(ident, np.bincount(np.frombuffer(numbers, dtype=np.int32), minlength=119),
             json.loads(kv)) for ident, numbers, kv in rows]


def realized_counts(sample, cfg):
    # The composition code (metal) minimizes squared count error subject to each sublattice total.
    # For floor/ceil choices its unique optimum assigns the remaining sites to
    # the largest fractional remainders. Check uniqueness and the saved DB.
    counts = np.zeros(119, dtype=int)
    counts[8] = 48
    for group, nsites in [('A_elements', 12), ('B_elements', 24)]:
        elements = cfg[group]
        target = sample[elements].to_numpy(dtype=float)*84
        base = np.floor(target).astype(int)
        remainder = int(nsites-base.sum())
        assert 0 <= remainder <= len(elements)
        fractions = target-base
        order = np.argsort(-fractions)
        if 0 < remainder < len(elements):
            assert fractions[order[remainder-1]]-fractions[order[remainder]] > 1e-10
        base[order[:remainder]] += 1
        counts[[atomic_numbers[e] for e in elements]] = base
    assert counts.sum() == 84
    return counts


def main():
    records, transforms, fingerprints = [], [], {}
    def register(path):
        name = path.relative_to(HEOC if path.is_relative_to(HEOC) else DATA)
        fingerprints[str(name)] = hashlib.sha256(path.read_bytes()).hexdigest()
    register(DATA/'selected_scores_non_mc.csv')
    for arm in ['mc', 'mc_clip']:
        for seed in range(1, 6):
            directory = HEOC/f'spinel_rev_{arm}_s{seed}'
            configs = list(directory.glob('*/config.json'))
            assert len(configs) == 1
            config_path = configs[0]
            cfg = json.loads(config_path.read_text())
            assert cfg['atoms_in_structure'] == 84 and np.isclose(cfg['tol'], 1/84)
            assert not cfg['predict_landscape']
            assert cfg['clip_otc_physical'] == (arm == 'mc_clip')
            sp_path, pc_path = [config_path.parent/f for f in ['sample_predictions.csv', 'prediction_comparison.csv']]
            for path in [config_path, sp_path, pc_path, directory/'training.db']:
                register(path)
            sp = pd.read_csv(sp_path)
            pc = pd.read_csv(pc_path).query("property == 'otc_quadratic_fit'")
            assert len(sp) == 200 and not sp.duplicated(['cycle', 'sample_rank']).any()
            pairs = sp.merge(pc, on=['cycle', 'sample_rank'], validate='one_to_one')
            training = read_db(directory/'training.db')
            training_by_counts = {tuple(c): (ident, kv) for ident, c, kv in training}
            assert len(training_by_counts) == len(training)
            elements = cfg['elements']
            zs = [atomic_numbers[e] for e in elements]
            for cycle in range(1, 21):
                p = pairs[pairs.cycle == cycle]
                assert len(p) >= 2 and (p.otc_unc > 0).all()
                # Uncertainty scaling recovers slope independently of means.
                scales = p.predicted_uncertainty/p.otc_unc
                scale = float(np.median(scales))
                shifts = p.predicted_mean-scale*p.otc_pred
                shift = float(np.median(shifts))
                assert scale > 0 and np.max(abs(scales-scale)) < 1e-10
                assert np.max(abs(shifts-shift)) < 1e-12
                fit = np.polyfit(p.otc_pred, p.predicted_mean, 1)
                assert np.allclose(fit, [scale, shift], rtol=0, atol=1e-10)
                # Independent full-training extrema check: random row deletion
                # did not change the extrema in these recorded MC cycles.
                values = np.array([kv['otc_quadratic_fit'] for _, _, kv in training if kv['cycle'] < cycle])
                assert np.isclose(shift, values.min(), rtol=0, atol=1e-10)
                assert np.isclose(scale, np.ptp(values), rtol=0, atol=1e-10)
                transforms.append(dict(arm=arm, replicate=seed, cycle=cycle,
                    n_log_pairs=len(p), scale=scale, shift=shift,
                    max_mean_residual=float(np.max(abs(shifts-shift))),
                    training_count=len(values), training_count_odd=bool(len(values)%2)))
                csv_path, db_path = [directory/f'cycle{cycle}'/f for f in ['sample.csv', 'sample.db']]
                register(csv_path); register(db_path)
                selected = pd.read_csv(csv_path, sep='\t')
                saved_rows = read_db(db_path)
                saved = {tuple(c): (ident, kv) for ident, c, kv in saved_rows}
                assert len(selected) == 10 and len(saved) == len(saved_rows)
                seen = set()
                for rank, sample in selected.iterrows():
                    counts = realized_counts(sample, cfg)
                    key = tuple(counts)
                    assert key in saved
                    seen.add(key)
                    # Replay export_top_unique's independent component rounding.
                    rounded = (sample[elements]/cfg['tol']).round().to_numpy(dtype=float)*cfg['tol']*84
                    selection_cap = 7*MASSES[8]/(rounded @ MASSES[zs])
                    cap = 7*MASSES[8]/(counts @ MASSES)
                    score = sample.otc*scale+shift
                    prediction = sp[(sp.cycle == cycle)&(sp.sample_rank == rank)].iloc[0]
                    match = training_by_counts.get(key)
                    if match:
                        assert np.isclose(counts @ MASSES, match[1]['mass_'], rtol=0, atol=1e-7)
                    actual = match[1]['otc_quadratic_fit']*100 if match else np.nan
                    status = ('new_evaluation' if match and match[1]['cycle'] == cycle else
                              'previously_evaluated' if match else 'absent_from_training')
                    log_pair = p[p.sample_rank == rank]
                    logged_actual = float(log_pair.actual_value.iloc[0])*100 if len(log_pair) else np.nan
                    records.append(dict(arm=arm, replicate=seed, cycle=cycle, rank=rank,
                        score=score*100, cap=cap*100, above_cap=bool(score>cap+TOL),
                        at_cap=bool(abs(score-cap)<TOL), actual=actual,
                        logged_prediction=(prediction.otc_pred*scale+shift)*100,
                        logged_uncertainty=prediction.otc_unc*scale*100,
                        selection_cap=selection_cap*100,
                        above_selection_cap=bool(score>selection_cap+TOL),
                        at_selection_cap=bool(abs(score-selection_cap)<TOL),
                        selection_cation_count=int(round(rounded[:-1].sum())),
                        rounding_differs=bool(not np.array_equal(np.rint(rounded).astype(int), counts[zs])),
                        cap_relative_error_pct=100*(selection_cap/cap-1),
                        sample_db_id=saved[key][0], evaluation_status=status,
                        formula=''.join(chemical_symbols[z]+str(n) for z,n in enumerate(counts) if n),
                        logged_actual=logged_actual,
                        logged_actual_matches_exact=(bool(abs(logged_actual-actual)<1e-7)
                                                     if np.isfinite(logged_actual) and match else None)))
                assert seen == set(saved)
    scores = pd.DataFrame(records)
    assert len(scores) == 2000
    assert not scores.query("arm == 'mc_clip'").above_selection_cap.any()
    scores.to_csv(DATA/'selected_scores_mc.csv', index=False)
    pd.DataFrame(transforms).to_csv(DATA/'mc_normalization_audit.csv', index=False)
    combined = pd.concat([pd.read_csv(DATA/'selected_scores_non_mc.csv'), scores], ignore_index=True)
    assert len(combined) == 10000 and (combined.groupby(['arm','replicate']).size() == 200).all()
    combined.to_csv(DATA/'selected_scores_all.csv', index=False)
    summary = scores.groupby(['arm','replicate'])[
        ['above_cap','at_cap','above_selection_cap','at_selection_cap','rounding_differs']].mean()
    summary.to_csv(DATA/'selected_score_summary_mc.csv')
    (DATA/'mc_score_audit_manifest.json').write_text(json.dumps(dict(
        inputs_sha256=fingerprints, selected_scores=2000,
        normalization_cycles=200, exact_structure_mappings=2000,
        maximum_mean_transform_residual=max(t['max_mean_residual'] for t in transforms),
        tolerance_mass_fraction=TOL), indent=2)+'\n')
    print(summary.groupby('arm').mean().to_string())
    print(scores.groupby(['arm','evaluation_status']).size().to_string())
    print('Logged actual agreement:', scores.groupby('arm').logged_actual_matches_exact.agg(['count','mean']).to_string())


if __name__ == '__main__':
    main()
