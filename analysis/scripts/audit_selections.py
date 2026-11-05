#!/usr/bin/env python3
"""Check the selections of each run and the fuel-reactor reconstruction of the vacancy-input target.

Run after analyze_runs.py. For every cycle of every run, each candidate in
cycle<N>/sample.db is matched to its training record by exact composition. For the
strategies other than MC, the selected acquisition scores are converted back to OTC
and compared with the ceiling C(x) (MC is handled by audit_mc_selected_scores.py).
The fuel-reactor energies reconstructed as G_AR(v) + OFFSET*v are compared with the
evaluated ones, and the stored OTC is recomputed from the Gibbs energies.

Outputs: selection_audit.csv, selected_scores_non_mc.csv,
selected_score_summary_non_mc.csv and reactor_offset_audit.csv.
"""
import json
import sqlite3

import numpy as np
import pandas as pd
from ase.data import atomic_numbers, chemical_symbols

from analyze_runs import ARMS, O_MASS
from paths import DATA, HEOC

OFFSET = -1.8836231615976065


def load(path):
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
        records = con.execute('SELECT id,numbers,key_value_pairs FROM systems ORDER BY id').fetchall()
    out = []
    for ident, numbers, kv in records:
        counts = np.bincount(np.frombuffer(numbers, dtype=np.int32), minlength=119)
        out.append((ident, counts, json.loads(kv)))
    return out


def formula(counts):
    return ''.join(chemical_symbols[z] + str(n) for z,n in enumerate(counts) if n)


def minima(g):
    # Minima of the least-squares quadratic on [0, 7], as for the stored OTC.
    v = np.arange(8, dtype=float)
    a,b,c = (np.linalg.pinv(np.vander(v,3)) @ np.asarray(g).T)
    endpoint = np.where(49*a+7*b > 0, 0., 7.)
    stationary = np.clip(-b / np.where(abs(a)>1e-10,2*a,1),0,7)
    return np.where(abs(a)<1e-10,np.where(b>0,0.,7.),np.where(a<0,endpoint,stationary))


def main():
    opportunities, scores, offsets = [], [], []
    for arm in ARMS:
        for seed in range(1,6):
            d=HEOC/f'spinel_rev_{arm}_s{seed}'
            rows=load(d/'training.db')
            exact={tuple(c):(ident,kv) for ident,c,kv in rows}
            cfg_path=next(d.glob('*/config.json'))
            cfg=json.loads(cfg_path.read_text())
            pc=pd.read_csv(cfg_path.parent/'prediction_comparison.csv')
            pc=pc[pc.property=='otc_quadratic_fit'].set_index(['cycle','sample_rank'])
            sp=pd.read_csv(cfg_path.parent/'sample_predictions.csv').set_index(['cycle','sample_rank'])
            subsets=[('acquired',[r for r in rows if r[2]['cycle']>0])]
            if arm=='de' and seed==1:
                subsets.append(('initial',[r for r in rows if r[2]['cycle']==0]))
            for kind,subset in subsets:
                g_ar=np.array([[r[2][f'gibbs_AR_{v}'] for v in range(8)] for r in subset])
                g_fr=np.array([[r[2][f'gibbs_FR_{v}'] for v in range(8)] for r in subset])
                mass=np.array([r[2]['mass_'] for r in subset])
                actual=np.array([r[2]['otc_quadratic_fit'] for r in subset])
                recomputed=(minima(g_fr)-minima(g_ar))*O_MASS/mass
                assert np.allclose(recomputed,actual,atol=1e-10)
                approximate=g_ar+OFFSET*np.arange(8)
                otc_approx=(minima(approximate)-minima(g_ar))*O_MASS/mass
                residual=approximate-g_fr
                offsets.append(dict(arm=arm,replicate=seed,subset=kind,n=len(subset),
                                    fr_energy_rmse_eV=np.sqrt(np.mean(residual**2)),
                                    fr_energy_max_abs_eV=abs(residual).max(),
                                    otc_offset_mae_pp=abs(otc_approx-actual).mean()*100,
                                    otc_offset_bias_pp=(otc_approx-actual).mean()*100,
                                    otc_offset_max_abs_pp=abs(otc_approx-actual).max()*100))
            for cycle in range(1,21):
                sample_rows=load(d/f'cycle{cycle}/sample.db')
                selected=pd.read_csv(d/f'cycle{cycle}/sample.csv',sep='\t')
                assert len(selected)==10
                for ident,counts,kv in sample_rows:
                    match=exact.get(tuple(counts))
                    status='new_evaluation' if match and match[1]['cycle']==cycle else 'previously_evaluated' if match else 'absent_from_training'
                    opportunities.append(dict(arm=arm,replicate=seed,cycle=cycle,sample_db_id=ident,
                                              formula=formula(counts),status=status,n_cations=int((counts>0).sum()-1),
                                              evaluated_cycle=match[1]['cycle'] if match else np.nan,
                                              otc=match[1]['otc_quadratic_fit']*100 if match else np.nan))
                for i in range(10-len(sample_rows)):
                    opportunities.append(dict(arm=arm,replicate=seed,cycle=cycle,status='no_distinct_sample_db_row'))
                # For non-MC arms the CSV compositions are already exact integer
                # counts and row mapping can be checked without tolerances.
                if arm.startswith('mc'):
                    continue
                training=np.array([r[2]['otc_quadratic_fit'] for r in rows if r[2]['cycle']<cycle])
                for rank,sample in selected.iterrows():
                    counts=np.zeros(119,dtype=int)
                    for e in cfg['elements']:
                        counts[atomic_numbers[e]]=round(sample[e]*84)
                        assert abs(sample[e]*84-counts[atomic_numbers[e]])<1e-4
                    match=exact[tuple(counts)]
                    assert match[1]['cycle']==cycle
                    cap=7*O_MASS/match[1]['mass_']
                    score=sample.otc
                    logged_raw_mean=sp.loc[(cycle,rank),'otc_pred']
                    if not cfg['predict_landscape']:
                        scale=training.std() if arm.startswith('de') else np.ptp(training)
                        shift=training.mean() if arm.startswith('de') else training.min()
                        score=score*scale+shift
                        assert np.isclose(logged_raw_mean*scale+shift,pc.loc[(cycle,rank),'predicted_mean'],atol=1e-7)
                    scores.append(dict(arm=arm,replicate=seed,cycle=cycle,rank=rank,
                                       score=score*100,cap=cap*100,
                                       above_cap=score>cap+1e-7,
                                       at_cap=abs(score-cap)<1e-7,
                                       actual=match[1]['otc_quadratic_fit']*100,
                                       logged_prediction=pc.loc[(cycle,rank),'predicted_mean']*100))
    pd.DataFrame(opportunities).to_csv(DATA/'selection_audit.csv',index=False)
    sd=pd.DataFrame(scores)
    sd.to_csv(DATA/'selected_scores_non_mc.csv',index=False)
    sd.groupby(['arm','replicate'])[['above_cap','at_cap']].mean().to_csv(DATA/'selected_score_summary_non_mc.csv')
    od=pd.DataFrame(offsets)
    od.to_csv(DATA/'reactor_offset_audit.csv',index=False)
    print(pd.DataFrame(opportunities).groupby(['arm','status']).size().to_string())
    print('\nScores:',sd.groupby('arm')[['above_cap','at_cap']].mean().round(3).to_string())
    print('\nOffset approximation:',od.groupby(['subset','arm'])[['fr_energy_rmse_eV','otc_offset_mae_pp','otc_offset_bias_pp']].mean().round(5).to_string())


if __name__=='__main__':
    main()
