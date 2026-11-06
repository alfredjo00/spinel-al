"""Held-out test of the surrogates (Supplementary Information of the article).

Every surrogate/target combination of the paper is trained exactly as at cycle
1 of the spinel_rev runs (cycle-0 data of spinel_base_rev, production 85/15
split with seed 42, production hyperparameters and normalisation) and asked for
the OTC of two test sets. No oracle calls; all databases are only read.

  random  the 500 random-reference compositions in spinel_Li_random that are
          not among the shared 1,000 (distribution of the initial data, mostly
          censored at zero)
  acq     every unique composition acquired by the 10 paper arms x 5 seeds and
          not among the 1,000 (the ceiling-heavy region the AL actually ranks
          in). A composition acquired by several runs gets the mean of their
          oracle OTCs.
  acq_mc  acq restricted to compositions picked by an MC arm (mc, mc_clip). MC is
          not among the compared models, so no model's own family chose these:
          a common subset free of the selection bias of acq. Headline set with
          random; acq supports it, acq_loo is for the SI.
  acq_loo acq without the picks of the model's own family (gp/gp_cap: gp and
          gp_clip arms; de/de_cap: de, de_clip; others: their own arm). Each
          model sees a different subset, so compare it with acq, not across
          models. Robustness check for the selection bias of acq.

OTC is a mass fraction in the databases and in the models; everything reported
here is converted to wt% (x100). The truth is `otc_quadratic_fit`, as in the
runs (--quadratic-fit-otc). The ceiling C(x) = 7 m_O / m_cell(x) is the fully
oxidised, vacancy-free 84-atom cell (al_functions._otc_physical_upper_bound).

Models
  gp, de              scalar OTC (GP minmax, DE zscore)
  <scalar>_cap        the same predictions min'd with C(x) (clipping is
                      acquisition-only in the runs; this is its offline analogue)
  mogp, de_land       16-output Gibbs landscape (grouped zscore)
  dgp, dde            AR curve with the vacancy level as input, FR from the
                      analytic offset (grouped zscore, production adapters)
MC dropout is left out: production MC never predicts on given compositions, it
decodes latent samples, so there is no faithful offline analogue.
Landscape OTC is derived as in al_functions._predict_otc_from_landscape
(quadratic-fit branch): per member, denormalise the
curve, fit quadratics to AR and FR, dv = v_FR - v_AR, OTC = m_O dv / m_cell,
then average over members. Reimplemented here rather than called because with
log_path=None that function falls through to its MC branch.

Metrics per model x seed x test set: MAE/RMSE/Spearman over all rows and the
groups zero / mid / ceil (true OTC) and nonzero; top-k for each k in TOPK:
mean true OTC of the k highest predictions, and hit@k = fraction of them whose
true OTC is at least the true k-th largest (ties at the ceiling count as hits).

Outputs: OUT (one row per model x seed x set) and PRED_OUT (one row per test
composition with truth, ceiling, source arms and every model x seed prediction,
for slicing by source arm later).

Env knobs: SEEDS (1,2,3,4,5), MODELS (all), OUT (holdout_censoring.csv),
PRED_OUT (holdout_censoring_pred.csv), TOPK (10,50), CEIL_RTOL (1e-3),
ZERO_TOL (1e-6, mass fraction).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
os.environ.setdefault('METAL_CHEMSYS_CACHE', os.path.join(_root, '.cache', 'metal', 'chemsys'))
sys.path.append(os.path.join(_root, 'metal'))

from al_config import Config, parse_args
from data_manager import DataManager
from al_functions import (GPRegressor, train_GP_new, create_df, _otc_physical_upper_bound,
                          _calculate_composition_masses, _fit_quadratic_and_find_minimum)
from ANN_functions_new import same_seeds
from ensemble_mlp import DeepEnsemble, train_deep_ensemble
from mogp.ensemble_adapter import MOGPEnsembleAdapter, split_constant_columns
from mogp.models import ICMGPRegressor, fit_gp as fit_mogp
from mogp.delta_gp import (DeltaGPRegressor, DeltaGPEnsembleAdapter, DeltaEnsembleAdapter,
                           build_training_matrix, train_delta_gp, MU_O_FR_MINUS_AR)
from ase.data import atomic_masses, atomic_numbers

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('censoring')

SEEDS = [int(s) for s in os.environ.get('SEEDS', '1,2,3,4,5').split(',')]
ALL = ['gp', 'de', 'mogp', 'de_land', 'dgp', 'dde']
MODELS = os.environ.get('MODELS', ','.join(ALL)).split(',')
OUT = os.environ.get('OUT', 'holdout_censoring.csv')
PRED_OUT = os.environ.get('PRED_OUT', 'holdout_censoring_pred.csv')
TOPK = [int(k) for k in os.environ.get('TOPK', '10,50').split(',')]
CEIL_RTOL = float(os.environ.get('CEIL_RTOL', '1e-3'))
ZERO_TOL = float(os.environ.get('ZERO_TOL', '1e-6'))

# Paper arms whose acquisitions form the acq set, and the arms each model's
# own family picked with (excluded in acq_loo)
ARMS = ['gp', 'gp_clip', 'mogp', 'dgp', 'de', 'de_clip', 'dde', 'de_land', 'mc', 'mc_clip']
RUN_SEEDS = range(1, 6)
FAMILY = {'gp': {'gp', 'gp_clip'}, 'de': {'de', 'de_clip'}, 'mogp': {'mogp'},
          'de_land': {'de_land'}, 'dgp': {'dgp'}, 'dde': {'dde'}}

# Shared flags of the spinel_rev jobscripts, plus the per-arm surrogate flags
# (dgp: --delta-gp-iters 1000 --delta-gp-patience 100; dde: --delta-de-epochs
# 200 --delta-de-lr 1e-4; mogp: --mogp-rank 1). They only affect their own arm.
BASE = ['--method', 'spinel_base_rev', '--seed', '1', '--cycles', '2',
        '--ensemble-size', '16', '--n-samples-total', '25000', '--max-element-count', '8',
        '--use-validation', '--val-split', '0.15', '--val-split-mode', 'random',
        '--round-compositions', '--quadratic-fit-otc', '--bootstrap-fraction', 'none']
LAND = ['--predict-landscape', '--quadratic-fit-landscape-otc', '--mogp-rank', '1',
        '--delta-gp-iters', '1000', '--delta-gp-patience', '100',
        '--delta-de-epochs', '200', '--delta-de-lr', '1e-4']


def make(model, landscape, mode):
    cfg = Config(parse_args(BASE + ['--model', model] + (LAND if landscape else [])))
    cfg.set_method_list()
    # constants used below (single-site runs: one composition column per element)
    cfg.column_scaling = cfg.scaling_dictionary
    cfg.mass_O = float(atomic_masses[atomic_numbers['O']])
    cfg.n_landscape_dims = 2 * cfg.n_vacancies_max
    cfg.vacancy_points = np.arange(cfg.n_vacancies_max, dtype=np.float64)
    cfg.mu_o_fr_minus_ar = MU_O_FR_MINUS_AR
    dm = DataManager(cfg, logger)
    dm.load_data(cfg.method_list, cfg.db_name, cfg.elements)
    dm.normalize(mode=mode)
    dm.split_data(val_split=cfg.val_split, mode=cfg.val_split_mode)  # seed 42, as production
    return cfg, dm


# One DataManager per normalisation the arms use
cfg_m, dm_m = make('gp', False, 'minmax')       # gp
cfg_z, dm_z = make('ensemble', False, 'zscore')  # de
cfg_l, dm_l = make('mogp', True, 'zscore')       # mogp, de_land, dgp, dde
for dm in (dm_m, dm_z, dm_l):
    assert len(dm.raw_x) == 1000, f'expected 1000 training rows, got {len(dm.raw_x)}'
assert np.allclose(dm_m.comps_x, dm_l.comps_x), 'scalar and landscape loads differ'
dev = cfg_l.device

# ------------------------------------------------------------ test sets
els = cfg_l.elements                            # element symbols + O
cations = els[:-1]


def load(db):
    return create_df(cfg_l.method_list, db, cfg_l.elements,
                     use_quadratic_fit_otc=True)


def keys(df):
    return [tuple(r) for r in np.rint(df[cations].values * cfg_l.atoms_in_structure).astype(int)]


train_keys = set(keys(dm_l.raw_df))


def unseen(df):
    return df.loc[[k not in train_keys for k in keys(df)]].reset_index(drop=True)


rand = unseen(load(os.path.join(cfg_l.heoc_path, 'spinel_Li_random', 'training.db')))
print(f'random test rows (not in the 1000): {len(rand)}')
assert len(rand) == 500, f'expected 500 random test rows, got {len(rand)}'
rand['arms'] = [frozenset()] * len(rand)

parts = []
for arm in ARMS:
    for s in RUN_SEEDS:
        df = unseen(load(os.path.join(cfg_l.heoc_path, f'spinel_rev_{arm}_s{s}', 'training.db')))
        parts.append(df.assign(arm=arm))
acq_all = pd.concat(parts, ignore_index=True)
acq_all['key'] = ['-'.join(map(str, k)) for k in keys(acq_all)]   # str: tuples would become a MultiIndex
by_key = acq_all.groupby('key', sort=False)
acq = acq_all.drop_duplicates('key').set_index('key')
acq['otc'] = by_key['otc'].mean()
acq['arms'] = by_key['arm'].agg(frozenset)
acq = acq.reset_index(drop=True)
print(f'acquisitions: {len(acq_all)} rows, {len(acq)} unique compositions; '
      f'max spread of repeated OTC {by_key["otc"].agg(np.ptp).max() * 100:.3f} wt%')

te = pd.concat([rand[els + ['otc', 'arms']].assign(set='random'),
                acq[els + ['otc', 'arms']].assign(set='acq')], ignore_index=True)

# Model inputs: sublattice fractions, as DataManager.load_data scales them
te_x_np = np.stack([te[c].values / cfg_l.column_scaling[c] for c in cations], axis=1)
te_x = torch.tensor(te_x_np, dtype=torch.float32, device=dev)
otc_true = te['otc'].values                                     # mass fraction
ceil = np.asarray(_otc_physical_upper_bound(te, els, cfg_l.atoms_in_structure, cfg_l))
m_cell = _calculate_composition_masses(te_x, els, len(cfg_l.A_elements), len(cfg_l.B_elements),
                                       cfg_l.column_scaling, dev, config=cfg_l).cpu().numpy()
assert np.allclose(ceil, cfg_l.mass_O * (cfg_l.n_vacancies_max - 1) / m_cell)


def ens_args(dm, ycols=None):
    """train/val tensors as production stacks them"""
    tr_x, tr_y, va_x, va_y, _, _ = dm.get_ensemble_data()
    if ycols is None:
        return tr_x, tr_y['y0'], va_x, va_y['y0']
    return (tr_x, torch.stack([tr_y[f'y{i}'] for i in ycols], 1),
            va_x, torch.stack([va_y[f'y{i}'] for i in ycols], 1))


def train_de(cfg, x, y, vx, vy, out_dim, epochs, lr, noise_std):
    de = DeepEnsemble(n_members=cfg.ensemble_size, input_dim=x.shape[1],
                      hidden_dims=cfg.ensemble_hidden_dims, dropout_rate=cfg.ensemble_dropout_rate,
                      device=dev, output_dim=out_dim, prior_variance=cfg.ensemble_prior_variance)
    train_deep_ensemble(de, x, y, val_x=vx, val_y=vy, bootstrap_fraction=cfg.bootstrap_fraction,
                        epochs=epochs, batch_size=cfg.ensemble_batch_size, lr=lr,
                        weight_decay=cfg.ensemble_weight_decay, base_seed=cfg.default_seed,
                        config=cfg, patience=cfg.early_stopping_patience,
                        prior_reg=cfg.ensemble_prior_reg, noise_std=noise_std, logger=logger)
    return de


def landscape_otc(surrogate):
    """Landscape OTC with member means, as ensemble_otc_quadratic_fit_member_mean of the runs."""
    with torch.no_grad():
        im = surrogate.predict(te_x)['individual_means'].detach().cpu().numpy()   # [N, 16, M]
    n_vac = cfg_l.n_vacancies_max
    inv = dm_l.get_inverse_transform_fn(group='gibbs')
    vp = np.arange(n_vac)
    otc = []
    for m in range(im.shape[2]):
        g = np.asarray(inv(im[:, :, m], 'gibbs_AR_0'))
        dv = (_fit_quadratic_and_find_minimum(g[:, n_vac:], vp)
              - _fit_quadratic_and_find_minimum(g[:, :n_vac], vp))
        otc.append(cfg_l.mass_O * dv / m_cell)
    return np.mean(otc, axis=0)


def run(name):
    """Predicted OTC (mass fraction) on all test rows."""
    if name == 'gp':
        tr_x, tr_y, _, _ = ens_args(dm_m)
        gp = GPRegressor(tr_x, tr_y, device=dev).to(dev)
        train_GP_new(gp, tr_x, tr_y, cfg_m.default_gp_training_iters, logger=logger)
        mean, _, _ = gp.predict_decomposed(te_x)
        pred = dm_m.get_inverse_transform_fn()(mean.cpu().numpy(), 'otc')
    elif name == 'de':
        de = train_de(cfg_z, *ens_args(dm_z), 1, cfg_z.ensemble_epochs,
                      cfg_z.ensemble_lr, None)
        with torch.no_grad():
            mean = de.predict(te_x)['mean'].reshape(-1).cpu().numpy()
        pred = dm_z.get_inverse_transform_fn()(mean, 'otc')
    elif name == 'mogp':
        tr_x, tr_y, _, _ = ens_args(dm_l, ycols=range(cfg_l.n_landscape_dims))
        active, const = split_constant_columns(tr_y)
        gp = ICMGPRegressor(tr_x, tr_y[:, active], rank=cfg_l.mogp_rank, device=dev,
                            max_cholesky=cfg_l.mogp_max_cholesky)
        fit_mogp(gp, tr_x, tr_y[:, active], n_iters=cfg_l.mogp_iters, lr=cfg_l.mogp_lr,
                 patience=cfg_l.mogp_patience, logger=logger)
        pred = landscape_otc(MOGPEnsembleAdapter(gp, active, const, output_dim=cfg_l.n_landscape_dims,
                                                 n_draws=cfg_l.mogp_draws,
                                                 chunk_size=cfg_l.mogp_chunk_size))
    elif name == 'de_land':
        noise = cfg_l.build_ensemble_noise_std(gibbs_std=dm_l.norm_stats['gibbs_AR_0']['std'])
        de = train_de(cfg_l, *ens_args(dm_l, ycols=range(cfg_l.n_landscape_dims)),
                      cfg_l.n_landscape_dims, cfg_l.ensemble_epochs, cfg_l.ensemble_lr, noise)
        pred = landscape_otc(de)
    else:   # dgp, dde: AR curve only, FR from the analytic offset, as in active_learning_HEO.py
        n_vac, vp = cfg_l.n_vacancies_max, cfg_l.vacancy_points
        tr_x, tr_y, va_x, va_y = ens_args(dm_l, ycols=range(n_vac))
        gstd = dm_l.norm_stats['gibbs_AR_0']['std']
        if name == 'dgp':
            gx, gy = build_training_matrix(tr_x.cpu().numpy(), tr_y.cpu().numpy(), vp)
            gx = torch.tensor(gx, dtype=torch.float32, device=dev)
            gy = torch.tensor(gy, dtype=torch.float32, device=dev)
            gp = DeltaGPRegressor(gx, gy, device=dev,
                                  max_cholesky=cfg_l.delta_gp_max_cholesky).to(dev)
            train_delta_gp(gp, gx, gy, iters=cfg_l.delta_gp_iters, lr=cfg_l.delta_gp_lr,
                           patience=cfg_l.delta_gp_patience, min_delta=cfg_l.delta_gp_min_delta,
                           logger=logger)
            ad = DeltaGPEnsembleAdapter(gp, vp, output_dim=cfg_l.n_landscape_dims,
                                        n_draws=cfg_l.delta_gp_draws,
                                        chunk_size=cfg_l.delta_gp_chunk_size,
                                        offset=cfg_l.mu_o_fr_minus_ar, gibbs_scale=gstd)
        else:
            def expand(x, y):
                v = torch.as_tensor(vp, dtype=x.dtype, device=x.device).repeat(x.shape[0])
                return torch.cat([x.repeat_interleave(n_vac, 0), v[:, None]], 1), y.reshape(-1, 1)
            de = train_de(cfg_l, *expand(tr_x, tr_y), *expand(va_x, va_y), 1,
                          cfg_l.delta_de_epochs, cfg_l.delta_de_lr, None)
            ad = DeltaEnsembleAdapter(de, vp, output_dim=cfg_l.n_landscape_dims,
                                      chunk_size=cfg_l.delta_gp_chunk_size,
                                      offset=cfg_l.mu_o_fr_minus_ar, gibbs_scale=gstd)
        pred = landscape_otc(ad)
    return np.asarray(pred, dtype=np.float64).reshape(-1)


def at_zero(v):
    return v <= ZERO_TOL


def at_ceil(v, c):
    return v >= c * (1 - CEIL_RTOL)


groups = {'all': np.ones(len(otc_true), bool), 'zero': at_zero(otc_true),
          'ceil': at_ceil(otc_true, ceil)}
groups['mid'] = ~groups['zero'] & ~groups['ceil']
groups['nonzero'] = ~groups['zero']
sets = {'random': (te['set'] == 'random').values, 'acq': (te['set'] == 'acq').values}
# picked by an MC arm: no compared model's family chose these (common, unbiased subset)
sets['acq_mc'] = sets['acq'] & np.array([bool(a & {'mc', 'mc_clip'}) for a in te['arms']])
for s, sm in sets.items():
    print(f'{s} true groups: ' + ', '.join(f'{g} {(m & sm).sum()}' for g, m in groups.items()))


def own_picks(name):
    fam = FAMILY[name.removesuffix('_cap')]
    return np.array([bool(a & fam) for a in te['arms']])


def metrics(name, seed, pred, set_name, sm):
    row = {'model': name, 'seed': seed, 'set': set_name}
    for g, gm in groups.items():
        m = sm & gm
        e = (pred[m] - otc_true[m]) * 100   # wt%
        row[f'n_{g}'] = int(m.sum())
        row[f'mae_{g}'] = np.abs(e).mean() if m.any() else np.nan
        row[f'rmse_{g}'] = np.sqrt((e ** 2).mean()) if m.any() else np.nan
        row[f'spearman_{g}'] = spearmanr(pred[m], otc_true[m])[0] if m.sum() > 2 else np.nan
    p, t = pred[sm], otc_true[sm]
    for k in TOPK:
        if k > len(p):
            continue
        top = np.argsort(-p, kind='stable')[:k]
        row[f'top{k}_mean_true'] = t[top].mean() * 100
        row[f'hit{k}'] = (t[top] >= np.sort(t)[-k]).mean()
        row[f'top{k}_best_possible'] = np.sort(t)[-k:].mean() * 100
    row.update(frac_pred_zero=at_zero(p).mean(), frac_true_zero=at_zero(t).mean(),
               frac_pred_ceil=at_ceil(p, ceil[sm]).mean(), frac_true_ceil=at_ceil(t, ceil[sm]).mean(),
               frac_pred_above_ceil=(p > ceil[sm]).mean())
    return row


rows = []
preds = te[els[:-1] + ['set', 'otc']].copy()
preds['arms'] = te['arms'].map(lambda a: ';'.join(sorted(a)))
preds['ceil'] = ceil
for seed in SEEDS:
    for cfg in (cfg_m, cfg_z, cfg_l):
        cfg.default_seed = seed       # DE member seeds come from base_seed
    for name in MODELS:
        same_seeds(seed, full=True)
        raw = run(name)
        variants = [(name, raw)] + ([(f'{name}_cap', np.minimum(raw, ceil))]
                                    if name in ('gp', 'de') else [])
        for vname, pred in variants:
            preds[f'{vname}_s{seed}'] = pred
            for s, sm in sets.items():
                rows.append(metrics(vname, seed, pred, s, sm))
            rows.append(metrics(vname, seed, pred, 'acq_loo', sets['acq'] & ~own_picks(vname)))
        pd.DataFrame(rows).to_csv(OUT, index=False)   # written as it goes
        preds.to_csv(PRED_OUT, index=False)
        print(f'seed {seed} {name} done', flush=True)

df = pd.DataFrame(rows)
print(f'\nwrote {OUT} and {PRED_OUT}')
cols = ['mae_all', 'mae_zero', 'mae_mid', 'mae_ceil', 'spearman_all', 'spearman_nonzero',
        'frac_pred_zero', 'frac_pred_ceil'] + [c for k in TOPK for c in (f'top{k}_mean_true', f'hit{k}')]
for s in ('random', 'acq_mc', 'acq', 'acq_loo'):
    sub = df[df['set'] == s]
    agg = sub.groupby('model', sort=False)[[c for c in cols if c in sub]].agg(['mean', 'std'])
    print(f'\n=== {s}: mean +- sd over seeds (MAE and top-k OTC in wt%)')
    print(pd.DataFrame({c: agg[c]['mean'].map('{:.3f}'.format) + ' +- ' + agg[c]['std'].map('{:.3f}'.format)
                        for c in agg.columns.levels[0] if c in agg}).to_string())
