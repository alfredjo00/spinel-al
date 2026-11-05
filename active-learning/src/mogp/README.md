# Multi-output GP (ICM) for the spinel landscape

Fills the empty GP × multi-output cell in the Part 1 surrogate comparison: the
deep ensemble already has all three target representations, while the GP and MC
baselines only ever predicted scalar OTC.

## Why ICM

Diagnostics on the training data drove the choice (740-row db; the same picture
holds on the 2280-row `spinel_random` set):

| | |
|---|---|
| Design | complete grid — every composition has all 16 Gibbs values |
| `gibbs_AR_0`, `gibbs_FR_0` | identically zero (v=0 reference) → 14 real tasks |
| Mean \|corr\| between tasks | 0.93 (min 0.81) |
| PCA | PC1 = 97.7%, 3 PCs = 99.0% |
| Per-task std | grows ~linearly with v, 1.14 → 6.40 eV |

Complete grid ⇒ GPyTorch keeps Kronecker structure, so inference is
O(N³ + T³) and exact — no variational approximation to confound the comparison
against the exact scalar GP baseline. Near-collinear tasks ⇒ ICM's single shared
input kernel is enough; LMC's per-latent lengthscales buy nothing. N in the low
thousands ⇒ LVMOGP, convolved processes and deep coregionalization are all
overkill.

## The rank trap

PC1 alone explains 97.7% of the Gibbs variance, so a rank-1 ICM may look
excellent on Gibbs RMSE while being useless for acquisition: OTC depends on
where dG/dδ crosses zero, i.e. on the curvature residual living in PC2/PC3.
**Select rank on downstream OTC error, never on Gibbs RMSE.** That is why
`benchmark.py` reports OTC RMSE, Spearman ρ and precision@k as the primary
metrics and Gibbs RMSE only as a diagnostic.

That prediction turned out to be wrong in one direction and right in the other:
rank 1 wins on *both* OTC and Gibbs, and performance degrades monotonically with
rank (higher ranks drive the per-task noise onto its 0.07 constraint floor —
overfitting). Hence `--mogp-rank 1` as the default.

## Benchmark results

5-fold CV on `spinel_random` (2280 rows), job 17200773:

| model | OTC RMSE (wt%) | Spearman ρ | prec@10 | prec@25 | Gibbs RMSE (eV) |
|---|---|---|---|---|---|
| scalar GP (paper baseline) | 0.370 ± 0.02 | 0.765 ± 0.01 | 0.340 | 0.768 | — |
| independent-14 GP | 0.324 ± 0.01 | 0.849 ± 0.01 | 0.600 | 0.760 | 1.042 |
| **ICM rank 1** | **0.315 ± 0.03** | **0.859 ± 0.02** | **0.640** | **0.808** | **1.034** |
| ICM rank 2 | 0.349 ± 0.02 | 0.846 ± 0.02 | 0.620 | 0.744 | 1.119 |
| ICM rank 3 | 0.385 ± 0.02 | 0.826 ± 0.03 | 0.600 | 0.752 | 1.190 |
| ICM rank 5 | 0.439 ± 0.04 | 0.811 ± 0.02 | 0.620 | 0.760 | 1.319 |

Paired one-sided Wilcoxon over the 5 shared folds (0.0312 is the n=5 floor):

- independent-14 vs scalar — 5/5 folds, p = 0.0312 (both ρ and RMSE)
- ICM rank 1 vs scalar — 5/5 folds, p = 0.0312
- ICM rank 1 vs independent-14 — 3/5 folds, p = 0.156 (ρ) / 0.219 (RMSE)
- ICM rank 2 vs rank 1 — 1/5 folds, p = 0.906

So the multi-output **representation** is a significant win (ρ 0.765 → 0.849);
explicit coregionalization on top of it is not detectable at this sample size.
Same shape as the existing deep-ensemble ablation.

Learned per-task noise at rank 1, which independently reproduces the 0.2 → 0.8 eV
ramp `al_config.build_ensemble_noise_std` hard-codes:

```
AR v=1..7: 0.37 0.40 0.43 0.48 0.58 0.57 0.65
FR v=1..7: 0.40 0.43 0.45 0.49 0.59 0.58 0.66
```

## Contents

- `data.py` — loads an ASE db into `X [N,17]`, `Y [N,16]`, mass and reference
  OTC. Mirrors `al_functions.create_df` + `DataManager` featurisation exactly
  (per-sublattice fractions), with no imports from the main packages.
- `otc.py` — vendored quadratic-fit → Δv → OTC. Vectorised (one pseudo-inverse
  for the whole batch instead of per-row `np.polyfit`), and `fit_ar=True` by
  default: the stored `vacancy_diff_quadratic_fit` is `v*_FR − v*_AR` with
  quadratics fit to *both* reactors, verified to 1e-14. `check_against_reference`
  asserts this on every run.
- `models.py` — `ICMGPRegressor`, `IndependentMOGPRegressor` (the ablation with
  the same target representation but no cross-task correlation), and
  `ScalarGPRegressor` (the current paper baseline). Same
  `__init__(train_x, train_y, ...)` / `.predict(x)` shape as
  `al_functions.GPRegressor`, so they drop into `active_learning_HEO.py`.
- `acquisition.py` — standalone Thompson sampling / top-k selection over the
  induced OTC posterior, used by the offline studies.
- `ensemble_adapter.py` — presents a fitted ICM through the `DeepEnsemble`
  prediction interface so `samples_ensemble` runs unchanged; see below.
- `benchmark.py` — k-fold CV comparing all three.
- `job_benchmark.sh` — Berzelius sbatch wrapper; forwards all arguments.

## Use in the AL loop

Wired into `active_learning_HEO.py` as `--model mogp`. It requires
`--predict-landscape` (the ICM models the joint landscape; on scalar OTC it
degenerates to `--model gp`) and `--quadratic-fit-landscape-otc` (OTC is derived
per posterior draw by fitting that draw's G(v) curve). `Config.validate` enforces
both. Normalisation follows the ensemble's grouped z-score, not min-max: the
coregionalization matrix ties all tasks, so they must share a scale.

```bash
python active_learning_HEO.py --model mogp \
    --predict-landscape --quadratic-fit-landscape-otc \
    --sampling thompson --ensemble-size 16 --mogp-rank 1
```

Flags: `--mogp-rank`, `--mogp-iters`, `--mogp-lr`, `--mogp-patience`,
`--mogp-draws` (defaults to `--ensemble-size`), `--mogp-chunk-size`,
`--mogp-max-cholesky`.

`MOGPEnsembleAdapter` maps **one posterior draw to one ensemble member**, which
reuses the whole tested OTC pipeline (rounding, quadratic fit, per-member CSV
logging, empirical Thompson). It also makes acquisition *more* correct than the
scalar GP baseline: `_apply_acquisition_strategy` picks a member uniformly at
random per candidate, and for the ICM that is exact Thompson sampling instead of
the parametric-Gaussian approximation `samples_GP` uses. Draws use
`observation_noise=False` — per-task noise is independent across vacancy levels
and would corrupt the fitted minimum the quadratic reads.

## Running the benchmark

```bash
sbatch -t 0:30:00 -J mogp_probe mogp/job_benchmark.sh \
    --folds 2 --iters 100 --ranks 3 --models icm

sbatch -t 3:00:00 -J mogp_full mogp/job_benchmark.sh \
    --folds 5 --iters 2000 --lr 0.02 --patience 300 \
    --ranks 1,2,3,5 --models scalar,independent,icm
```

Convergence matters more than usual here, and **the learned per-task noise is the
diagnostic**. At 100 iterations it sits at ~3.9 eV for every task, which is just
the GPyTorch initialisation. A converged fit instead recovers something close to
the 0.2 → 0.8 eV ramp that `al_config.build_ensemble_noise_std` hard-codes. Flat
noise across tasks means under-converged, whatever the loss looks like.

The rank-1 result above came from `--iters 3000 --lr 0.02 --patience 500`, where
the ICM early-stopped between 1153 and 1801 iterations per fold at a final loss
of about −0.73. For comparison, 400 iterations at lr 0.01 stops at the cap with
loss −0.47, still descending, and noise flat at 0.80–0.85 eV. Those are the
values `--mogp-iters` / `--mogp-lr` / `--mogp-patience` default to in
`al_config`, so the AL loop converges the same way the benchmark did.

## Design notes

- `MultitaskMean` gives each task its own constant. Not optional: raw task means
  span 0 → 10.5 eV.
- Likelihood uses `rank=0` (independent noise per task + one global term). Keeps
  the Kronecker fast path and lets the model *learn* the vacancy-dependent noise
  ramp rather than having it hard-coded, which is the GP's analogue of the deep
  ensemble's heteroskedastic head.
- `ICMGPRegressor.sample` draws jointly over tasks, so each Thompson draw is a
  coherent G(v) curve. This is the actual methodological argument for the ICM in
  the AL loop — 14 independent GPs give 14 independently jittered points that the
  quadratic fit then has to make sense of.
