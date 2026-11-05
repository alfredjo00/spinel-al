"""Delta-GP: the Gibbs landscape as one curve in a continuous vacancy coordinate.

The ICM surrogate treats the landscape as 16 unordered tasks. Two facts about
the data make that the wrong shape:

1. Eight of the sixteen columns are the *same* curve sampled at eight vacancy
   levels. Vacancy count is ordered and G varies smoothly along it, so it
   belongs in the kernel as an input dimension, not in a coregionalization
   matrix as a categorical label.

2. The other eight are the same curve again, shifted. The two reactors share a
   phonon calculation and differ only in temperature and oxygen partial
   pressure, so

       G_FR(v) - G_AR(v) = [mu_O(FR) - mu_O(AR)] * v + (small temperature term)

   with the bracket a constant of the reactor conditions. Measured over 4016
   compositions the offset at v=1 is -1.8836 +- 0.039 eV against an analytic
   -1.88362, and the residual after removing the vacancy-only part has standard
   deviation 0.04-0.08 eV on a 3.6 eV signal.

So a single GP over (composition, v) predicting the air-reactor curve carries
essentially the whole landscape. A 2-task ICM over the reactors was tried and
rejected: it can only learn a scalar task correlation, reaching 0.84 against a
true 0.9997, because the real relationship is an additive shift that grows with
v rather than a constant scaling.

Held out over 3 splits at 2500 training compositions this predicts G with 15%
lower RMSE than the rank-1 ICM (0.757 vs 0.889) and gives a slightly better OTC
(0.00246 vs 0.00274 MAE), while fitting one GP instead of a 16-task model.

The quadratic fit that turns a predicted curve into v* is deliberately kept:
with delta continuous the minimum can also be read off a fine grid, but that was
measurably worse (0.00296 vs 0.00246), because the parabola is a global fit over
all eight points and averages out prediction error, whereas the grid argmin
chases whatever wiggle the posterior happens to have. Keeping it also leaves the
surrogate as the only difference between this arm and the others.
"""
import copy

import gpytorch
import numpy as np
import torch

# mu_O(1050 C, pO2 1e-14) - mu_O(950 C, pO2 0.2), in eV per vacancy, from
# metal.applications.crystals.tools.get_mu_oxygen with the reactor conditions
# defined in update_training_database.
MU_O_FR_MINUS_AR = -1.8836231615976065


class DeltaGPRegressor(gpytorch.models.ExactGP):
    """Exact GP over [composition, vacancy_level] -> G_AR.

    Matern 5/2 with ARD, matching the scalar GP and the ICM, so the vacancy
    dimension gets its own lengthscale and the comparison between surrogates is
    not confounded by the kernel family.
    """

    def __init__(self, train_x, train_y, likelihood=None, device=None,
                 max_cholesky=32768):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.device = device or train_x.device
        # GPyTorch silently falls back to conjugate gradients above
        # max_cholesky_size (default 800), and CG does not converge on this
        # kernel: it stops at the iteration cap with a residual norm around 1e5
        # against a tolerance of 1, so the fit is meaningless. The threshold has
        # to exceed the number of *rows*, which is compositions times vacancy
        # levels -- eight times larger than for a per-composition model.
        self.max_cholesky = max_cholesky
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(-1)))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def build_training_matrix(comp, gibbs_ar, vac_points):
    """[N, d] compositions and [N, T] curves -> [N*T, d+1] inputs and [N*T] targets.

    Each composition contributes one row per vacancy level, with the level
    appended as the final input column. This is the whole data-shape change the
    model needs.
    """
    comp = np.asarray(comp, dtype=np.float64)
    gibbs_ar = np.asarray(gibbs_ar, dtype=np.float64)
    n, t = gibbs_ar.shape
    xs = np.repeat(comp, t, axis=0)
    vs = np.tile(np.asarray(vac_points, dtype=np.float64), n).reshape(-1, 1)
    return np.hstack([xs, vs]), gibbs_ar.reshape(-1)


def train_delta_gp(model, train_x, train_y, iters=1000, lr=0.02,
                   patience=200, min_delta=1e-4, logger=None):
    """Fit by marginal likelihood, stopping early when the loss stops improving.

    `min_delta` is what makes the stop actually fire. The marginal likelihood
    keeps creeping down by ~1e-5 per step long after the fit has settled, so a
    threshold near machine precision resets the patience counter forever and the
    run always reaches the iteration cap: one probe improved by 4e-4 over 709
    iterations while never triggering. A tolerance of 1e-4 treats those as noise,
    which is what they are relative to a loss around -0.68.
    """
    model.train()
    model.likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    best, best_it, history = float('inf'), 0, []
    best_state = None
    max_chol = getattr(model, 'max_cholesky', 32768)
    if max_chol <= len(train_x) and logger:
        logger.warning(
            f'Delta-GP max_cholesky ({max_chol}) <= training rows ({len(train_x)}); '
            f'GPyTorch will fall back to CG, which does not converge here.')
    with gpytorch.settings.max_cholesky_size(max_chol):
        for it in range(iters):
            opt.zero_grad()
            loss = -mll(model(train_x), train_y)
            loss.backward()
            opt.step()
            val = float(loss.item())
            history.append(val)
            # Same cadence as fit_gp for the ICM, so the two arms' logs can be
            # compared directly and a stalled fit is visible while it runs
            # rather than only from the final warning.
            if logger and (it % 50 == 0 or it == iters - 1):
                logger.info(f'Delta-GP iter {it}/{iters}: loss {val:.4f} '
                            f'(best {min(best, val):.4f} @ {best_it})')
            if val < best - min_delta:
                best, best_it = val, it
                # Keep the parameters that produced the best loss. Patience only
                # expires after the fit has moved *away* from its optimum, so
                # whatever the last iteration leaves behind is by construction
                # worse than what was already found.
                best_state = copy.deepcopy(model.state_dict())
            elif it - best_it >= patience:
                if logger:
                    logger.info(f'Delta-GP: early stop at iter {it} (best {best:.4f})')
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if logger:
            logger.info(f'Delta-GP: restored parameters from iter {best_it} '
                        f'(loss {best:.4f})')
    model.eval()
    model.likelihood.eval()
    return history


@torch.no_grad()
def predict_curves(model, comp, vac_points, chunk=256, n_samples=0,
                   observation_noise=False):
    """Predicted AR curves for [N, d] compositions, optionally sampled.

    Returns
    -------
    mean : [N, T]
    std  : [N, T]
    draws : [n_samples, N, T] or None
        Joint draws over the vacancy axis. Sampling jointly is the point: an
        independent draw per vacancy level gives a jagged curve whose minimum is
        dominated by noise, while a draw from the GP posterior over the vacancy
        input is a smooth function, so each sample yields one coherent curve and
        therefore one OTC estimate -- the same semantics as one ensemble member.
    """
    dev = model.device
    t = len(vac_points)
    n = len(comp)
    comp = np.asarray(comp, dtype=np.float64)
    vac = np.asarray(vac_points, dtype=np.float64)

    means = torch.empty(n, t, device=dev)
    stds = torch.empty(n, t, device=dev)
    draws = torch.empty(n_samples, n, t, device=dev) if n_samples else None

    max_chol = getattr(model, 'max_cholesky', 32768)
    # fast_pred_var (LOVE) is deliberately OFF here. It returns a low-rank
    # approximation of the joint covariance which is accurate for the marginal
    # variances exposed by .variance, but is not a valid covariance matrix when
    # materialised: measured on this model every diagonal entry of the extracted
    # per-candidate blocks came out negative (min -0.10 against a true 7e-03).
    # Curves drawn from that are noise, not samples of the posterior. The exact
    # path costs more but is the only correct one when the full covariance is
    # needed rather than just its diagonal.
    with gpytorch.settings.fast_pred_var(False), \
         gpytorch.settings.max_cholesky_size(max_chol):
        for i in range(0, n, chunk):
            cb = comp[i:i + chunk]
            m = len(cb)
            xs = np.repeat(cb, t, axis=0)
            vs = np.tile(vac, m).reshape(-1, 1)
            xb = torch.as_tensor(np.hstack([xs, vs]), dtype=torch.float32,
                                 device=dev)
            # Draws come from the latent f, not the predictive distribution.
            # The likelihood adds sigma^2 to the diagonal only, so including it
            # leaves the mean untouched but makes each vacancy level nearly
            # independent: measured on a fitted model the correlation between
            # neighbouring levels falls from 0.87 to 0.37, and draws stop
            # looking like curves. The quadratic fit downstream reads curvature,
            # so that jitter would land directly in Delta v. This matches
            # ICMGPRegressor.sample, whose docstring records the same failure.
            f_post = model(xb)
            post = model.likelihood(f_post) if observation_noise else f_post
            means[i:i + chunk] = post.mean.reshape(m, t)
            # Marginal variances come from the diagonal of the reshaped
            # covariance rather than post.stddev. The latter reduces over the
            # whole chunk's joint covariance, which is near rank-deficient once
            # rounding makes candidates near-duplicates, and returns small
            # negative variances that GPyTorch clamps with a warning. The
            # diagonal of each candidate's own block is the same quantity
            # computed without that reduction.
            cov_full = post.covariance_matrix.reshape(m, t, m, t)
            rows_ = torch.arange(m, device=cov_full.device)
            cov_blocks = cov_full[rows_, :, rows_, :]              # [m, t, t]
            var = cov_blocks.diagonal(dim1=-2, dim2=-1)
            stds[i:i + chunk] = var.clamp_min(1e-12).sqrt()
            if n_samples:
                # Sample each candidate's curve from its own t x t covariance
                # rather than from the joint (m*t) x (m*t) block.
                #
                # Only the vacancy axis needs to be drawn jointly -- that is what
                # makes a draw a smooth curve. Candidates are independent draws
                # anyway, matching how samples_GP treats the scalar GP, so the
                # cross-candidate blocks contribute nothing but cost. They also
                # make the matrix badly conditioned: after rounding to discrete
                # atom counts many candidates are near-duplicates, so the joint
                # covariance is close to rank-deficient and Cholesky fails
                # through escalating jitter into an eigendecomposition, at which
                # point negative variances appear and are clamped.
                cov_b = cov_blocks                            # [m, t, t]
                mu_b = post.mean.reshape(m, t)

                # Draw through a Cholesky factor rather than handing the
                # covariance to MultivariateNormal, which rejects anything not
                # numerically positive definite. Jitter is scaled to the
                # diagonal because the absolute scale of the posterior shrinks
                # as the fit sharpens: a fixed 1e-6 is large early on and far
                # too small once the diagonal is ~1e-3.
                eye = torch.eye(t, device=cov_b.device, dtype=cov_b.dtype)
                diag = cov_b.diagonal(dim1=-2, dim2=-1).mean(-1)
                jitter = (diag.clamp_min(1e-12) * 1e-4).view(-1, 1, 1)
                L = None
                for _ in range(6):
                    L, info = torch.linalg.cholesky_ex(cov_b + jitter * eye)
                    if not bool(info.any()):
                        break
                    jitter = jitter * 10
                if L is None or bool(info.any()):
                    # Give up on the joint draw for this chunk and fall back to
                    # independent per-point noise. The curve is no longer smooth,
                    # so this is a last resort rather than an equivalent path.
                    sd = post.stddev.reshape(m, t)
                    eps = torch.randn(n_samples, m, t, device=mu_b.device)
                    draws[:, i:i + chunk] = mu_b + eps * sd
                else:
                    eps = torch.randn(n_samples, m, t, 1, device=L.device,
                                      dtype=L.dtype)
                    draws[:, i:i + chunk] = mu_b + (L @ eps).squeeze(-1)
    return means, stds, draws


class DeltaGPEnsembleAdapter:
    """Present a fitted Delta-GP as a trained DeepEnsemble.

    Mirrors MOGPEnsembleAdapter so the acquisition code is untouched: the full
    16-wide landscape is reconstructed from the eight predicted AR points plus
    the analytic reactor offset, and each posterior draw becomes one 'member'.
    Because both reactors come from the same draw they are perfectly correlated,
    which matches the physics -- they are the same phonon calculation evaluated
    at two conditions -- and is what an OTC estimate needs.
    """

    def __init__(self, model, vac_points, output_dim=16, n_draws=16,
                 chunk_size=256, offset=MU_O_FR_MINUS_AR, gibbs_scale=1.0):
        self.model = model
        self.vac_points = np.asarray(vac_points, dtype=np.float64)
        self.n_vac = len(self.vac_points)
        self.output_dim = output_dim
        self.n_members = n_draws
        self.chunk_size = chunk_size
        self.device = model.device
        self.temperature = 1.0
        # The offset is a physical energy; targets are z-scored by a shared
        # Gibbs scale, so it has to be expressed in the same normalised units.
        self.offset = torch.as_tensor(
            offset * self.vac_points / gibbs_scale,
            dtype=torch.float32, device=self.device)

    def _landscape(self, ar):
        """[..., T] air-reactor values -> [..., 2T] landscape ordered [AR, FR]."""
        return torch.cat([ar, ar + self.offset], dim=-1)

    def predict(self, x):
        x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        mean, std, draws = predict_curves(
            self.model, x_np, self.vac_points,
            chunk=self.chunk_size, n_samples=self.n_members)

        full_mean = self._landscape(mean)                       # [N, 2T]
        # The offset is deterministic, so FR carries the same uncertainty as AR.
        full_std = torch.cat([std, std], dim=-1)
        full_draws = self._landscape(draws)                     # [S, N, 2T]
        individual_means = full_draws.permute(1, 2, 0)          # [N, 2T, S]

        # Epistemic/aleatoric split: the likelihood's learned noise is the
        # aleatoric part, the rest of the predictive variance is epistemic.
        noise = self.model.likelihood.noise.detach().reshape(-1)[0]
        aleatoric = noise.sqrt().expand_as(full_std)
        epistemic = (full_std ** 2 - aleatoric ** 2).clamp_min(1e-12).sqrt()

        log_vars = (aleatoric ** 2).log().unsqueeze(-1).expand(
            -1, -1, self.n_members)

        return {
            'individual_means': individual_means,
            'individual_log_vars': log_vars,
            'mean': full_mean,
            'aleatoric_std': aleatoric,
            'epistemic_std': epistemic,
            'total_std': full_std,
        }

    def eval(self):
        self.model.eval()
        self.model.likelihood.eval()
        return self

    def parameters(self):
        return self.model.parameters()


class DeltaEnsembleAdapter:
    """Present a DeepEnsemble trained on delta-as-input rows as a landscape model.

    The ensemble itself is the production one, trained by the production
    `train_deep_ensemble` with the same validation split, early stopping,
    anchoring and noise settings. Only the data shape differs: it sees one row
    per (composition, vacancy level) with a single output, instead of one row
    per composition with sixteen.

    This exists so the delta-as-input representation can be compared against the
    ICM within *both* surrogate families, rather than being confounded with the
    choice of GP.

    A member plays the role a posterior draw plays for the GP: it predicts a
    whole AR curve, the FR curve follows from the analytic offset, and the pair
    gives one OTC estimate. Unlike a GP draw a member's curve is not smooth by
    construction -- nothing couples its predictions across vacancy levels except
    the network having learned the trend -- which is exactly the difference the
    comparison is meant to expose.
    """

    def __init__(self, ensemble, vac_points, output_dim=16, chunk_size=256,
                 offset=MU_O_FR_MINUS_AR, gibbs_scale=1.0):
        self.ensemble = ensemble
        self.vac_points = np.asarray(vac_points, dtype=np.float64)
        self.n_vac = len(self.vac_points)
        self.output_dim = output_dim
        self.n_members = ensemble.n_members
        self.chunk_size = chunk_size
        self.device = ensemble.device
        self.temperature = getattr(ensemble, 'temperature', 1.0)
        self.offset = torch.as_tensor(
            offset * self.vac_points / gibbs_scale,
            dtype=torch.float32, device=self.device)

    def _expand(self, x):
        """[N, d] compositions -> [N*T, d+1] rows with the vacancy level appended."""
        n = x.shape[0]
        vac = torch.as_tensor(self.vac_points, dtype=x.dtype, device=x.device)
        xs = x.repeat_interleave(self.n_vac, dim=0)
        vs = vac.repeat(n).unsqueeze(-1)
        return torch.cat([xs, vs], dim=-1)

    def _landscape(self, ar):
        return torch.cat([ar, ar + self.offset], dim=-1)

    def predict(self, x):
        x = x.to(self.device)
        n = x.shape[0]
        out = self.ensemble.predict(self._expand(x))

        # [N*T, 1, M] -> [N, T, M]; the ensemble is single-output here.
        im = out['individual_means'].reshape(n, self.n_vac, -1)
        lv = out['individual_log_vars'].reshape(n, self.n_vac, -1)
        mean = out['mean'].reshape(n, self.n_vac)
        ale = out['aleatoric_std'].reshape(n, self.n_vac)
        epi = out['epistemic_std'].reshape(n, self.n_vac)
        tot = out['total_std'].reshape(n, self.n_vac)

        # Each member's AR curve plus the deterministic offset is that member's
        # landscape, so a member yields one OTC exactly as a GP draw does.
        full_means = torch.cat([im, im + self.offset.view(1, -1, 1)], dim=1)
        full_lv = torch.cat([lv, lv], dim=1)

        return {
            'individual_means': full_means,
            'individual_log_vars': full_lv,
            'mean': self._landscape(mean),
            'aleatoric_std': torch.cat([ale, ale], dim=-1),
            'epistemic_std': torch.cat([epi, epi], dim=-1),
            'total_std': torch.cat([tot, tot], dim=-1),
        }

    def eval(self):
        self.ensemble.eval()
        return self

    def parameters(self):
        return self.ensemble.parameters()
