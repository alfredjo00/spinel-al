"""Expose an ICM multi-output GP through the DeepEnsemble prediction interface.

`samples_ensemble` / `_predict_otc_from_landscape` already implement candidate
generation, composition rounding, quadratic-fit OTC derivation, empirical
Thompson sampling and all the per-member CSV logging. None of that is specific
to a deep ensemble -- it only needs an object with `.device`, `.n_members` and
a `.predict()` returning per-member landscapes.

So instead of duplicating that pipeline for the GP, this adapter maps **one ICM
posterior draw to one "ensemble member"**. That is not a hack: drawing a member
uniformly at random (al_functions.py:1604) and drawing a posterior sample are
the same operation, so the existing empirical Thompson path becomes exactly
correct Thompson sampling for the GP, rather than the parametric Gaussian
approximation the scalar GP baseline uses.

Draws use `observation_noise=False`. Each member must be a *smooth* landscape,
because the quadratic fit downstream reads curvature; per-task observation noise
is independent across vacancy levels and would corrupt the fitted minimum.
"""
import numpy as np
import torch


def split_constant_columns(train_y, tol=1e-9):
    """Separate degenerate target columns from the ones worth modelling.

    `gibbs_AR_0` and `gibbs_FR_0` are identically zero by construction (the v=0
    reference). Left in, they give the coregionalization matrix zero rows and
    drive the per-task noise onto its constraint boundary. They are dropped from
    the GP and reinstated afterwards at their constant value.

    Note the constant is taken in *normalised* space: raw 0 maps to
    (0 - mean)/std under the grouped z-score, which is not 0.

    Parameters
    ----------
    train_y : torch.Tensor [N, D]

    Returns
    -------
    (active_idx, const_map) : (list[int], dict[int, float])
    """
    stds = train_y.std(dim=0)
    active_idx, const_map = [], {}
    for i in range(train_y.size(1)):
        if stds[i].item() > tol:
            active_idx.append(i)
        else:
            const_map[i] = float(train_y[0, i].item())
    return active_idx, const_map


class MOGPEnsembleAdapter:
    """Wrap an ICMGPRegressor so it quacks like a trained DeepEnsemble.

    Parameters
    ----------
    model : ICMGPRegressor
        Already fitted on `train_y[:, active_idx]`.
    active_idx : list[int]
        Columns the GP actually models.
    const_map : dict[int, float]
        Columns held at a constant (normalised) value.
    output_dim : int
        Full landscape width the pipeline expects (16).
    n_draws : int
        Posterior draws per prediction; plays the role of ensemble size. Match
        it to `--ensemble-size` so acquisition noise has comparable granularity
        across surrogates.
    chunk_size : int
        Candidates per posterior draw batch. Smaller is *cheaper* here: cost
        scales as N * chunk^2 * T^3, so 128 beats 512 by ~16x. It is also closer
        to the intended semantics, since draws should be independent across
        candidates anyway (matching `samples_GP`).
    """

    def __init__(self, model, active_idx, const_map, output_dim,
                 n_draws=16, chunk_size=128):
        self.model = model
        self.active_idx = list(active_idx)
        self.const_map = dict(const_map)
        self.output_dim = output_dim
        self.n_members = n_draws
        self.chunk_size = chunk_size
        self.device = model.device
        self.temperature = 1.0

    def _scatter(self, values, n_rows, trailing=None):
        """Place active-task values into the full landscape width."""
        shape = (n_rows, self.output_dim) if trailing is None else (n_rows, self.output_dim, trailing)
        out = torch.empty(shape, dtype=torch.float32, device=self.device)
        for col, const in self.const_map.items():
            out[:, col] = const
        idx = torch.as_tensor(self.active_idx, device=self.device)
        out[:, idx] = values
        return out

    def predict(self, x):
        """Mimic `DeepEnsemble.predict`.

        The uncertainty split is exact here rather than estimated: `epistemic`
        is the std of the latent f and `aleatoric` is what the likelihood's
        learned per-task and global noise add on top, whereas a deep ensemble
        infers the same two from member spread and mean predicted variance.

        Returns
        -------
        dict with
            individual_means    [N, D, M] -- one posterior landscape per draw
            individual_log_vars [N, D, M] -- aleatoric log-variance, broadcast
            mean                [N, D]    -- posterior mean
            aleatoric_std       [N, D]
            epistemic_std       [N, D]
            total_std           [N, D]    -- predictive std
        """
        x = x.to(self.device)
        n = len(x)

        draws = self.model.sample(x, n_samples=self.n_members,
                                  chunk_size=self.chunk_size,
                                  observation_noise=False)          # [M, N, T]
        mean, epistemic, total = self.model.predict_decomposed(x, chunk_size=self.chunk_size)

        draws = draws.permute(1, 2, 0).to(self.device)               # [N, T, M]
        individual_means = self._scatter(draws, n, trailing=self.n_members)

        # Aleatoric variance as the residual of the exact split. Taking it from
        # `task_noises` alone would miss the likelihood's global noise term.
        epistemic = epistemic.to(self.device)
        total = total.to(self.device)
        aleatoric = (total ** 2 - epistemic ** 2).clamp_min(1e-12).sqrt()   # [N, T]

        log_var_active = (aleatoric ** 2).log().unsqueeze(-1).expand(-1, -1, self.n_members)
        individual_log_vars = self._scatter(log_var_active, n, trailing=self.n_members)
        for col in self.const_map:
            individual_log_vars[:, col] = -20.0   # exp(-20) ~ 2e-9 variance

        full_mean = self._scatter(mean.to(self.device), n)
        full_total = self._scatter(total, n)
        full_epistemic = self._scatter(epistemic, n)
        full_aleatoric = self._scatter(aleatoric, n)
        for col in self.const_map:
            full_total[:, col] = 1e-6
            full_epistemic[:, col] = 1e-6
            full_aleatoric[:, col] = 1e-6

        return {
            'individual_means': individual_means,
            'individual_log_vars': individual_log_vars,
            'mean': full_mean,
            'aleatoric_std': full_aleatoric,
            'epistemic_std': full_epistemic,
            'total_std': full_total,
        }

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()
