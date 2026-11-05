"""Thompson sampling over OTC for the multi-output GP.

The scalar GP has a posterior over OTC directly, so a Thompson draw is one
normal sample. The deep ensemble picks a member. The MOGP sits between the two:
its posterior is over the 14-D Gibbs landscape, so a draw is a joint 14-dim
normal sample per candidate, pushed through the same quadratic fit the DE uses.

One consequence worth exploiting: the quadratic fit is a *nonlinear* functional
of the landscape, so the induced OTC posterior is non-Gaussian and generally
asymmetric. Sampling gives it to you for free. A scalar GP cannot represent it
at all -- it is constrained to a symmetric normal around its mean.
"""
import numpy as np

from .otc import otc_from_landscape


def sample_landscapes(model, x, scaler, keep_idx, n_landscape_dims=16,
                      n_draws=1, chunk_size=512, observation_noise=False):
    """Posterior draws of the full landscape, in eV, shape [n_draws, N, 16].

    Draws the active tasks from the model, undoes the grouped z-score, and
    reinserts the structurally-zero v=0 columns so downstream code sees the
    same 16-D vector the deep ensemble produces.
    """
    draws = model.sample(x, n_samples=n_draws, chunk_size=chunk_size,
                         observation_noise=observation_noise)
    draws = draws.numpy().astype(np.float64)

    full = np.zeros((draws.shape[0], draws.shape[1], n_landscape_dims))
    full[:, :, keep_idx] = scaler.inverse_transform(draws)
    return full


def thompson_sample_otc(model, x, scaler, keep_idx, mass, n_draws=1,
                        n_vacancies_max=8, chunk_size=512,
                        observation_noise=False, return_landscapes=False):
    """Thompson draws of OTC for each candidate, shape [n_draws, N].

    `observation_noise=False` (the default) draws the latent f. That is what
    Thompson sampling wants and what matches the deep ensemble; see
    `ICMGPRegressor.sample` for why adding per-task noise here corrupts the
    quadratic fit.

    Parameters
    ----------
    model : ICMGPRegressor or IndependentMOGPRegressor
    x : torch.Tensor [N, 17]
        Decoded candidate compositions.
    scaler : GroupedZScore
        The normaliser fitted on the training landscape.
    keep_idx : np.ndarray
        Indices of the non-degenerate Gibbs columns.
    mass : np.ndarray [N]
        Supercell mass per candidate (the OTC denominator).
    """
    landscapes = sample_landscapes(
        model, x, scaler, keep_idx, n_landscape_dims=2 * n_vacancies_max,
        n_draws=n_draws, chunk_size=chunk_size, observation_noise=observation_noise,
    )
    otc = np.stack([
        otc_from_landscape(g, mass, n_vacancies_max=n_vacancies_max, fit_ar=True)
        for g in landscapes
    ])
    return (otc, landscapes) if return_landscapes else otc


def select_top_k(scores, k, exclude=None):
    """Indices of the k highest-scoring candidates.

    `scores` is one Thompson draw [N]. Greedy selection on a single posterior
    draw is the Thompson step; taking the top k of that one draw matches how
    `samples_ensemble` selects a batch from a single member.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if exclude is not None and len(exclude):
        scores = scores.copy()
        scores[np.asarray(exclude, dtype=int)] = -np.inf
    return np.argsort(-scores)[:k]


def otc_posterior_summary(otc_draws):
    """Mean, std and 95% interval of the induced (non-Gaussian) OTC posterior.

    Parameters
    ----------
    otc_draws : np.ndarray [n_draws, N]

    Returns
    -------
    dict of np.ndarray [N]
    """
    lo, hi = np.percentile(otc_draws, [2.5, 97.5], axis=0)
    return {
        'mean': otc_draws.mean(axis=0),
        'std': otc_draws.std(axis=0),
        'lo95': lo,
        'hi95': hi,
        # Nonzero skew is the signature of the quadratic fit's nonlinearity;
        # a scalar GP is structurally incapable of producing it.
        'skew': ((otc_draws - otc_draws.mean(axis=0)) ** 3).mean(axis=0)
                / np.clip(otc_draws.std(axis=0) ** 3, 1e-30, None),
    }
