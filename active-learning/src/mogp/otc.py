"""Derive OTC from a predicted Gibbs energy landscape.

Vendored from `al_functions._fit_quadratic_and_find_minimum` /
`compute_delta_v_from_gibbs`, with two deliberate changes:

1. The quadratic fit is vectorised (one lstsq pseudo-inverse applied to the
   whole batch) instead of looping `np.polyfit` per sample. Identical result,
   but it matters when scoring 25k candidates per AL cycle.
2. `fit_ar` defaults to True. The stored `vacancy_diff_quadratic_fit` in
   training.db equals `v*_FR - v*_AR` with quadratics fit to *both* reactors
   (verified to 1e-14); the `fit_ar=False` shortcut in the main repo is an
   acquisition-time approximation, not the ground-truth definition.
"""
import numpy as np

MASS_O = 15.999


def fit_quadratic_minimum(energies, vacancy_points):
    """Vacancy level minimising a quadratic fit of G(v), per row.

    Parameters
    ----------
    energies : np.ndarray [N, n_vac]
    vacancy_points : np.ndarray [n_vac]

    Returns
    -------
    np.ndarray [N]
        Optimal vacancy level, clipped to [v_min, v_max].
    """
    energies = np.atleast_2d(np.asarray(energies, dtype=np.float64))
    v_min, v_max = vacancy_points[0], vacancy_points[-1]

    # Least-squares quadratic: coeffs = pinv(V) @ y, V = [v^2, v, 1]
    V = np.vstack([vacancy_points**2, vacancy_points, np.ones_like(vacancy_points)]).T
    coeffs = energies @ np.linalg.pinv(V).T          # [N, 3] -> a, b, c
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    def polyval(v):
        return a * v**2 + b * v + c

    # Default branch: convex parabola, interior minimum clipped to range.
    with np.errstate(divide='ignore', invalid='ignore'):
        v_opt = np.clip(np.where(a != 0, -b / (2 * a), v_min), v_min, v_max)

    # Concave: minimum sits at whichever endpoint is lower.
    concave = a < 0
    v_opt = np.where(concave, np.where(polyval(v_min) < polyval(v_max), v_min, v_max), v_opt)

    # Effectively linear: minimum at the downhill endpoint.
    linear = np.abs(a) < 1e-10
    v_opt = np.where(linear, np.where(b > 0, v_min, v_max), v_opt)

    return v_opt


def delta_v_from_landscape(gibbs, n_vacancies_max=8, fit_ar=True):
    """Δv = v*_FR - v*_AR from a [N, 2*n_vac] landscape ordered [AR..., FR...]."""
    gibbs = np.atleast_2d(np.asarray(gibbs, dtype=np.float64))
    points = np.arange(n_vacancies_max, dtype=np.float64)

    ar, fr = gibbs[:, :n_vacancies_max], gibbs[:, n_vacancies_max:]
    v_ar = fit_quadratic_minimum(ar, points) if fit_ar else np.zeros(len(gibbs))
    v_fr = fit_quadratic_minimum(fr, points)
    return v_fr - v_ar


def otc_from_landscape(gibbs, mass, n_vacancies_max=8, fit_ar=True):
    """OTC (mass fraction) = M_O * Δv / M_supercell."""
    delta_v = delta_v_from_landscape(gibbs, n_vacancies_max, fit_ar)
    return MASS_O * delta_v / np.asarray(mass, dtype=np.float64)


def check_against_reference(dataset, atol=1e-8):
    """Sanity-check the vendored pipeline against the values stored in the db.

    Returns (max_dv_error, max_otc_error). Both should be at machine precision;
    if not, the vendored fit has drifted from the pipeline that generated the
    training targets and every downstream metric is suspect.
    """
    dv = delta_v_from_landscape(dataset.Y, fit_ar=True)
    otc = otc_from_landscape(dataset.Y, dataset.mass, fit_ar=True)
    return (
        float(np.abs(dv - dataset.dv_ref).max()),
        float(np.abs(otc - dataset.otc_ref).max()),
    )
