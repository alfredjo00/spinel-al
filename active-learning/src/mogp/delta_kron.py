"""Ways of locating the minimum of G(delta), for the delta-as-input GP.

The quadratic fit in the existing pipeline is a workaround for the ICM model:
it only ever produces G at the 8 sampled delta points, so a parabola is the
cheapest way to interpolate between them and find v*. A GP that takes delta as
an input has no such restriction -- it can be evaluated at any delta -- so the
minimum can be read off a densely sampled curve instead of assumed parabolic.

`min_via_quadratic` reproduces the current pipeline exactly, so the two can be
compared on the same predictions.
"""
import numpy as np

MASS_O = 15.999


def min_via_quadratic(g, deltas):
    """Current pipeline: least-squares parabola through all sampled points.

    Mirrors `mogp/otc.py:fit_quadratic_minimum`, including its handling of the
    concave and near-linear cases, so any difference against the grid method is
    the interpolation and not a change of convention.
    """
    g = np.atleast_2d(np.asarray(g, float))
    deltas = np.asarray(deltas, float)
    lo, hi = deltas[0], deltas[-1]

    V = np.vstack([deltas ** 2, deltas, np.ones_like(deltas)]).T
    coeffs = g @ np.linalg.pinv(V).T
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    def polyval(v):
        return a * v ** 2 + b * v + c

    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.clip(np.where(a != 0, -b / (2 * a), lo), lo, hi)
    # concave: minimum is at whichever endpoint is lower
    v = np.where(a < 0, np.where(polyval(lo) < polyval(hi), lo, hi), v)
    # effectively linear: minimum at the downhill endpoint
    v = np.where(np.abs(a) < 1e-10, np.where(b > 0, lo, hi), v)
    return v


def min_via_grid(g_fine, deltas_fine, refine=True):
    """Minimum read straight off a densely sampled curve.

    With `refine`, a parabola through the three points bracketing the discrete
    minimum gives sub-grid resolution, so the answer is limited by the model
    rather than by the grid spacing. Unlike `min_via_quadratic` this parabola is
    local, so it does not assume the whole curve is quadratic.
    """
    g_fine = np.atleast_2d(np.asarray(g_fine, float))
    deltas_fine = np.asarray(deltas_fine, float)
    idx = np.argmin(g_fine, axis=1)
    v = deltas_fine[idx].astype(float)
    if not refine:
        return v

    m = g_fine.shape[1]
    interior = (idx > 0) & (idx < m - 1)
    if not interior.any():
        return v
    rows = np.where(interior)[0]
    i = idx[rows]
    y0, y1, y2 = g_fine[rows, i - 1], g_fine[rows, i], g_fine[rows, i + 1]
    denom = y0 - 2 * y1 + y2
    step = deltas_fine[1] - deltas_fine[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    v[rows] = deltas_fine[i] + np.clip(shift, -1.0, 1.0) * step
    return np.clip(v, deltas_fine[0], deltas_fine[-1])


def otc_from_v(v_ar, v_fr, mass):
    """OTC (mass fraction) = M_O * (v*_FR - v*_AR) / M_supercell."""
    return MASS_O * (np.asarray(v_fr, float) - np.asarray(v_ar, float)) \
        / np.asarray(mass, float)
