"""Fitting smooth coherence fields in the ordered coordinates.

Each coordinate is a complex basis expansion over a time-frequency grid, mapped smoothly into the unit disc.
Every parameter vector then gives admissible matrices, so an ordinary unconstrained optimizer can be used
without a repair inside the objective.
"""
import numpy as np
from scipy.optimize import minimize

from .coordinates import assemble, from_coordinates
from .repair import higham_projection


def to_unit_disk(z, radius=.999):
    """Smooth map of unconstrained complex numbers into |z| < radius."""
    return radius * z / np.sqrt(1. + abs(z) ** 2)


def grid_indices(freqs, times, n_freq=6, n_time=6, band=(.5, 12.)):
    """Indices of an evenly spaced n_freq x n_time grid of bins inside the frequency band."""
    eligible = np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))
    fi = eligible[np.linspace(0, len(eligible) - 1, n_freq, dtype=int)]
    ti = np.linspace(0, len(times) - 1, n_time, dtype=int)
    return fi, ti


def bilinear_basis(n_freq, n_time):
    """Basis 1, f, t, f t on a grid scaled to [-1, 1] in frequency and time; shape (n_freq, n_time, 4)."""
    f, t = np.meshgrid(np.linspace(-1, 1, n_freq), np.linspace(-1, 1, n_time), indexing='ij')
    return np.stack([np.ones_like(f), f, t, f * t], axis=-1)


def fields_from_parameters(theta, basis, formulation='coordinates'):
    """Matrices (batch, n_freq, n_time, 3, 3) from parameter vectors theta (batch, 6 * n_basis).

    For 'pairwise+higham' these are the raw pairwise matrices; fit_coherence applies the projection.
    """
    if formulation not in ('coordinates', 'pairwise+higham'):
        raise ValueError("formulation must be 'coordinates' or 'pairwise+higham'")
    p = np.reshape(theta, (-1, 3, basis.shape[-1], 2))
    fields = np.einsum('ftk,bok->boft', basis, p[..., 0] + 1j * p[..., 1], optimize=False)
    first, second, third = np.moveaxis(to_unit_disk(fields), 1, 0)
    if formulation == 'coordinates':
        return from_coordinates(first, second, third)
    return assemble(first, second, third)


def fit_coherence(target, weights, basis=None, formulation='coordinates', objective='norm', theta0=None,
                  maxiter=1000, gtol=1e-6, step=5e-6):
    """Fit smooth coherence fields to target matrices (n_freq, n_time, 3, 3) with weights (n_freq, n_time).

    formulation='coordinates': the expansions give alpha, beta and delta, so every iterate is admissible.
    formulation='pairwise+higham': they give c12, c13 and c23, and Higham projection repairs every evaluation.
    objective: 'norm' (weighted sum of Frobenius norms) or 'squared' (weighted sum of their squares).
    BFGS with central finite differences. Returns the parameters, the fitted matrices and the optimizer result.
    """
    if formulation not in ('coordinates', 'pairwise+higham'):
        raise ValueError("formulation must be 'coordinates' or 'pairwise+higham'")
    if objective not in ('norm', 'squared'):
        raise ValueError("objective must be 'norm' or 'squared'")
    w = weights / max(weights.sum(), 1e-300)
    basis = bilinear_basis(*target.shape[:2]) if basis is None else basis
    n = 6 * basis.shape[-1]

    def values(thetas):
        c = fields_from_parameters(thetas, basis, formulation)
        if formulation == 'pairwise+higham':
            c = higham_projection(c, tol=1e-12, maxiter=500)
        error = np.sum(abs(c - target[None]) ** 2, axis=(-2, -1))
        if objective == 'norm':
            error = np.sqrt(error)
        return np.sum(error * w[None], axis=(-2, -1))

    def value_and_gradient(theta):
        perturb = np.eye(n) * step
        v = values(np.concatenate([theta[None], theta[None] + perturb, theta[None] - perturb]))
        return float(v[0]), (v[1:n + 1] - v[n + 1:]) / (2. * step)

    x0 = np.zeros(n) if theta0 is None else np.asarray(theta0, float)
    result = minimize(value_and_gradient, x0, jac=True, method='BFGS', options=dict(maxiter=maxiter, gtol=gtol))
    fitted = fields_from_parameters(result.x, basis, formulation)[0]
    if formulation == 'pairwise+higham':
        fitted = higham_projection(fitted, tol=1e-12, maxiter=500)
    return result.x, fitted, result
