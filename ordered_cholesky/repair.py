"""Repairs that return coherence matrices to the admissible set.

coordinate_repair keeps c12 and c13 and moves only c23; the other three are the usual full-matrix repairs,
which also change the vertical-horizontal coherences.
"""
import numpy as np

from .coordinates import assemble, disc


def coordinate_repair(c):
    """Nearest admissible matrix with the same c12 and c13: c23 moves radially onto the admissible disc."""
    a, b, h = c[..., 0, 1], c[..., 0, 2], c[..., 1, 2]
    centre, radius = disc(a, b)
    residual = h - centre
    return assemble(a, b, centre + residual * np.minimum(1., radius / np.maximum(abs(residual), 1e-300)))


def higham_projection(c, tol=1e-12, maxiter=2000):
    """Nearest correlation matrix in the Frobenius norm (alternating projections with Dykstra's correction)."""
    shape = c.shape
    y = np.array(c.reshape(-1, 3, 3), copy=True)
    y = .5 * (y + y.swapaxes(-2, -1).conj())
    y[:, np.arange(3), np.arange(3)] = 1.
    active = np.flatnonzero(np.linalg.eigvalsh(y)[:, 0] < -tol)
    ds = np.zeros_like(y)
    for _ in range(maxiter):
        if not len(active):
            break
        r = y[active] - ds[active]
        v, u = np.linalg.eigh(.5 * (r + r.swapaxes(-2, -1).conj()))
        x = (u * np.maximum(v, 0.)[:, None, :]) @ u.swapaxes(-2, -1).conj()
        ds[active] = x - r
        new = .5 * (x + x.swapaxes(-2, -1).conj())
        new[:, np.arange(3), np.arange(3)] = 1.
        change = np.linalg.norm(new - y[active], axis=(-2, -1))
        y[active] = new
        active = active[change >= tol]
    if len(active) and np.linalg.eigvalsh(y[active])[:, 0].min() < -1e-8:
        raise RuntimeError('Higham projection did not converge')
    return y.reshape(shape)


def shrink_to_identity(c, epsilon=1e-10):
    """Smallest shrinkage toward the identity that lifts the smallest eigenvalue to epsilon."""
    eigen = np.linalg.eigvalsh(c)[..., 0]
    eta = np.minimum(np.maximum(0., (epsilon - eigen) / np.maximum(1. - eigen, 1e-300)), 1.)
    return (1. - eta[..., None, None]) * c + eta[..., None, None] * np.eye(3)


def eigenvalue_clipping(c, epsilon=1e-10):
    """Raise negative eigenvalues to epsilon, then restore the unit diagonal."""
    v, u = np.linalg.eigh(c)
    p = (u * np.maximum(v, epsilon)[..., None, :]) @ u.swapaxes(-1, -2).conj()
    d = np.sqrt(np.diagonal(p, axis1=-2, axis2=-1).real)
    return p / d[..., :, None] / d[..., None, :]
