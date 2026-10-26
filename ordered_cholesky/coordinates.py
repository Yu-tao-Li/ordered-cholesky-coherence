"""Ordered Cholesky coordinates of 3x3 coherence matrices.

alpha = c12, beta = c13 and delta = (c23 - conj(alpha) beta) / sqrt((1 - |alpha|^2)(1 - |beta|^2)).
Every |alpha|, |beta|, |delta| <= 1 gives an admissible matrix, and every admissible matrix has such coordinates.
"""
import numpy as np

EVERY_PHASE, SOME_PHASES, NO_PHASE = 0, 1, 2


def assemble(c12, c13, c23):
    """Hermitian matrices with unit diagonal from the three coherences."""
    c12, c13, c23 = np.broadcast_arrays(*(np.asarray(x, complex) for x in (c12, c13, c23)))
    c = np.zeros(c12.shape + (3, 3), complex)
    c[..., np.arange(3), np.arange(3)] = 1.
    for (i, j), value in zip([(0, 1), (0, 2), (1, 2)], [c12, c13, c23]):
        c[..., i, j] = value
        c[..., j, i] = value.conj()
    return c


def disc(alpha, beta):
    """Centre conj(alpha) beta and radius sqrt((1 - |alpha|^2)(1 - |beta|^2)) of the admissible values of c23."""
    radius = np.sqrt(np.maximum(1. - abs(alpha) ** 2, 0.) * np.maximum(1. - abs(beta) ** 2, 0.))
    return np.conj(alpha) * beta, radius


def to_coordinates(c):
    """Ordered coordinates (alpha, beta, delta); delta is 0 where |alpha| = 1 or |beta| = 1."""
    alpha, beta = c[..., 0, 1], c[..., 0, 2]
    centre, radius = disc(alpha, beta)
    delta = np.divide(c[..., 1, 2] - centre, radius, out=np.zeros_like(alpha), where=radius > 1e-12)
    return alpha, beta, delta


def from_coordinates(alpha, beta, delta):
    """Coherence matrices from ordered coordinates; admissible whenever |alpha|, |beta|, |delta| <= 1."""
    alpha, beta, delta = (np.asarray(x, complex) for x in (alpha, beta, delta))
    centre, radius = disc(alpha, beta)
    return assemble(alpha, beta, centre + delta * radius)


def min_eigenvalue(c):
    """Smallest eigenvalue of each matrix; a matrix is admissible when it is >= 0."""
    return np.linalg.eigvalsh(c)[..., 0]


def phase_class(c):
    """Whether the magnitudes |c12|, |c13|, |c23| are admissible for every phase, some phases or no phase.

    Returns EVERY_PHASE (0), SOME_PHASES (1) or NO_PHASE (2) for each matrix. Pairwise models that set the
    magnitudes and phases separately can only be trusted in the first class.
    """
    a, b, h = abs(c[..., 0, 1]), abs(c[..., 0, 2]), abs(c[..., 1, 2])
    radius = np.sqrt(np.maximum(1. - a * a, 0.) * np.maximum(1. - b * b, 0.))
    out = np.full(a.shape, SOME_PHASES)
    out[h + a * b <= radius + 1e-12] = EVERY_PHASE
    out[abs(h - a * b) > radius + 1e-12] = NO_PHASE
    return out
