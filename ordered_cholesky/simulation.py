"""Simulation of three-component motions with a prescribed coherence through the ordered Cholesky factor."""
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d

from .coordinates import to_coordinates
from .estimation import istft


def ordered_factor(s, c):
    """Lower-triangular H with H H^H = D C D, where D = sqrt(diag s), written directly in the coordinates of C."""
    sigma = np.sqrt(np.maximum(np.diagonal(s, axis1=-2, axis2=-1).real, 0.))
    a, b, d = to_coordinates(c)
    r, q = np.sqrt(np.maximum(1. - abs(a) ** 2, 0.)), np.sqrt(np.maximum(1. - abs(b) ** 2, 0.))
    d = d * np.minimum(1., 1. / np.maximum(abs(d), 1e-300))
    h = np.zeros(np.shape(s), complex)
    h[..., 0, 0] = sigma[..., 0]
    h[..., 1, 0], h[..., 1, 1] = sigma[..., 1] * a.conj(), sigma[..., 1] * r
    h[..., 2, 0], h[..., 2, 1] = sigma[..., 2] * b.conj(), sigma[..., 2] * d.conj() * q
    h[..., 2, 2] = sigma[..., 2] * q * np.sqrt(np.maximum(1. - abs(d) ** 2, 0.))
    return h


def simulate(s, c, dt, n_samples=1, seed=None, nperseg=128, noverlap=96, nfft=256):
    """Motions (n_samples, n_points, 3) whose local spectral matrices follow D C D, with D = sqrt(diag s).

    s gives the marginal spectra and c the admissible coherence, both on the grid of estimation.stft().
    Each time-frequency atom gets an independent Gaussian driver: complex in the interior bins and real
    at the zero and Nyquist frequencies, where the covariance is made real.
    """
    c = np.array(c, complex)
    endpoints = [0, -1] if nfft % 2 == 0 else [0]
    c[endpoints] = c[endpoints].real
    h = ordered_factor(s, c)
    h[endpoints] = h[endpoints].real
    nf, nt = h.shape[:2]
    rng = np.random.default_rng(seed)
    scale = np.sqrt(nfft / (nperseg - noverlap))
    out = []
    for _ in range(n_samples):
        g = (rng.normal(size=(nf, nt, 3)) + 1j * rng.normal(size=(nf, nt, 3))) / np.sqrt(2.)
        g[endpoints] = rng.normal(size=(len(endpoints), nt, 3))
        coeff = np.einsum('ftij,ftj->ift', h, g, optimize=True) * scale
        out.append(istft(np.moveaxis(coeff, 0, -1), dt, nperseg, noverlap, nfft))
    return np.stack(out)


def realized_target(s, n_points, n_freq=5, n_time=5, nperseg=128, noverlap=96, nfft=256):
    """Expected local spectral matrix that the estimator returns for motions simulated from s.

    Synthesis by overlap-add and re-analysis do not return s itself: each atom leaks into its neighbours.
    The result is computed exactly from the two operators (with the same smoothing as the estimator), so the
    pooled spectral estimate can be compared with this expectation. Normalize the pooled estimate to compare
    coherence; the mean of separately normalized coherence estimates need not equal coherence of this target.
    nperseg must be even. n_points is the length of the simulated motions,
    (n_time - 1) * (nperseg - noverlap).
    """
    s = np.array(s, dtype=complex, copy=True)
    # An odd FFT has no Nyquist bin; its final bin remains complex.
    endpoints = [0, -1] if nfft % 2 == 0 else [0]
    s[endpoints] = s[endpoints].real
    if nperseg % 2:
        raise ValueError('realized_target requires an even nperseg')
    hop = nperseg - noverlap
    nf, nt = s.shape[:2]
    if nf != nfft // 2 + 1 or n_points != (nt - 1) * hop:
        raise ValueError('s and n_points do not match the STFT settings')
    window = signal.get_window('hann', nperseg)
    denominator = np.zeros(n_points)
    for q in range(nt):
        n = q * hop - nperseg // 2 + np.arange(nperseg)
        valid = (n >= 0) & (n < n_points)
        denominator[n[valid]] += window[valid] ** 2
    f, k = np.arange(nf)[:, None], np.arange(nf)[None, :]
    difference, summation = (f - k) % nfft, (f + k) % nfft
    result, cache = np.zeros_like(s), {}
    for t in range(nt):
        v = np.arange(nperseg)
        n = t * hop - nperseg // 2 + v
        for q in range(max(0, t - (nperseg - 1) // hop), min(nt, t + (nperseg - 1) // hop + 1)):
            u = n - q * hop + nperseg // 2
            valid = (n >= 0) & (n < n_points) & (u >= 0) & (u < nperseg)
            product = np.zeros(nfft)
            product[v[valid]] = window[v[valid]] * window[u[valid]] / denominator[n[valid]]
            key = product.tobytes()
            if key not in cache:
                power = abs(np.fft.fft(product)) ** 2 / (nfft * hop)
                plus, minus = power[difference], power[summation].copy()
                minus[:, endpoints] = 0.       # real drivers have no mirrored term
                cache[key] = (plus, minus)
            plus, minus = cache[key]
            result[:, t] += ((plus @ s[:, q].reshape(nf, -1)) + (minus @ s[:, q].conj().reshape(nf, -1))).reshape(nf, 3, 3)
    for axis, width in [(0, n_freq), (1, n_time)]:
        kernel = np.ones(width) / width
        result = convolve1d(result.real, kernel, axis=axis, mode='nearest') + 1j * convolve1d(result.imag, kernel, axis=axis, mode='nearest')
    return result
