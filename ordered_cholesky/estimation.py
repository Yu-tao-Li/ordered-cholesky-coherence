"""Time-frequency estimate of the local spectral matrix and the coherence of three-component motions.

Arrays of matrices have shape (n_freq, n_time, 3, 3); component 1 is the vertical one.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import convolve1d


def stft(acc, dt, nperseg=128, noverlap=96, nfft=256):
    """STFT (Hann window) of the three components of acc (n_samples, 3).

    Returns freqs, times and the coefficients Z with shape (n_freq, n_time, 3).
    """
    coeffs = []
    for ch in range(3):
        f, t, z = signal.stft(acc[:, ch], fs=1. / dt, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                              boundary='zeros', padded=True, return_onesided=True)
        coeffs.append(z)
    return f, t, np.stack(coeffs, axis=-1)


def istft(z, dt, nperseg=128, noverlap=96, nfft=256):
    """Inverse of stft(): three-component time series (n_samples, 3) from coefficients (n_freq, n_time, 3)."""
    x = [signal.istft(z[..., i], fs=1. / dt, window='hann', nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                      input_onesided=True, boundary=True)[1] for i in range(3)]
    n = min(len(v) for v in x)
    return np.column_stack([v[:n] for v in x])


def local_spectra(z, n_freq=5, n_time=5):
    """Local spectral matrix: Z Z^H averaged over n_freq frequency bins and n_time frames.

    One common set of positive weights for all entries keeps every matrix positive semidefinite.
    """
    s = z[..., :, None] * z[..., None, :].conj()
    for axis, width in [(0, n_freq), (1, n_time)]:
        kernel = np.ones(width) / width
        s = convolve1d(s.real, kernel, axis=axis, mode='nearest') + 1j * convolve1d(s.imag, kernel, axis=axis, mode='nearest')
    return s


def coherence(s):
    """Coherence matrices from local spectral matrices, normalized exactly by their diagonal.

    The normalization is a congruence, so an admissible spectral matrix gives an admissible coherence matrix.
    """
    d = np.maximum(np.diagonal(s, axis1=-2, axis2=-1).real, 0.)
    den = np.sqrt(d[..., :, None]) * np.sqrt(d[..., None, :])
    c = np.divide(s, den, out=np.zeros_like(s), where=den > 0.)
    c = .5 * (c + np.swapaxes(c.conj(), -2, -1))
    c[..., np.arange(3), np.arange(3)] = 1.
    return c


def energy_weights(s, freqs=None, band=None):
    """Weights proportional to the total local energy (trace of s), optionally kept to a frequency band, summing to 1."""
    w = np.maximum(np.trace(s, axis1=-2, axis2=-1).real, 0.)
    if band is not None:
        w = np.where(((freqs >= band[0]) & (freqs <= band[1]))[:, None], w, 0.)
    return w / max(w.sum(), 1e-300)
