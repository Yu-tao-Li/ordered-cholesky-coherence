"""Bias correction of estimated coherence with a surrogate floor.

A smoothed estimate of coherence is biased upward: motions without any coherence still give |c| > 0.
Phase-randomized surrogates of the record measure this floor, and the correction removes it from the
squared magnitudes. Correcting alpha, beta and delta keeps every matrix admissible; correcting c12, c13
and c23 separately does not.
"""
import numpy as np

from .coordinates import assemble, from_coordinates, to_coordinates
from .estimation import coherence, energy_weights, istft, local_spectra, stft

PAIRS = [('c12', 0, 1), ('c13', 0, 2), ('c23', 1, 2)]


def phase_surrogate(z, rng, nfft=None):
    """STFT coefficients with unchanged magnitudes and independent phases; real signs at DC and Nyquist.

    Pass nfft for an odd FFT size; by default an even size of 2 * (n_freq - 1) is assumed.
    """
    nfft = 2 * (z.shape[0] - 1) if nfft is None else nfft
    if z.shape[0] != nfft // 2 + 1:
        raise ValueError('z does not match nfft')
    zc = np.moveaxis(z, -1, 0)
    phase = rng.uniform(-np.pi, np.pi, size=zc.shape)
    out = abs(zc) * np.exp(1j * phase)
    out[:, 0, :] = abs(zc[:, 0, :]) * rng.choice([-1., 1.], size=out[:, 0, :].shape)
    if nfft % 2 == 0:
        out[:, -1, :] = abs(zc[:, -1, :]) * rng.choice([-1., 1.], size=out[:, -1, :].shape)
    return np.moveaxis(out, 0, -1)


def surrogate_floor(acc, dt, n_surrogates=20, band=(.5, 12.), seed=None, n_freq=5, n_time=5,
                    nperseg=128, noverlap=96, nfft=256):
    """Energy-weighted mean of |c12|^2, |c13|^2, |c23|^2 and |delta|^2 over phase-randomized surrogates.

    Independent phases preserve the input STFT magnitudes. Synthesis and re-analysis then define the
    surrogate reference distribution; the returned values estimate its mean squared coherence.
    """
    rng = np.random.default_rng(seed)
    freqs, _, z = stft(acc, dt, nperseg, noverlap, nfft)
    scale = np.sqrt(nfft / (nperseg - noverlap))
    values = {name: [] for name in ['c12', 'c13', 'c23', 'delta']}
    for _ in range(n_surrogates):
        x = istft(phase_surrogate(z, rng, nfft), dt, nperseg, noverlap, nfft) * scale
        s = local_spectra(stft(x, dt, nperseg, noverlap, nfft)[2], n_freq, n_time)
        c, w = coherence(s), energy_weights(s, freqs, band)
        for name, v in [(name, c[..., i, j]) for name, i, j in PAIRS] + [('delta', to_coordinates(c)[2])]:
            magnitude = abs(v)
            values[name].append(float(np.sum(w * magnitude * magnitude)))
    return {name: float(np.mean(v)) for name, v in values.items()}


def correct_magnitude(z, floor):
    """Remove a floor from the squared magnitude and keep the phase: |z|^2 -> (|z|^2 - floor) / (1 - floor)."""
    z = np.asarray(z, dtype=np.result_type(z, np.float64))
    magnitude = abs(z)
    value = np.sqrt(np.clip((magnitude * magnitude - floor) / max(1. - floor, 1e-12), 0., 1.))
    return np.divide(z, magnitude, out=np.zeros_like(z), where=magnitude > 0.) * value


def correct_coherence(c, floor, method='coordinates'):
    """Bias-corrected coherence matrices.

    method='coordinates' corrects alpha, beta and delta with the floors of c12, c13 and delta, and the result is
    admissible. method='pairwise' corrects c12, c13 and c23 one at a time, for comparison; it can leave the
    admissible set.
    """
    if method == 'coordinates':
        alpha, beta, delta = to_coordinates(c)
        return from_coordinates(correct_magnitude(alpha, floor['c12']), correct_magnitude(beta, floor['c13']),
                                correct_magnitude(delta, floor['delta']))
    if method == 'pairwise':
        return assemble(*(correct_magnitude(c[..., i, j], floor[name]) for name, i, j in PAIRS))
    raise ValueError("method must be 'coordinates' or 'pairwise'")
