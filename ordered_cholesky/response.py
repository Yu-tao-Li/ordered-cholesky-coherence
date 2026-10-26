"""Linear single-degree-of-freedom response and response spectra of three-component motions."""
import numpy as np
from scipy import signal


def sdof_response(acc, dt, period, damping=.05):
    """Pseudo-acceleration history of a linear oscillator (bilinear-transform filter) for each column of acc."""
    omega = 2. * np.pi / period
    b, a = signal.bilinear([-omega * omega], [1., 2. * damping * omega, omega * omega], fs=1. / dt)
    initial = np.stack([-b[0] * acc[0], (b[0] - b[1]) * acc[0]])
    return signal.lfilter(b, a, acc, axis=0, zi=initial)[0]


def response_spectra(acc, dt, periods, damping=.05):
    """Peak pseudo-acceleration per period: vertical, first and second horizontal, geometric mean of the
    horizontals, RotD50 and RotD100 (rotations 0-179 degrees). Shape (n_periods, 6), units of acc."""
    angles = np.deg2rad(np.arange(180.))
    rotations = np.stack([np.cos(angles), np.sin(angles)])
    rows = []
    for period in periods:
        response = sdof_response(acc, dt, float(period), damping)
        peaks = np.max(abs(response), axis=0)
        directional = np.max(abs(response[:, 1:] @ rotations), axis=0)
        rows.append([*peaks, np.sqrt(peaks[1] * peaks[2]), np.quantile(directional, .5), directional.max()])
    return np.asarray(rows)
