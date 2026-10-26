"""Reading PEER NGA-West2 AT2 files and preparing three-component records."""
import re

import numpy as np
from scipy import signal


def read_at2(path):
    """Acceleration values (g) and time step of one PEER AT2 file."""
    with open(path, encoding='utf-8-sig') as handle:
        lines = handle.read().splitlines()
    for i, line in enumerate(lines):
        if 'NPTS' in line.upper() and 'DT' in line.upper():
            count = re.search(r'NPTS\s*=\s*([0-9]+)', line, re.I)
            interval = re.search(r'DT\s*=\s*([0-9.Ee+\-]+)', line, re.I)
            if count is None or interval is None:
                raise ValueError('invalid NPTS/DT header in ' + str(path))
            npts, dt = int(count.group(1)), float(interval.group(1))
            values = np.array(' '.join(lines[i + 1:]).split(), dtype=float)
            if npts <= 0 or len(values) != npts:
                raise ValueError('NPTS does not match the sample count in ' + str(path))
            if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(values)):
                raise ValueError('non-finite samples or invalid DT in ' + str(path))
            return values, dt
    raise ValueError('no NPTS/DT header in ' + str(path))


def read_record(h1_file, h2_file, v_file):
    """Three components (n_samples, 3) in the order vertical, first horizontal, second horizontal, and dt."""
    (h1, dt1), (h2, dt2), (v, dt3) = (read_at2(f) for f in (h1_file, h2_file, v_file))
    if max(dt1, dt2, dt3) - min(dt1, dt2, dt3) > 1e-8:
        raise ValueError('the three components have different time steps')
    n = min(len(h1), len(h2), len(v))
    return np.column_stack([v[:n], h1[:n], h2[:n]]), dt1


def preprocess(acc, dt, dt_target=.01, band=(.1, 12.), order=4, taper=.02):
    """Linear detrend, resampling to dt_target, cosine taper and zero-phase Butterworth band-pass."""
    acc = signal.detrend(acc, axis=0, type='linear')
    if dt_target is not None and abs(dt - dt_target) > 1e-12:
        up, down = int(round(1000 / dt_target)), int(round(1000 / dt))
        g = np.gcd(up, down)
        acc, dt = signal.resample_poly(acc, up // g, down // g, axis=0), dt_target
    m = int(np.floor(taper * len(acc)))
    window = np.ones(len(acc))
    if m > 0:
        ramp = .5 * (1 - np.cos(np.pi * (np.arange(m) + 1) / m))
        window[:m], window[-m:] = ramp, ramp[::-1]
    sos = signal.butter(order, [band[0] / (.5 / dt), band[1] / (.5 / dt)], btype='bandpass', output='sos')
    return signal.sosfiltfilt(sos, acc * window[:, None], axis=0), dt
