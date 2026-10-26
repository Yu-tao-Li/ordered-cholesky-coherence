"""Simulate motions with the coherence of a record and check them against the realized target and the record.

usage: python examples/03_simulation.py <H1.AT2> <H2.AT2> <V.AT2>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ordered_cholesky as oc

N_SAMPLES = 20


def main(h1, h2, v):
    acc, dt = oc.preprocess(*oc.read_record(h1, h2, v))
    freqs, times, z = oc.stft(acc, dt)
    s = oc.local_spectra(z)
    c = oc.coherence(s)
    motions = oc.simulate(s, c, dt, n_samples=N_SAMPLES, seed=0)
    print(f'{N_SAMPLES} motions of {motions.shape[1]} points, dt = {dt} s')

    # The pooled estimate of the simulations follows the realized target, not the declared coherence.
    pooled = np.mean([oc.local_spectra(oc.stft(m, dt)[2]) for m in motions], axis=0)
    realized = oc.coherence(oc.realized_target(s, motions.shape[1]))
    w = oc.energy_weights(s, freqs, band=(.5, 12.))
    for name, i, j in [('c12', 0, 1), ('c13', 0, 2), ('c23', 1, 2)]:
        to_realized = np.sqrt(np.sum(w * abs(oc.coherence(pooled)[..., i, j] - realized[..., i, j]) ** 2))
        to_declared = np.sqrt(np.sum(w * abs(oc.coherence(pooled)[..., i, j] - c[..., i, j]) ** 2))
        print(f'{name}: weighted RMS difference to the realized target {to_realized:.3f}, to the declared one {to_declared:.3f}')

    periods = np.array([.1, .2, .5, 1., 2.])
    recorded = oc.response_spectra(acc, dt, periods)
    simulated = np.median([oc.response_spectra(m, dt, periods) for m in motions], axis=0)
    print('median simulated / recorded response spectrum (vertical, hx, hy, RotD50):')
    for period, ratio in zip(periods, simulated / recorded):
        print(f'  T = {period:4.1f} s   ' + '  '.join(f'{x:.2f}' for x in ratio[[0, 1, 2, 4]]))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
