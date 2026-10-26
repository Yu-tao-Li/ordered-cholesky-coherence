"""Fit a smooth coherence field to a record: ordered coordinates against pairwise fields repaired by Higham projection.

usage: python examples/02_fit_in_coordinates.py <H1.AT2> <H2.AT2> <V.AT2>
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ordered_cholesky as oc


def main(h1, h2, v):
    acc, dt = oc.preprocess(*oc.read_record(h1, h2, v))
    freqs, times, z = oc.stft(acc, dt)
    s = oc.local_spectra(z)
    c = oc.coherence(s)
    fi, ti = oc.grid_indices(freqs, times, n_freq=6, n_time=6, band=(.5, 12.))
    target = c[np.ix_(fi, ti)]
    weights = np.maximum(np.trace(s, axis1=-2, axis2=-1).real, 0.)[np.ix_(fi, ti)]
    print(f'target: {target.shape[0]} x {target.shape[1]} bins between {freqs[fi[0]]:.2f} and {freqs[fi[-1]]:.2f} Hz')
    for formulation in ['coordinates', 'pairwise+higham']:
        start = time.process_time()
        theta, fitted, result = oc.fit_coherence(target, weights, formulation=formulation)
        seconds = time.process_time() - start
        print(f'{formulation:16s} weighted error {result.fun:.4f}, {result.nit} iterations, {seconds:.2f} s CPU, '
              f'smallest eigenvalue {oc.min_eigenvalue(fitted).min():.1e}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
