"""Coherence of one record, its ordered coordinates, bias correction and the four repairs.

usage: python examples/01_coherence_and_coordinates.py <H1.AT2> <H2.AT2> <V.AT2>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ordered_cholesky as oc


def invalid_share(c, w):
    """Energy share of the bins whose matrix is not admissible."""
    return float(np.sum(w * (oc.min_eigenvalue(c) < -1e-8)))


def main(h1, h2, v):
    acc, dt = oc.preprocess(*oc.read_record(h1, h2, v))
    freqs, times, z = oc.stft(acc, dt)
    s = oc.local_spectra(z)
    c = oc.coherence(s)
    w = oc.energy_weights(s, freqs, band=(.5, 12.))
    alpha, beta, delta = oc.to_coordinates(c)
    print(f'{len(freqs)} frequencies x {len(times)} frames, dt = {dt} s')
    print(f'estimate: smallest eigenvalue {oc.min_eigenvalue(c).min():.1e}; '
          f'round trip through the coordinates {abs(oc.from_coordinates(alpha, beta, delta) - c).max():.1e}')

    floor = oc.surrogate_floor(acc, dt, n_surrogates=20, seed=0)
    print('surrogate floor (mean squared magnitude):', {k: round(v, 4) for k, v in floor.items()})
    pairwise = oc.correct_coherence(c, floor, method='pairwise')
    corrected = oc.correct_coherence(c, floor, method='coordinates')
    print(f'inadmissible energy after correction: pairwise {invalid_share(pairwise, w):.2%}, '
          f'coordinates {invalid_share(corrected, w):.2%}')
    classes = oc.phase_class(pairwise)
    print('energy by phase class of the pairwise correction: ' + ', '.join(
        f'{name} {np.sum(w * (classes == k)):.1%}' for name, k in
        [('every phase', oc.EVERY_PHASE), ('some phases', oc.SOME_PHASES), ('no phase', oc.NO_PHASE)]))

    print('repairs of the pairwise correction (largest change of c12 and c13, inadmissible energy after):')
    for name, repair in [('coordinate repair', oc.coordinate_repair), ('Higham projection', oc.higham_projection),
                         ('shrinkage', oc.shrink_to_identity), ('eigenvalue clipping', oc.eigenvalue_clipping)]:
        r = repair(pairwise)
        moved = np.max(abs(r[..., 0, 1:] - pairwise[..., 0, 1:]))
        print(f'  {name:20s} {moved:.3f}   {invalid_share(r, w):.2%}')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
