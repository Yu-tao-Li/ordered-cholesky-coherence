"""Unit tests on synthetic data; run with `python -m pytest tests`."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ordered_cholesky as oc

RNG = np.random.default_rng(7)


def random_disc(n, radius=1.):
    return radius * np.sqrt(RNG.uniform(size=n)) * np.exp(2j * np.pi * RNG.uniform(size=n))


def test_every_coordinate_gives_an_admissible_matrix():
    c = oc.from_coordinates(random_disc(5000), random_disc(5000), random_disc(5000))
    assert oc.min_eigenvalue(c).min() > -1e-12


def test_round_trip_through_the_coordinates():
    alpha, beta, delta = random_disc(2000, .99), random_disc(2000, .99), random_disc(2000)
    a, b, d = oc.to_coordinates(oc.from_coordinates(alpha, beta, delta))
    assert max(abs(a - alpha).max(), abs(b - beta).max(), abs(d - delta).max()) < 1e-10


def test_estimate_is_admissible_and_free_of_units():
    acc = RNG.normal(size=(3000, 3)) @ np.array([[1., .5, .2], [0., 1., .4], [0., 0., 1.]])
    c = oc.coherence(oc.local_spectra(oc.stft(acc, .01)[2]))
    scaled = oc.coherence(oc.local_spectra(oc.stft(9.81 * acc, .01)[2]))
    assert oc.min_eigenvalue(c).min() > -1e-12
    assert abs(c - scaled).max() < 1e-12


def test_coordinate_repair_keeps_c12_and_c13():
    bad = oc.assemble(np.full(500, .8 + 0j), .75 * np.exp(1j * RNG.uniform(0, 2 * np.pi, 500)), random_disc(500))
    repaired = oc.coordinate_repair(bad)
    assert oc.min_eigenvalue(repaired).min() > -1e-12
    assert abs(repaired[..., 0, 1:] - bad[..., 0, 1:]).max() == 0.
    good = oc.from_coordinates(random_disc(500, .9), random_disc(500, .9), random_disc(500, .9))
    assert abs(oc.coordinate_repair(good) - good).max() < 1e-12


@pytest.mark.parametrize('repair', [oc.higham_projection, oc.shrink_to_identity, oc.eigenvalue_clipping])
def test_full_matrix_repairs_are_admissible_correlation_matrices(repair):
    bad = oc.assemble(random_disc(300, .95), random_disc(300, .95), random_disc(300, .95))
    r = repair(bad)
    assert oc.min_eigenvalue(r).min() > -1e-8
    assert abs(np.diagonal(r, axis1=-2, axis2=-1) - 1.).max() < 1e-10


def test_phase_classes():
    alpha, beta = np.full(3, .3 + 0j), np.full(3, .4 + 0j)
    centre, radius = oc.disc(alpha[0], beta[0])
    ab = abs(centre)
    magnitudes = np.array([.5 * (radius - ab), radius, .5 * (radius + ab + 1.)])
    classes = oc.phase_class(oc.assemble(alpha, beta, magnitudes + 0j))
    assert list(classes) == [oc.EVERY_PHASE, oc.SOME_PHASES, oc.NO_PHASE]
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 361))
    for h, k in zip(magnitudes, classes):
        eigen = oc.min_eigenvalue(oc.assemble(np.full(361, alpha[0]), np.full(361, beta[0]), h * phases))
        if k == oc.EVERY_PHASE:
            assert eigen.min() > -1e-12
        if k == oc.NO_PHASE:
            assert eigen.max() < 0.


def test_ordered_factor_reproduces_the_spectral_matrix():
    c = oc.from_coordinates(random_disc(400, .95), random_disc(400, .95), random_disc(400))
    d = RNG.uniform(.5, 2., size=(400, 3))
    s = c * d[:, :, None] * d[:, None, :]
    h = oc.ordered_factor(s, c)
    assert abs(h @ h.swapaxes(-2, -1).conj() - s).max() < 1e-12
    assert abs(np.triu(h, 1)).max() == 0.


def test_correction_removes_the_floor_and_keeps_admissibility():
    assert abs(oc.correct_magnitude(np.array([np.sqrt(.04) * 1j]), .04)[0]) < 1e-6
    c = oc.from_coordinates(random_disc(1000, .9), random_disc(1000, .9), random_disc(1000, .9))
    floor = dict(c12=.05, c13=.05, c23=.05, delta=.08)
    assert oc.min_eigenvalue(oc.correct_coherence(c, floor)).min() > -1e-12


def test_simulation_follows_the_realized_target():
    nf, nt, dt = 129, 60, .01
    f = np.linspace(0, 50, nf)[:, None, None, None]
    d = np.ones((nf, nt, 3)) * np.array([1., 2., 1.5])
    c = np.broadcast_to(oc.from_coordinates(np.array(.6 + .2j), np.array(.5), np.array(-.4j)), (nf, nt, 3, 3))
    s = c * d[..., :, None] * d[..., None, :] * np.exp(-f / 20.)
    motions = oc.simulate(s, c, dt, n_samples=60, seed=1)
    pooled = np.mean([oc.local_spectra(oc.stft(m, dt)[2]) for m in motions], axis=0)
    realized = oc.realized_target(s, motions.shape[1])
    band = slice(5, 100)
    error = abs(oc.coherence(pooled) - oc.coherence(realized))[band, 5:-5]
    assert error.mean() < .05


def test_fit_in_coordinates_is_admissible_and_recovers_a_smooth_field():
    basis = oc.bilinear_basis(6, 6)
    theta = RNG.normal(scale=.4, size=24)
    target = oc.fields_from_parameters(theta, basis)[0]
    _, fitted, result = oc.fit_coherence(target, np.ones((6, 6)), objective='squared')
    assert oc.min_eigenvalue(fitted).min() > -1e-12
    assert result.fun < 1e-6


def test_coordinates_broadcast_scalars_and_fields():
    beta = np.array([.2, .3j, -.4])
    c = oc.from_coordinates(.5, beta, .1j)
    assert c.shape == (3, 3, 3)
    assert np.allclose(c[:, 0, 1], .5)
    assert np.allclose(c[:, 0, 2], beta)
    assert oc.min_eigenvalue(c).min() > 0.


def test_magnitude_correction_accepts_real_integer_and_scalar_inputs():
    assert np.allclose(oc.correct_magnitude(np.array([0, 1]), .04), [0., 1.])
    assert np.allclose(oc.correct_magnitude(.5, .04), np.sqrt(.21 / .96))


@pytest.mark.parametrize('kwargs', [{'formulation': 'coordinate'}, {'objective': 'square'}])
def test_fit_rejects_unknown_options(kwargs):
    with pytest.raises(ValueError):
        oc.fit_coherence(np.tile(np.eye(3), (2, 2, 1, 1)), np.ones((2, 2)), **kwargs)


def test_at2_rejects_truncated_record(tmp_path):
    path = tmp_path / 'truncated.AT2'
    path.write_text('synthetic\nNPTS= 4, DT= .01 SEC\n1 2 3\n', encoding='utf-8')
    with pytest.raises(ValueError, match='NPTS'):
        oc.read_at2(path)


def test_at2_reader_and_component_order(tmp_path):
    paths = []
    for name, values in [('h1', [1., 2., 3.]), ('h2', [4., 5., 6.]), ('v', [7., 8., 9.])]:
        path = tmp_path / (name + '.AT2')
        path.write_text('synthetic\nNPTS= 3, DT= .005 SEC\n' + ' '.join(map(str, values)), encoding='utf-8')
        paths.append(path)
    acc, dt = oc.read_record(*paths)
    assert dt == .005
    assert np.array_equal(acc, [[7., 1., 4.], [8., 2., 5.], [9., 3., 6.]])


@pytest.mark.parametrize('nfft', [16, 17])
def test_realized_target_matches_exact_driver_covariance_at_endpoints(nfft):
    # Sum responses to a basis of independent unit-variance real drivers.
    # This is an exact second-moment calculation, with no Monte Carlo tolerance.
    nf, nt, segment, overlap = nfft // 2 + 1, 5, 8, 4
    endpoints = [0, nf - 1] if nfft % 2 == 0 else [0]
    c = np.broadcast_to(oc.from_coordinates(.6 + .2j, .4j, -.3j), (nf, nt, 3, 3)).copy()
    s = c * np.linspace(.5, 2., nf)[:, None, None, None]
    real_c = c.copy()
    real_c[endpoints] = real_c[endpoints].real
    h = oc.ordered_factor(s, real_c)
    pooled = np.zeros_like(s)
    for f in range(nf):
        phases = [1.] if f in endpoints else [1. / np.sqrt(2.), 1j / np.sqrt(2.)]
        for t in range(nt):
            for channel in range(3):
                for phase in phases:
                    z = np.zeros((nf, nt, 3), complex)
                    z[f, t] = h[f, t, :, channel] * phase * np.sqrt(nfft / (segment - overlap))
                    acc = oc.istft(z, .01, segment, overlap, nfft)
                    pooled += oc.local_spectra(oc.stft(acc, .01, segment, overlap, nfft)[2], 1, 1)
    target = oc.realized_target(s, len(acc), 1, 1, segment, overlap, nfft)
    assert np.allclose(target, pooled, rtol=1e-12, atol=1e-12)


def test_odd_fft_surrogate_has_no_nyquist_bin():
    z = np.ones((9, 5, 3), complex)
    surrogate = oc.phase_surrogate(z, np.random.default_rng(1), nfft=17)
    assert np.allclose(abs(surrogate), abs(z))
    assert np.all(surrogate[0].imag == 0.)
    assert np.any(abs(surrogate[-1].imag) > .1)


def test_odd_fft_simulation_keeps_complex_final_bin(monkeypatch):
    from ordered_cholesky import simulation
    captured = []
    monkeypatch.setattr(simulation, 'istft', lambda z, *args: captured.append(z.copy()) or np.zeros((16, 3)))
    s = np.tile(np.eye(3, dtype=complex), (9, 5, 1, 1))
    oc.simulate(s, s, .01, seed=1, nperseg=8, noverlap=4, nfft=17)
    assert np.all(captured[0][0].imag == 0.)
    assert np.any(abs(captured[0][-1].imag) > .1)
