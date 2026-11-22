import numpy as np
import pytest
import scipy.fft as sfft
from scipy.special import j1

from espi_sim.config import Camera, Optics
from espi_sim.optics import circular_pupil, image_field, make_grid, rough_surface, warp
from espi_sim.rng import stream

LAM = 632.8e-9


def _grid(f_number=8.0, oversample=None, shape=(64, 64)):
    return make_grid(Camera(shape=shape), Optics(f_number=f_number), LAM, oversample=oversample)


@pytest.mark.parametrize("f_number", [2.0, 4.0, 8.0, 16.0])
def test_auto_oversample_is_minimal_alias_free(f_number):
    g = _grid(f_number)
    assert g.cutoff_cycles_per_sample <= 0.25
    if g.oversample > 1:
        coarser = g.cutoff_cycles_per_sample * g.oversample / (g.oversample - 1)
        assert coarser > 0.25


def test_explicit_oversample_that_would_alias_is_rejected():
    with pytest.raises(ValueError, match="oversample"):
        _grid(f_number=2.0, oversample=1)


def test_pixel_centres_are_symmetric_and_pitch_spaced():
    g = _grid(shape=(4, 6))
    xi, eta = g.pixel_object_coords((4, 6))
    p_obj = 3.45e-6 / 0.1
    np.testing.assert_allclose(xi[0], (np.arange(6) - 2.5) * p_obj)
    np.testing.assert_allclose(eta[:, 0], (1.5 - np.arange(4)) * p_obj)  # y up


def _speckle(shape=(1024, 1024), f_number=8.0, seed=1):
    g = make_grid(Camera(shape=(8, 8)), Optics(f_number=f_number), LAM, oversample=2)
    rng = stream(seed, "test")
    p = circular_pupil(shape, g.cutoff_cycles_per_sample)
    return image_field(rough_surface(shape, rng), p), p, g


def test_fully_developed_speckle_statistics():
    e, p, _ = _speckle()
    i = np.abs(e) ** 2
    eta = p.mean()
    assert i.mean() == pytest.approx(eta, rel=0.03)
    assert i.std() / i.mean() == pytest.approx(1.0, abs=0.03)  # contrast of 1
    for t in (1.0, 2.0, 3.0):  # negative-exponential tail
        assert np.mean(i > t * i.mean()) == pytest.approx(np.exp(-t), rel=0.06)


def test_speckle_autocovariance_matches_airy_theory():
    e, _, g = _speckle()
    i = np.abs(e) ** 2
    d = i - i.mean()
    acov = sfft.ifft2(np.abs(sfft.fft2(d)) ** 2).real / d.size
    acov /= acov[0, 0]
    for lag in range(1, 7):
        v = 2 * np.pi * g.cutoff_cycles_per_sample * lag
        theory = (2 * j1(v) / v) ** 2
        assert acov[0, lag] == pytest.approx(theory, abs=0.03)


def test_warp_by_integer_shift_is_a_roll():
    f = rough_surface((32, 32), stream(0, "w"))
    out = warp(f, np.full((32, 32), 2.0), np.full((32, 32), -3.0))
    np.testing.assert_allclose(out, np.roll(f, (2, -3), axis=(0, 1)), atol=1e-10)


def test_fractional_warp_of_band_limited_field_matches_fourier_shift():
    e, _, _ = _speckle(shape=(256, 256))
    shift = 0.37
    out = warp(e, np.zeros(e.shape), np.full(e.shape, shift))
    fx = sfft.fftfreq(e.shape[1])
    exact = sfft.ifft(sfft.fft(e, axis=1) * np.exp(-2j * np.pi * fx * shift), axis=1)
    err = np.linalg.norm(out - exact) / np.linalg.norm(exact)
    assert err < 0.02
