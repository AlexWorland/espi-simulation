import numpy as np
import pytest
from scipy.special import j0

from espi_sim.optics import circular_pupil, image_field, rough_surface
from espi_sim.rng import stream
from espi_sim.time_average import cycle_averaged_fields, cycle_samples, exposure_average

N, CUT = 128, 0.2
PUPIL = circular_pupil((N, N), CUT)
U = rough_surface((N, N), stream(0, "ta"))
STATIC = image_field(U, PUPIL)


def test_cycle_samples():
    assert cycle_samples(0.0) == 32
    n = cycle_samples(40.2)
    assert n >= 57 and n % 4 == 0


def test_static_vibration_returns_static_field():
    mf, mi = cycle_averaged_fields(U, np.zeros((N, N), complex), PUPIL)
    scale = np.abs(STATIC).mean()
    np.testing.assert_allclose(mf, STATIC, atol=1e-9 * scale)
    np.testing.assert_allclose(mi, np.abs(STATIC) ** 2, atol=1e-9 * scale**2)


def test_uniform_vibration_scales_field_by_j0():
    mf, mi = cycle_averaged_fields(U, np.full((N, N), 2.0 + 0j), PUPIL)
    np.testing.assert_allclose(mf, j0(2.0) * STATIC, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(mi, np.abs(STATIC) ** 2, rtol=1e-6, atol=1e-9)


def test_oscillating_tilt_smears_speckle():
    cols = np.broadcast_to(np.arange(N, dtype=float), (N, N))
    b = 2 * np.pi * CUT * cols + 0j  # spectrum swings by one pupil radius
    _, mi = cycle_averaged_fields(U, b, PUPIL)
    static_i = np.abs(STATIC) ** 2
    assert static_i.std() / static_i.mean() > 0.9
    assert mi.std() / mi.mean() < 0.8


def test_exposure_average_of_linear_ramp():
    avg = exposure_average(lambda t: 3.0 * t, 0.0, 0.5)
    assert abs(avg) == pytest.approx(abs(np.sinc(1.5 / (2 * np.pi))), rel=1e-3)
    assert np.angle(avg) == pytest.approx(0.75, abs=1e-9)
