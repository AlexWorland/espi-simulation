import numpy as np
import pytest

from espi_sim.camera import expose, integrate_pixels
from espi_sim.config import Camera
from espi_sim.optics import Grid
from espi_sim.rng import stream


def _grid(nx=32, s=4, margin=4):
    return Grid(
        camera_shape=(2, nx),
        oversample=s,
        margin=(0, margin),
        pixel_pitch=1.0,
        magnification=1.0,
        cutoff_cycles_per_sample=0.1,
    )


@pytest.mark.parametrize("ff", [1.0, 0.5])
def test_uniform_light_collects_fill_factor_fraction(ff):
    g = _grid()
    out = integrate_pixels(np.full(g.shape, 1000.0), g, fill_factor=ff)
    assert out.shape == (2, 32)
    np.testing.assert_allclose(out, 1000.0 * ff)


@pytest.mark.parametrize("ff", [1.0, 0.25])
def test_pixel_aperture_mtf_and_centre_sampling(ff):
    g = _grid()
    s, f_px = g.oversample, 0.25  # cycles per pixel
    n_sim = g.shape[1]
    i = np.arange(n_sim)
    fringe = 1.0 + 0.5 * np.cos(2 * np.pi * (f_px / s) * i)
    intensity = np.broadcast_to(fringe, g.shape)
    out = integrate_pixels(intensity, g, fill_factor=ff)
    k = np.arange(32)
    centre = g.margin[1] * s + k * s + (s - 1) / 2
    expected = ff * (1 + 0.5 * np.sinc(np.sqrt(ff) * f_px) * np.cos(2 * np.pi * (f_px / s) * centre))
    np.testing.assert_allclose(out[0], expected, atol=1e-9)


def test_noiseless_exposure_is_quantised_electrons():
    cam = Camera(shot_noise=False, read_noise=0.0, gain=2.0, bit_depth=12)
    e = np.array([[0.0, 3.0, 1001.0]])
    adu = expose(e, cam, stream(0, "c"))
    assert adu.dtype == np.uint16
    np.testing.assert_array_equal(adu, [[0, 2, 500]])  # round-half-even of 1.5 and 500.5


def test_saturation_clips_at_full_well_and_adc_range():
    cam = Camera(shot_noise=False, read_noise=0.0, full_well=4095.0, bit_depth=12)
    adu = expose(np.array([[1e6]]), cam, stream(0, "c"))
    assert adu[0, 0] == 4095


def test_shot_and_read_noise_statistics():
    cam = Camera(read_noise=5.0, gain=1.0, full_well=1e5, bit_depth=16)
    adu = expose(np.full((256, 256), 1000.0), cam, stream(0, "c")).astype(float)
    assert adu.mean() == pytest.approx(1000.0, abs=1.0)
    assert adu.var() == pytest.approx(1000.0 + 25.0 + 1 / 12, rel=0.03)
