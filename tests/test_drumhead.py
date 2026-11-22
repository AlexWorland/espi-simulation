import numpy as np
import pytest
from scipy.special import j0

from espi_sim.drumhead import Drumhead, PointDrive, UniformPressure

A01 = 2.404826


def _polar(drum, f, level, r, theta_deg):
    t = np.radians(theta_deg)
    return drum.complex_amplitude(r * np.cos(t), r * np.sin(t), f, level)


def test_natural_frequency_ratios_follow_bessel_zeros():
    d = Drumhead()
    assert d.mode_frequency(1, 1) / d.f01 == pytest.approx(3.831706 / A01, rel=1e-6)
    assert d.mode_frequency(2, 1) / d.f01 == pytest.approx(5.135622 / A01, rel=1e-6)
    assert d.mode_frequency(0, 2) / d.f01 == pytest.approx(5.520078 / A01, rel=1e-6)


def test_response_vanishes_at_rim_and_outside():
    d = Drumhead()
    theta = np.linspace(0, 360, 13)
    inside = np.abs(_polar(d, 450.0, 1.0, 0.03 * np.ones_like(theta), theta)).max()
    for r in (d.radius, 1.2 * d.radius):
        w = _polar(d, 450.0, 1.0, np.full_like(theta, r), theta)
        assert np.abs(w).max() < 1e-9 * inside


def test_on_resonance_fundamental_shape_is_j0():
    d = Drumhead(q=500, drive=PointDrive(r=0.0))
    r = np.linspace(0, 0.999 * d.radius, 200)
    w = np.abs(_polar(d, d.f01, 1.0, r, np.zeros_like(r)))
    assert np.corrcoef(w, j0(A01 * r / d.radius))[0, 1] > 0.99


def test_off_centre_drive_at_f11_has_nodal_diameter():
    d = Drumhead(q=500, drive=PointDrive(r=0.04, theta_deg=30))
    f = d.mode_frequency(1, 1)
    r = np.linspace(0, 0.99 * d.radius, 100)
    along = np.abs(_polar(d, f, 1.0, r, np.full_like(r, 30.0))).max()
    across = np.abs(_polar(d, f, 1.0, r, np.full_like(r, 120.0))).max()
    assert across < 0.05 * along


def test_uniform_pressure_is_axisymmetric():
    d = Drumhead(drive=UniformPressure())
    r = np.linspace(0, 0.99 * d.radius, 50)
    ref = _polar(d, 1.3 * d.f01, 1.0, r, np.zeros_like(r))
    for th in (90.0, 200.0):
        np.testing.assert_allclose(
            _polar(d, 1.3 * d.f01, 1.0, r, np.full_like(r, th)),
            ref,
            rtol=1e-9,
            atol=1e-12 * np.abs(ref).max(),
        )


def test_modal_phase_lag_across_fundamental():
    d = Drumhead(q=50)

    def a01(f):
        return next(m.amplitude for m in d.modes(f) if (m.m, m.n) == (0, 1))

    assert np.angle(a01(0.5 * d.f01)) == pytest.approx(0.0, abs=0.05)
    assert np.angle(a01(d.f01)) == pytest.approx(-np.pi / 2, abs=1e-3)
    assert abs(np.angle(a01(2 * d.f01))) > 0.95 * np.pi


def test_calibrate_sets_peak_amplitude():
    d = Drumhead()
    level = d.calibrate(0.5e-6, at_hz=d.f01)
    x = np.linspace(-d.radius, d.radius, 401)
    xi, eta = np.meshgrid(x, x)
    assert np.abs(d.complex_amplitude(xi, eta, d.f01, level)).max() == pytest.approx(
        0.5e-6, rel=0.02
    )


def test_fixed_level_amplitude_follows_resonance():
    d = Drumhead()
    level = d.calibrate(0.5e-6, at_hz=d.f01)
    x = np.linspace(-d.radius, d.radius, 201)
    xi, eta = np.meshgrid(x, x)
    on = np.abs(d.complex_amplitude(xi, eta, d.f01, level)).max()
    off = np.abs(d.complex_amplitude(xi, eta, 0.8 * d.f01, level)).max()
    assert on > 5 * off


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        Drumhead(radius=0.1, drive=PointDrive(r=0.1))
    with pytest.raises(ValueError):
        Drumhead(q=0.0)
    with pytest.raises(ValueError):
        Drumhead(f01=-1.0)
    with pytest.raises(ValueError):
        Drumhead().modes(0.0)


def _direct_sum(drum, xi, eta, f, level):
    from scipy.special import jv

    r, t = np.hypot(xi, eta), np.arctan2(eta, xi)
    th_d = np.radians(drum.drive.theta_deg)
    w = sum(
        md.amplitude * jv(md.m, md.alpha * r / drum.radius) * np.cos(md.m * (t - th_d))
        for md in drum.modes(f)
    )
    return np.where(r < drum.radius, level * w, 0)


def test_fast_evaluation_matches_direct_bessel_sum():
    d = Drumhead()
    rng = np.random.default_rng(4)
    xi, eta = rng.uniform(-0.1, 0.1, (2, 2000))
    for f in (d.f01, 1.7 * d.f01, 3.1 * d.f01):
        ref = _direct_sum(d, xi, eta, f, 1.0)
        got = d.complex_amplitude(xi, eta, f, 1.0)
        assert np.abs(got - ref).max() < 1e-6 * np.abs(ref).max()


def test_calibrate_and_grid_evaluation_are_fast():
    import time

    d = Drumhead()
    t = time.perf_counter()
    d.calibrate(0.5e-6, at_hz=d.f01)
    x = np.linspace(-0.11, 0.11, 1024)
    d.complex_amplitude(*np.meshgrid(x, x), d.f01, 1.0)
    assert time.perf_counter() - t < 1.0
