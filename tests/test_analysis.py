import numpy as np

from espi_sim.analysis import (
    phase_difference,
    phase_from_steps,
    sincos_filter,
    subtraction_fringes,
    unwrap,
    wrap,
)

RNG = np.random.default_rng(0)
PHI = RNG.uniform(-np.pi, np.pi, (16, 16))


def _frames(steps, a=1000.0, b=400.0, phi=PHI):
    return np.array([a + b * np.cos(phi - d) for d in steps])


def test_four_step_recovers_phase_modulation_and_background():
    steps = np.array([0, np.pi / 2, np.pi, 1.5 * np.pi])
    r = phase_from_steps(_frames(steps), steps)
    np.testing.assert_allclose(wrap(r.phase - PHI), 0, atol=1e-9)
    np.testing.assert_allclose(r.modulation, 400.0)
    np.testing.assert_allclose(r.background, 1000.0)


def test_least_squares_handles_arbitrary_known_steps():
    steps = np.array([0.1, 1.3, 2.0, 3.9, 5.0])
    r = phase_from_steps(_frames(steps), steps)
    np.testing.assert_allclose(wrap(r.phase - PHI), 0, atol=1e-9)


def test_wrong_steps_cause_phase_error():
    true_steps = 1.1 * np.array([0, np.pi / 2, np.pi, 1.5 * np.pi])
    r = phase_from_steps(_frames(true_steps), true_steps / 1.1)
    err = np.abs(wrap(r.phase - PHI))
    assert err.max() > 0.02


def test_wrap_range_and_difference():
    w = wrap(np.array([np.pi, -np.pi, 3 * np.pi, 0.5]))
    assert np.all(w <= np.pi) and np.all(w > -np.pi)
    np.testing.assert_allclose(
        phase_difference(np.array([3.0]), np.array([-3.0])), [6.0 - 2 * np.pi]
    )


def test_subtraction_fringes():
    np.testing.assert_allclose(
        subtraction_fringes(np.array([5.0, 2.0]), np.array([1.0, 4.0])), [4, 2]
    )


def test_sincos_filter_reduces_noise_and_keeps_a_ramp():
    y, x = np.mgrid[0:64, 0:64]
    ramp = 0.3 * x
    noisy = wrap(ramp + RNG.normal(0, 0.5, ramp.shape))
    filt = sincos_filter(noisy, size=7)
    inner = (slice(8, -8), slice(8, -8))
    assert np.std(wrap(filt - ramp)[inner]) < 0.5 * np.std(wrap(noisy - ramp)[inner])
    assert np.abs(np.mean(wrap(filt - ramp)[inner])) < 0.05


def test_carrier_phase_recovers_phase_from_one_frame():
    from espi_sim.analysis import carrier_phase

    rows, cols = np.mgrid[0:128, 0:128]
    # Periodic over the frame so the test isolates correctness from FFT edge leakage.
    phi = 2 * np.pi / 128 * (cols - 64) + 0.8 * np.sin(2 * np.pi * rows / 128) + 0.5
    carrier = (0.25, 0.125)  # cycles/pixel along (cols, rows)
    frame = 1000 + 400 * np.cos(phi - 2 * np.pi * (carrier[0] * cols + carrier[1] * rows))
    rec = carrier_phase(frame, carrier, radius=0.1)
    np.testing.assert_allclose(wrap(rec - phi), 0, atol=1e-9)


def test_unwrap_recovers_smooth_phase():
    y, x = np.mgrid[0:64, 0:64]
    true = 0.002 * ((x - 32) ** 2 + (y - 32) ** 2)
    u = unwrap(wrap(true))
    np.testing.assert_allclose(u - u.mean(), true - true.mean(), atol=1e-9)
