import numpy as np
import pytest

from espi_sim.config import Camera, ObjectBeam, Optics, PhaseShifter, ReferenceBeam, Setup
from espi_sim.rng import stream

LAM = 632.8e-9


def test_normal_out_of_plane_sensitivity_is_minus_4pi_over_lambda_along_z():
    s = Setup.out_of_plane(theta_deg=0).sensitivity(LAM)
    np.testing.assert_allclose(s, [0, 0, -4 * np.pi / LAM], atol=1e-6)


def test_oblique_out_of_plane_mixes_in_plane_u():
    s = Setup.out_of_plane(theta_deg=30).sensitivity(LAM)
    k = 2 * np.pi / LAM
    np.testing.assert_allclose(
        s, [-k * np.sin(np.pi / 6), 0, -k * (1 + np.cos(np.pi / 6))], rtol=1e-12
    )


def test_dual_beam_in_plane_is_insensitive_to_w():
    s = Setup.in_plane(theta_deg=45).sensitivity(LAM)
    np.testing.assert_allclose(s, [-4 * np.pi / LAM * np.sin(np.pi / 4), 0, 0], atol=1e-6)


def test_setup_validation():
    with pytest.raises(ValueError):
        Setup(beams=(ObjectBeam(),), reference=None)
    with pytest.raises(ValueError):
        Setup(beams=(ObjectBeam(), ObjectBeam()), reference=ReferenceBeam())


def test_optics_cutoff_and_speckle_size():
    o = Optics(f_number=8, magnification=0.1)
    assert o.coherent_cutoff(LAM) == pytest.approx(1 / (2 * LAM * 1.1 * 8))
    assert o.speckle_size(LAM) == pytest.approx(1.22 * LAM * 1.1 * 8)


def test_camera_gain_defaults_to_full_well_over_adc_range():
    cam = Camera(full_well=10_000, bit_depth=12)
    assert cam.electrons_per_adu == pytest.approx(10_000 / 4095)


def test_phase_shifter_nominal_and_miscalibrated_steps():
    rng = stream(0, "vib")
    np.testing.assert_allclose(PhaseShifter().actual_steps(rng), [0, np.pi / 2, np.pi, 1.5 * np.pi])
    bad = PhaseShifter(step_error=0.05).actual_steps(rng)
    np.testing.assert_allclose(bad, 1.05 * np.array([0, np.pi / 2, np.pi, 1.5 * np.pi]))
    jit = PhaseShifter(jitter_rms=0.1).actual_steps(rng)
    assert not np.allclose(jit, [0, np.pi / 2, np.pi, 1.5 * np.pi])


def test_pzt_displacement_for_quarter_wave_step():
    # A pi/2 step in reflection is a lambda/8 mirror move: 79.1 nm for HeNe.
    assert PhaseShifter.mirror_displacement(np.pi / 2, LAM) == pytest.approx(79.1e-9, rel=1e-3)


def test_simconfig_video_defaults():
    from espi_sim.config import SimConfig
    from espi_sim.waveforms import VideoTiming

    assert SimConfig().reference_drive is None
    assert SimConfig().timing == VideoTiming(30.0, 0.03)
