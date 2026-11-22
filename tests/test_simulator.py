import numpy as np
import pytest
from scipy.special import j1

from espi_sim.analysis import (
    carrier_phase,
    phase_difference,
    phase_from_steps,
    sincos_filter,
    unwrap,
    wrap,
)
from espi_sim.config import (
    Camera,
    ObjectBeam,
    Optics,
    PhaseShifter,
    ReferenceBeam,
    Setup,
    SimConfig,
)
from espi_sim.displacement import (
    ClampedPlatePressure,
    RigidRotation,
    RigidTranslation,
    UniformStrain,
    Zero,
)
from espi_sim.laser import SPEED_OF_LIGHT, Laser
from espi_sim.simulator import Simulator

LAM = 632.8e-9
IDEAL = Camera(shape=(128, 128), shot_noise=False, read_noise=0.0)


def _sim(camera=IDEAL, **kw):
    return Simulator(SimConfig(camera=camera, **kw))


def _phase_error(before, after, steps_from="actual"):
    sb = before.phase_steps if steps_from == "actual" else before.nominal_steps
    sa = after.phase_steps if steps_from == "actual" else after.nominal_steps
    pb = phase_from_steps(before.electrons, sb)
    pa = phase_from_steps(after.electrons, sa)
    err = wrap(phase_difference(pa.phase, pb.phase) - after.truth.phase)
    valid = np.minimum(pa.modulation, pb.modulation) > 0.05 * np.median(pb.modulation)
    return err, valid


def test_acquisition_shapes_dtypes_and_zero_truth():
    acq = _sim().acquire(Zero())
    assert acq.frames.shape == (4, 128, 128) and acq.frames.dtype == np.uint16
    assert acq.electrons.shape == (4, 128, 128)
    assert acq.phase_steps.shape == (4,)
    assert acq.truth.phase.shape == (128, 128) and not np.any(acq.truth.phase)


def test_frames_are_reproducible_from_the_seed():
    cam = Camera(shape=(64, 64))
    a = Simulator(SimConfig(camera=cam, seed=11)).acquire(Zero()).frames
    b = Simulator(SimConfig(camera=cam, seed=11)).acquire(Zero()).frames
    c = Simulator(SimConfig(camera=cam, seed=12)).acquire(Zero()).frames
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_mean_irradiance_is_sum_of_beam_intensities():
    acq = _sim(setup=Setup.out_of_plane(object_electrons=2000, reference_electrons=3000)).acquire(
        Zero()
    )
    assert acq.electrons.mean() == pytest.approx(5000.0, rel=0.05)


def test_quarter_wave_rigid_out_of_plane_shift_gives_pi_phase():
    before, after = _sim().acquire_pair(RigidTranslation(w=LAM / 4))
    np.testing.assert_allclose(wrap(after.truth.phase), np.pi)
    err, valid = _phase_error(before, after)
    assert valid.mean() > 0.9
    assert np.abs(err[valid]).max() < 1e-6


def test_truth_phase_is_sensitivity_dot_displacement():
    sim = _sim(setup=Setup.out_of_plane(theta_deg=30))
    d = RigidRotation(rx=2e-5, ry=-1e-5)
    truth = sim.truth(d)
    xi, eta = sim.grid.pixel_object_coords()
    disp = d(xi, eta)
    s = Setup.out_of_plane(theta_deg=30).sensitivity(LAM)
    np.testing.assert_allclose(truth.phase, s[0] * disp.u + s[1] * disp.v + s[2] * disp.w)
    np.testing.assert_allclose(truth.w, disp.w)


def test_dual_beam_in_plane_recovers_thermal_strain_phase_without_bias():
    sim = _sim(setup=Setup.in_plane(theta_deg=45))
    before, after = sim.acquire_pair(UniformStrain.thermal(alpha=23e-6, delta_t=20.0))
    assert np.ptp(after.truth.phase) > 4 * np.pi  # several fringes across the field
    err, valid = _phase_error(before, after)
    # Residual is strain-induced speckle decorrelation (see next test): random, not biased.
    assert abs(np.mean(err[valid])) < 0.02
    assert np.median(np.abs(err[valid])) < 0.35


@pytest.mark.parametrize("delta_t", [2.5, 10.0, 20.0])
def test_strain_decorrelation_matches_pupil_overlap(delta_t):
    # Under oblique illumination each beam's phase ramps across a strained surface, which
    # shifts its scattered spectrum within the pupil. The before/after field correlation
    # is the overlap area of two pupil disks offset by that shift.
    sim = _sim(setup=Setup.in_plane(theta_deg=45))
    strain = UniformStrain.thermal(alpha=23e-6, delta_t=delta_t)
    f0, f1 = sim.image_fields(Zero()), sim.image_fields(strain)
    disp = strain(*sim.grid.object_coords())
    for j, beam in enumerate(sim.config.setup.beams):
        s = beam.sensitivity(LAM)
        ramp = s[0] * disp.u + s[1] * disp.v + s[2] * disp.w
        x0, x1 = sim.grid.crop(f0[j]), sim.grid.crop(f1[j] * np.exp(-1j * ramp))
        mu = abs(np.vdot(x1, x0)) / np.sqrt(np.vdot(x0, x0).real * np.vdot(x1, x1).real)
        shift = abs(s[0]) * strain.exx / (2 * np.pi) / sim.config.optics.magnification
        r = shift / sim.config.optics.coherent_cutoff(LAM)
        overlap = (2 * np.arccos(r / 2) - (r / 2) * np.sqrt(4 - r**2)) / np.pi
        assert mu == pytest.approx(overlap, abs=0.005)


def test_field_correlation_under_in_plane_translation_follows_airy_law():
    sim = _sim()
    fc = sim.config.optics.coherent_cutoff(LAM)  # cycles/m, image plane
    speckle = sim.config.optics.speckle_size(LAM)
    m = sim.config.optics.magnification
    e0 = sim.grid.crop(sim.image_fields(Zero())[0])
    for n_speckles in (0.25, 0.5, 1.0, 2.0):
        shift_img = n_speckles * speckle
        e1 = sim.grid.crop(sim.image_fields(RigidTranslation(u=shift_img / m))[0])
        mu = np.abs(np.vdot(e1, e0)) / np.sqrt(np.vdot(e0, e0).real * np.vdot(e1, e1).real)
        v = 2 * np.pi * fc * shift_img
        assert mu == pytest.approx(abs(2 * j1(v) / v), abs=0.05), n_speckles


def test_large_in_plane_motion_destroys_out_of_plane_fringes():
    speckle_obj = Optics().speckle_size(LAM) / Optics().magnification
    small = _sim().acquire_pair(RigidTranslation(u=0.05 * speckle_obj, w=LAM / 4))
    large = _sim().acquire_pair(RigidTranslation(u=3.0 * speckle_obj, w=LAM / 4))
    err_small, v_small = _phase_error(*small)
    err_large, v_large = _phase_error(*large)
    assert np.mean(np.abs(err_small[v_small])) < 0.3
    assert np.mean(np.abs(err_large[v_large])) > 1.2  # ~ pi/2 for random phase


def test_path_mismatch_of_one_cavity_length_kills_two_mode_hene_fringes():
    L = 0.3
    laser = Laser.hene_multimode(
        cavity_length=L, comb_offset_hz=SPEED_OF_LIGHT / (4 * L), gain_threshold=0.6
    )
    matched = _sim(laser=laser).acquire(Zero())
    mismatched = _sim(
        laser=laser, setup=Setup(beams=(ObjectBeam(),), reference=ReferenceBeam(path_length=L))
    ).acquire(Zero())
    mod_m = phase_from_steps(matched.electrons, matched.phase_steps).modulation
    mod_x = phase_from_steps(mismatched.electrons, mismatched.phase_steps).modulation
    assert np.median(mod_x) < 0.02 * np.median(mod_m)


def test_relative_intensity_noise_varies_frame_brightness():
    acq = _sim(
        laser=Laser(relative_intensity_noise=0.02),
        phase_shifter=PhaseShifter(steps=tuple(np.linspace(0, 2 * np.pi, 16, endpoint=False))),
    ).acquire(Zero())
    means = acq.electrons.mean(axis=(1, 2))
    assert np.std(means) / np.mean(means) == pytest.approx(0.02, rel=0.5)


def test_jittered_steps_are_reported_and_used():
    sim = _sim(phase_shifter=PhaseShifter(jitter_rms=0.2))
    before, after = sim.acquire_pair(RigidTranslation(w=LAM / 8))
    assert not np.allclose(after.phase_steps, after.nominal_steps)
    err_actual, valid = _phase_error(before, after, "actual")
    err_nominal, _ = _phase_error(before, after, "nominal")
    assert np.abs(err_actual[valid]).max() < 1e-6
    assert np.abs(err_nominal[valid]).mean() > 0.01


def test_spatial_carrier_single_frame_recovers_uniform_phase_step():
    carrier = (0.25, 0.0)
    sim = _sim(
        optics=Optics(f_number=40),  # small speckle bandwidth so the lobes separate
        setup=Setup(beams=(ObjectBeam(),), reference=ReferenceBeam(carrier=carrier)),
        phase_shifter=PhaseShifter(steps=(0.0,)),
    )
    before, after = sim.acquire_pair(RigidTranslation(w=LAM / 8))
    pb = carrier_phase(before.electrons[0], carrier, radius=0.09)
    pa = carrier_phase(after.electrons[0], carrier, radius=0.09)
    err = wrap(phase_difference(pa, pb) - after.truth.phase)[16:-16, 16:-16]
    assert np.median(np.abs(err)) < 0.1


def test_end_to_end_bending_plate_with_noisy_camera():
    cam = Camera(shape=(256, 256))  # default: shot + read noise, 12-bit
    sim = Simulator(SimConfig(camera=cam, seed=3))
    fov = 256 * cam.pixel_pitch / sim.config.optics.magnification
    plate = ClampedPlatePressure(radius=0.4 * fov, center_deflection=2 * LAM)  # 8 pi at centre
    before, after = sim.acquire_pair(plate)
    pb = phase_from_steps(before.frames, before.phase_steps)
    pa = phase_from_steps(after.frames, after.phase_steps)
    filtered = sincos_filter(phase_difference(pa.phase, pb.phase), size=7, iterations=2)
    recovered = unwrap(filtered)
    truth = after.truth.phase
    inner = (slice(8, -8), slice(8, -8))
    resid = (recovered - truth)[inner]
    resid -= np.median(resid)
    assert np.ptp(truth) > 7 * np.pi
    assert np.sqrt(np.mean(resid**2)) < 0.3
