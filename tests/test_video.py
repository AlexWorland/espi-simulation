import warnings

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from espi_sim.analysis import phase_from_steps, wrap
from espi_sim.config import Camera, Optics, Setup, SimConfig
from espi_sim.drumhead import Drumhead
from espi_sim.simulator import Simulator
from espi_sim.waveforms import SteppedSweep, Sweep, Tone, TriangleWave, VideoTiming

LAM = 632.8e-9
DRUM = Drumhead()
MAG = 96 * 3.45e-6 / (2.2 * 0.1)  # head (0.2 m) plus rim fills the frame
IDEAL = Camera(shape=(96, 96), shot_noise=False, read_noise=0.0)
RAMP = TriangleWave(frequency_hz=30 / 16, phase_pp=2 * np.pi)  # 8 frames cover 2pi rising


TIMING = VideoTiming(30.0, 0.03)


def _sim(camera=IDEAL, timing=TIMING, **kw):
    cfg = SimConfig(
        camera=camera,
        optics=Optics(f_number=8, magnification=MAG),
        reference_drive=RAMP,
        timing=timing,
        **kw,
    )
    return Simulator(cfg)


def test_video_shapes_times_and_reference_phase():
    v = _sim().acquire_video(DRUM, Tone(400.0, DRUM.calibrate(0.3e-6, 400.0)), n_frames=8)
    assert v.frames.shape == (8, 96, 96) and v.frames.dtype == np.uint16
    assert v.electrons.shape == (8, 96, 96)
    np.testing.assert_allclose(v.times, np.arange(8) / 30)
    np.testing.assert_allclose(
        wrap(v.reference_phase - RAMP.phase_at(v.times + 0.015)), 0, atol=1e-3
    )
    assert np.all(v.reference_visibility > 0.9) and np.all(v.reference_visibility <= 1.0)
    assert len(v.states) == 1 and np.all(v.state_index == 0)
    np.testing.assert_allclose(v.frequency_hz, 400.0)


def test_time_averaged_fringe_term_recovers_signed_j0():
    # Fit the complex fringe term of a vibrating and a static video (same speckle). Their
    # local least-squares ratio Re(c_v c_s*) / |c_s|^2 estimates the signed J0(S_z |W|): the
    # blur of the J0 envelope across a speckle adds an uncorrelated term that averages out.
    sim = _sim()
    vib = sim.acquire_video(DRUM, Tone(400.0, DRUM.calibrate(0.4e-6, 400.0)), n_frames=8)
    sta = sim.acquire_video(DRUM, Tone(400.0, 0.0), n_frames=8)
    rv = phase_from_steps(vib.electrons, vib.reference_phase)
    rs = phase_from_steps(sta.electrons, sta.reference_phase)
    cv = rv.modulation * np.exp(1j * rv.phase)
    cs = rs.modulation * np.exp(1j * rs.phase)
    est = uniform_filter(np.real(cv * np.conj(cs)), 3) / uniform_filter(np.abs(cs) ** 2, 3)
    xi, eta = sim.grid.pixel_object_coords()
    inside = np.hypot(xi, eta) < 0.9 * DRUM.radius
    truth = vib.states[0].modulation
    assert np.median(np.abs(est - truth)[inside]) < 0.05
    assert np.mean(np.sign(est[inside]) == np.sign(truth[inside])) > 0.95
    ring = inside & (np.abs(vib.states[0].amplitude - 121.1e-9) < 5e-9)  # first J0 zero
    assert ring.sum() > 10
    assert np.median(np.abs(est[ring])) < 0.12


def test_drive_off_frames_equal_static_scene():
    sim = _sim()
    a = sim.acquire_video(DRUM, Tone(400.0, 0.0), n_frames=3)
    b = sim.acquire_video(DRUM, Tone(450.0, 0.0), n_frames=3)
    assert not np.any(a.states[0].amplitude)
    np.testing.assert_array_equal(a.states[0].modulation, 1.0)
    np.testing.assert_allclose(a.electrons, b.electrons, rtol=1e-12)


def test_stepped_sweep_reuses_states_and_peaks_on_resonance():
    drum = Drumhead(q=10)
    level = drum.calibrate(0.4e-6, 400.0)
    tone = SteppedSweep((360.0, 400.0, 440.0), dwell_s=4 / 30, levels=level)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = _sim().acquire_video(drum, tone, n_frames=12)
    assert len(v.states) == 3
    np.testing.assert_array_equal(v.state_index, [0] * 4 + [1] * 4 + [2] * 4)
    peaks = [s.amplitude.max() for s in v.states]
    assert peaks[1] > peaks[0] and peaks[1] > peaks[2]


def test_continuous_sweep_makes_one_state_per_frame():
    drum = Drumhead(q=10)
    level = drum.calibrate(0.4e-6, 400.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = _sim().acquire_video(drum, Sweep(380.0, 420.0, 6 / 30, level), n_frames=6)
    assert len(v.states) == 6
    assert np.all(np.diff(v.frequency_hz) > 0)


def test_fast_sweep_warns_steady_state():
    level = DRUM.calibrate(0.4e-6, 400.0)
    with pytest.warns(UserWarning, match="steady"):
        _sim().acquire_video(DRUM, Sweep(300.0, 900.0, 0.2, level), n_frames=2)


def test_short_exposure_warns_cycles():
    level = DRUM.calibrate(0.4e-6, 400.0)
    with pytest.warns(UserWarning, match="cycles"):
        _sim(timing=VideoTiming(30.0, 0.001)).acquire_video(DRUM, Tone(400.0, level), n_frames=2)


def test_undersampled_vibration_warns():
    with pytest.warns(UserWarning, match="under-sampled"):
        _sim().acquire_video(DRUM, Tone(400.0, DRUM.calibrate(10e-6, 400.0)), n_frames=1)


def test_rejects_dual_beam_and_bad_frame_count():
    with pytest.raises(ValueError):
        _sim(setup=Setup.in_plane()).acquire_video(DRUM, Tone(400.0, 0.0), n_frames=2)
    with pytest.raises(ValueError):
        _sim().acquire_video(DRUM, Tone(400.0, 0.0), n_frames=0)


def test_video_reproducible_and_frame_offset_continues():
    cam = Camera(shape=(96, 96))  # noisy
    tone = Tone(400.0, DRUM.calibrate(0.3e-6, 400.0))
    v8 = _sim(camera=cam).acquire_video(DRUM, tone, n_frames=8)
    again = _sim(camera=cam).acquire_video(DRUM, tone, n_frames=8)
    np.testing.assert_array_equal(v8.frames, again.frames)
    v4 = _sim(camera=cam).acquire_video(DRUM, tone, n_frames=4, frame_offset=4)
    assert v4.times[0] == pytest.approx(4 / 30)
    np.testing.assert_allclose(v4.reference_phase, v8.reference_phase[4:])
    np.testing.assert_array_equal(v4.frames, v8.frames[4:])


def test_fast_reference_ramp_relative_to_vibration_warns():
    level = DRUM.calibrate(0.3e-6, 400.0)
    fast = TriangleWave(frequency_hz=8.0, phase_pp=2 * np.pi)  # ~4 % of the vibration rate
    sim = Simulator(
        SimConfig(
            camera=IDEAL,
            optics=Optics(f_number=8, magnification=MAG),
            reference_drive=fast,
            timing=TIMING,
        )
    )
    with pytest.warns(UserWarning, match="reference"):
        sim.acquire_video(DRUM, Tone(400.0, level), n_frames=2)


def test_continuous_sweep_does_not_retain_per_state_terms():
    import tracemalloc

    drum = Drumhead(q=10)
    level = drum.calibrate(0.3e-6, 400.0)
    n_px = 128
    cam = Camera(shape=(n_px, n_px), shot_noise=False, read_noise=0.0)
    sim = _sim(camera=cam)
    sim.acquire_video(drum, Sweep(380.0, 420.0, 2 / 30, level), n_frames=2)  # warm one-time caches

    def peak(n):
        tracemalloc.start()
        sim.acquire_video(drum, Sweep(380.0, 420.0, n / 30, level), n_frames=n)
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p

    per_frame = (peak(24) - peak(8)) / 16 / (n_px * n_px)
    # Outputs need ~34 B/px/frame (electrons, frames, 3 truth maps); retaining each
    # state's base + cross terms adds ~24 B/px/frame more.
    assert per_frame < 45
