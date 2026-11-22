import numpy as np
import pytest

from espi_sim.waveforms import SteppedSweep, Sweep, Tone, TriangleWave, VideoTiming


def test_tone_is_constant():
    t = Tone(frequency_hz=410.0, level=2e-8)
    assert t.tone_at(0.0) == (410.0, 2e-8) == t.tone_at(99.0)
    assert t.rate_at(3.0) == 0.0


def test_stepped_sweep_indexes_dwell_and_clamps():
    s = SteppedSweep(frequencies_hz=(100.0, 200.0, 300.0), dwell_s=1.0, levels=5.0)
    assert s.tone_at(0.5) == (100.0, 5.0)
    assert s.tone_at(1.0)[0] == 200.0
    assert s.tone_at(9.0)[0] == 300.0
    s2 = SteppedSweep(frequencies_hz=(100.0, 200.0, 300.0), dwell_s=1.0, levels=(1.0, 0.0, 2.0))
    assert s2.tone_at(1.5) == (200.0, 0.0)
    assert s2.rate_at(1.5) == 0.0


def test_linear_and_log_sweep_endpoints_midpoints_and_rates():
    lin = Sweep(f_start=100.0, f_stop=400.0, duration_s=10.0, level=1.0)
    assert lin.tone_at(5.0) == (pytest.approx(250.0), 1.0)
    assert lin.rate_at(5.0) == pytest.approx(30.0)
    assert lin.tone_at(-1.0)[0] == pytest.approx(100.0)
    assert lin.tone_at(20.0)[0] == pytest.approx(400.0)
    log = Sweep(f_start=100.0, f_stop=400.0, duration_s=10.0, level=1.0, spacing="log")
    assert log.tone_at(5.0)[0] == pytest.approx(200.0)
    assert log.rate_at(5.0) == pytest.approx(200.0 * np.log(4.0) / 10.0)
    with pytest.raises(ValueError):
        Sweep(100.0, 400.0, 10.0, 1.0, spacing="cubic")


def test_triangle_wave_shape():
    tri = TriangleWave(frequency_hz=1.0, phase_pp=4 * np.pi, offset=0.5)
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        tri.phase_at(t), [0.5, 0.5 + 2 * np.pi, 0.5 + 4 * np.pi, 0.5 + 2 * np.pi, 0.5]
    )
    nl = TriangleWave(frequency_hz=1.0, phase_pp=4 * np.pi, offset=0.5, nonlinearity=0.01)
    d = tri.phase_at(t)
    np.testing.assert_allclose(nl.phase_at(t), d + 0.01 * d**2)


def test_phase_for_travel():
    assert TriangleWave.phase_for_travel(632.8e-9 / 2, 632.8e-9) == pytest.approx(2 * np.pi)


def test_video_timing():
    vt = VideoTiming(frame_rate_hz=30.0, exposure_s=0.03)
    assert vt.frame_start(3) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        VideoTiming(frame_rate_hz=30.0, exposure_s=0.05)
    with pytest.raises(ValueError):
        VideoTiming(frame_rate_hz=0.0, exposure_s=0.01)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_tone_and_stepped_sweep_reject_non_positive_frequency(bad):
    with pytest.raises(ValueError):
        Tone(frequency_hz=bad, level=1.0)
    with pytest.raises(ValueError):
        SteppedSweep(frequencies_hz=(bad, 400.0), dwell_s=1.0, levels=1.0)
