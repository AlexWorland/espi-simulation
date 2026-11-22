import json

import numpy as np

import espi_sim
from espi_sim.io import load_acquisition, save_acquisition


def test_npz_round_trip_preserves_frames_truth_and_config(tmp_path):
    cfg = espi_sim.SimConfig(camera=espi_sim.Camera(shape=(32, 32)), seed=5)
    sim = espi_sim.Simulator(cfg)
    acq = sim.acquire(espi_sim.displacement.RigidTranslation(w=1e-7), frame_offset=4, label="x")
    path = tmp_path / "acq.npz"
    save_acquisition(path, acq, config=cfg)
    back, config_dict = load_acquisition(path)
    np.testing.assert_array_equal(back.frames, acq.frames)
    np.testing.assert_allclose(back.electrons, acq.electrons)
    np.testing.assert_allclose(back.phase_steps, acq.phase_steps)
    np.testing.assert_allclose(back.truth.phase, acq.truth.phase)
    np.testing.assert_allclose(back.truth.w, acq.truth.w)
    np.testing.assert_allclose(back.truth.sensitivity, acq.truth.sensitivity)
    assert back.label == "x" and back.frame_offset == 4
    assert config_dict["seed"] == 5
    assert config_dict["laser"]["wavelength"] == 632.8e-9
    json.dumps(config_dict)  # plain JSON-serialisable


def test_public_api_exports():
    for name in ("Simulator", "SimConfig", "Setup", "Camera", "Optics", "Laser", "PhaseShifter"):
        assert hasattr(espi_sim, name)


def test_video_npz_round_trip(tmp_path):
    import pytest

    from espi_sim.io import load_video, save_video

    cfg = espi_sim.SimConfig(
        camera=espi_sim.Camera(shape=(32, 32)),
        optics=espi_sim.Optics(magnification=32 * 3.45e-6 / 0.22),
        reference_drive=espi_sim.TriangleWave(frequency_hz=1.0),
    )
    drum = espi_sim.Drumhead(q=5)  # short dwell stays within the steady-state limit
    tone = espi_sim.SteppedSweep((400.0, 640.0), dwell_s=2 / 30, levels=drum.calibrate(3e-7, 400))
    video = espi_sim.Simulator(cfg).acquire_video(drum, tone, n_frames=4)
    path = tmp_path / "video.npz"
    save_video(path, video, config=cfg)
    back, config_dict = load_video(path)
    np.testing.assert_array_equal(back.frames, video.frames)
    for name in (
        "electrons",
        "times",
        "reference_phase",
        "reference_visibility",
        "frequency_hz",
        "level",
        "state_index",
        "sensitivity",
    ):
        np.testing.assert_allclose(getattr(back, name), getattr(video, name))
    assert len(back.states) == len(video.states) == 2
    for a, b in zip(back.states, video.states, strict=True):
        np.testing.assert_allclose(a.amplitude, b.amplitude)
        np.testing.assert_allclose(a.phase, b.phase)
        np.testing.assert_allclose(a.modulation, b.modulation)
        assert a.frequency_hz == b.frequency_hz and a.level == b.level
        assert [(m.m, m.n) for m in a.modes] == [(m.m, m.n) for m in b.modes]
        np.testing.assert_allclose([m.amplitude for m in a.modes], [m.amplitude for m in b.modes])
    assert back.metadata["tone"]["type"] == "SteppedSweep"
    assert config_dict["timing"]["frame_rate_hz"] == 30.0
    m1 = tmp_path / "m1.npz"
    espi_sim.save_acquisition(m1, espi_sim.Simulator(cfg).acquire())
    with pytest.raises(ValueError):
        load_video(m1)
