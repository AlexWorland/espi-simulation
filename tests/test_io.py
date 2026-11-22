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
