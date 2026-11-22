import numpy as np
import pytest

from espi_sim.laser import Laser

C = 299_792_458.0


def test_default_is_single_frequency_hene():
    laser = Laser()
    assert laser.wavelength == pytest.approx(632.8e-9)
    modes = laser.modes_at(0)
    assert len(modes) == 1 and modes[0].power == pytest.approx(1.0)
    assert abs(laser.coherence(10.0)) == pytest.approx(1.0)


def test_multimode_spacing_is_c_over_2l():
    laser = Laser.hene_multimode(cavity_length=0.3)
    assert laser.mode_spacing_hz == pytest.approx(C / 0.6)


def test_centered_comb_has_three_modes_with_strongest_at_line_centre():
    modes = Laser.hene_multimode(cavity_length=0.3, comb_offset_hz=0.0).modes_at(0)
    assert len(modes) == 3
    powers = sorted(m.power for m in modes)
    assert sum(powers) == pytest.approx(1.0)
    assert powers[-1] > powers[0]


def test_two_equal_modes_lose_visibility_at_opd_equal_cavity_length():
    L = 0.3
    # Comb straddles line centre at +/- spacing/2; threshold 0.6 keeps only those two.
    laser = Laser.hene_multimode(cavity_length=L, comb_offset_hz=C / (4 * L), gain_threshold=0.6)
    modes = laser.modes_at(0)
    assert len(modes) == 2
    assert modes[0].power == pytest.approx(modes[1].power)
    assert abs(laser.coherence(0.0)) == pytest.approx(1.0)
    assert abs(laser.coherence(L)) == pytest.approx(0.0, abs=1e-9)
    assert abs(laser.coherence(2 * L)) == pytest.approx(1.0)  # revival at 2L


def test_mode_sweep_changes_interference_phase_at_nonzero_opd():
    laser = Laser.hene_multimode(cavity_length=0.3, sweep_hz_per_frame=50e6)
    g0, g1 = laser.coherence(0.05, frame=0), laser.coherence(0.05, frame=1)
    assert abs(np.angle(g1 / g0)) > 1e-3
    # Zero path mismatch is immune to frequency drift.
    assert np.angle(laser.coherence(0.0, frame=7)) == pytest.approx(0.0)
