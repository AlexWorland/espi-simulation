import numpy as np
import pytest

from espi_sim.displacement import (
    ClampedPlatePointLoad,
    ClampedPlatePressure,
    RigidRotation,
    RigidTranslation,
    UniformStrain,
    Zero,
)

XI, ETA = np.meshgrid(np.linspace(-1e-2, 1e-2, 41), np.linspace(-1e-2, 1e-2, 41))


def test_rigid_translation_is_uniform():
    d = RigidTranslation(u=1e-7, v=-2e-7, w=3e-7)(XI, ETA)
    assert np.all(d.u == 1e-7) and np.all(d.v == -2e-7) and np.all(d.w == 3e-7)
    assert d.u.shape == XI.shape


def test_zero_field():
    d = Zero()(XI, ETA)
    assert not np.any(d.u) and not np.any(d.v) and not np.any(d.w)


def test_small_rotation_about_x_raises_positive_eta_toward_camera():
    d = RigidRotation(rx=1e-5)(XI, ETA)
    np.testing.assert_allclose(d.w, 1e-5 * ETA)
    assert not np.any(d.u) and not np.any(d.v)


def test_small_rotation_about_y_and_z():
    d = RigidRotation(ry=2e-5, rz=3e-5)(XI, ETA)
    np.testing.assert_allclose(d.w, -2e-5 * XI)
    np.testing.assert_allclose(d.u, -3e-5 * ETA)
    np.testing.assert_allclose(d.v, 3e-5 * XI)


@pytest.mark.parametrize(
    ("field", "rho_half"),
    [
        (ClampedPlatePressure, 0.75**2),
        (ClampedPlatePointLoad, 1 - 0.25 + 2 * 0.25 * np.log(0.5)),
    ],
)
def test_clamped_plate_profiles(field, rho_half):
    a, w0 = 8e-3, 2e-6
    plate = field(radius=a, center_deflection=w0)
    r = np.array([0.0, 0.5 * a, a, 1.5 * a])
    d = plate(r, np.zeros_like(r))
    np.testing.assert_allclose(d.w, [w0, rho_half * w0, 0.0, 0.0], atol=1e-15)
    assert not np.any(d.u) and not np.any(d.v)


def test_plate_center_offset():
    plate = ClampedPlatePressure(radius=1e-3, center_deflection=1e-6, center=(2e-3, -1e-3))
    d = plate(np.array([2e-3]), np.array([-1e-3]))
    np.testing.assert_allclose(d.w, [1e-6])


def test_uniform_strain_and_thermal_expansion():
    d = UniformStrain(exx=1e-4, eyy=2e-4, exy=5e-5)(XI, ETA)
    np.testing.assert_allclose(d.u, 1e-4 * XI + 5e-5 * ETA)
    np.testing.assert_allclose(d.v, 5e-5 * XI + 2e-4 * ETA)
    t = UniformStrain.thermal(alpha=23e-6, delta_t=10.0)
    assert t.exx == pytest.approx(2.3e-4) and t.eyy == pytest.approx(2.3e-4) and t.exy == 0


def test_fields_superpose():
    combo = RigidTranslation(w=1e-7) + RigidRotation(rx=1e-5) + UniformStrain(exx=1e-4)
    d = combo(XI, ETA)
    np.testing.assert_allclose(d.w, 1e-7 + 1e-5 * ETA)
    np.testing.assert_allclose(d.u, 1e-4 * XI)


def test_in_plane_flag():
    assert not ClampedPlatePressure(radius=1e-3, center_deflection=1e-6).has_in_plane
    assert UniformStrain(exx=1e-4).has_in_plane
    assert (RigidTranslation(w=1e-7) + RigidTranslation(u=1e-7)).has_in_plane
    assert not RigidTranslation(w=1e-7).has_in_plane
