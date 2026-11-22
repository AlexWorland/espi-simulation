"""Analytic displacement fields: the ground truth the simulator is scored against.

A field maps object-plane coordinates ``(xi, eta)`` in metres (origin at the centre of the
field of view, x to the right, y up, z toward the camera) to a displacement ``(u, v, w)``
in metres. Fields are small-displacement and linear, so they superpose with ``+``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Displacement(NamedTuple):
    u: NDArray[np.float64]
    v: NDArray[np.float64]
    w: NDArray[np.float64]


class DisplacementField:
    """Base class. Subclasses implement :meth:`evaluate`."""

    #: Whether the field can produce in-plane (u, v) motion. False lets the optics skip
    #: the costly microstructure warp.
    has_in_plane: bool = True

    def evaluate(self, xi: NDArray[np.float64], eta: NDArray[np.float64]) -> Displacement:
        raise NotImplementedError

    def __call__(self, xi: ArrayLike, eta: ArrayLike) -> Displacement:
        xi, eta = np.broadcast_arrays(np.asarray(xi, float), np.asarray(eta, float))
        return self.evaluate(xi, eta)

    def __add__(self, other: DisplacementField) -> DisplacementField:
        return Superposition((self, other))


@dataclass(frozen=True)
class Superposition(DisplacementField):
    fields: tuple[DisplacementField, ...]

    @property
    def has_in_plane(self) -> bool:  # type: ignore[override]
        return any(f.has_in_plane for f in self.fields)

    def evaluate(self, xi, eta):
        parts = [f(xi, eta) for f in self.fields]
        return Displacement(*(sum(p[i] for p in parts) for i in range(3)))

    def __add__(self, other):
        return Superposition((*self.fields, other))


def _zeros(xi) -> NDArray[np.float64]:
    return np.zeros_like(xi, dtype=float)


@dataclass(frozen=True)
class Zero(DisplacementField):
    """No motion: the reference (undeformed) state."""

    has_in_plane = False

    def evaluate(self, xi, eta):
        return Displacement(_zeros(xi), _zeros(xi), _zeros(xi))


@dataclass(frozen=True)
class RigidTranslation(DisplacementField):
    u: float = 0.0
    v: float = 0.0
    w: float = 0.0

    @property
    def has_in_plane(self) -> bool:  # type: ignore[override]
        return self.u != 0 or self.v != 0

    def evaluate(self, xi, eta):
        z = _zeros(xi)
        return Displacement(z + self.u, z + self.v, z + self.w)


@dataclass(frozen=True)
class RigidRotation(DisplacementField):
    """Small rigid rotation (radians) about the x, y and z axes through ``center``.

    ``w = rx*eta - ry*xi``, ``u = -rz*eta``, ``v = rz*xi`` (right-handed, small angles).
    """

    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)

    @property
    def has_in_plane(self) -> bool:  # type: ignore[override]
        return self.rz != 0

    def evaluate(self, xi, eta):
        x, y = xi - self.center[0], eta - self.center[1]
        return Displacement(-self.rz * y, self.rz * x, self.rx * y - self.ry * x)


@dataclass(frozen=True)
class _ClampedPlate(DisplacementField):
    radius: float
    center_deflection: float
    center: tuple[float, float] = (0.0, 0.0)

    has_in_plane = False

    def _profile(self, rho: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    def evaluate(self, xi, eta):
        rho = np.hypot(xi - self.center[0], eta - self.center[1]) / self.radius
        inside = rho < 1.0
        w = np.zeros_like(rho)
        w[inside] = self.center_deflection * self._profile(rho[inside])
        return Displacement(_zeros(xi), _zeros(xi), w)


@dataclass(frozen=True)
class ClampedPlatePressure(_ClampedPlate):
    """Clamped circular plate under uniform pressure: ``w = w0 (1 - rho^2)^2``."""

    def _profile(self, rho):
        return (1.0 - rho**2) ** 2


@dataclass(frozen=True)
class ClampedPlatePointLoad(_ClampedPlate):
    """Clamped circular plate, central point load: ``w = w0 (1 - rho^2 + 2 rho^2 ln rho)``."""

    def _profile(self, rho):
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(rho > 0, 2 * rho**2 * np.log(np.where(rho > 0, rho, 1.0)), 0.0)
        return 1.0 - rho**2 + log_term


@dataclass(frozen=True)
class UniformStrain(DisplacementField):
    """Homogeneous in-plane strain about ``center``.

    ``u = exx x + exy y``, ``v = exy x + eyy y``.
    """

    exx: float = 0.0
    eyy: float = 0.0
    exy: float = 0.0
    center: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def thermal(cls, alpha: float, delta_t: float, center=(0.0, 0.0)) -> UniformStrain:
        """Free isotropic thermal expansion: ``exx = eyy = alpha * delta_t``."""
        e = alpha * delta_t
        return cls(exx=e, eyy=e, center=center)

    def evaluate(self, xi, eta):
        x, y = xi - self.center[0], eta - self.center[1]
        return Displacement(self.exx * x + self.exy * y, self.exy * x + self.eyy * y, _zeros(xi))
