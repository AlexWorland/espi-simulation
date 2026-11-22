"""Vibrating drumhead: an ideal circular membrane, clamped at its rim, driven at one tone.

Modes ``(m, n)`` have shape ``J_m(alpha_mn r/a) cos(m(theta - theta_d))`` and frequency
``f_mn = f01 * alpha_mn / alpha_01``, where ``alpha_mn`` is the n-th zero of ``J_m``. For a
drive of angular frequency ``w`` the steady-state complex out-of-plane amplitude is

    W(x) = level * sum_mn a_mn phi_mn(x)
    a_mn = w01^2 F_mn / (N_mn (w_mn^2 - w^2 + 2 i zeta w_mn w)),   zeta = 1/(2Q)

with ``w(x, t) = Re(W(x) exp(i w t))``. ``N_mn`` is the area-normalised modal norm and
``F_mn`` the drive's projection onto the mode. ``level`` (metres) is a fixed drive
strength, so the amplitude follows the resonance curve. :meth:`Drumhead.calibrate` converts
a desired peak amplitude at one frequency into a level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import CubicSpline
from scipy.special import jn_zeros, jv

_MODE_THRESHOLD = 1e-4  # keep modes with |a| >= this fraction of the largest
_RADIAL_SAMPLES = 2049  # 1-D radial grid for R_m(rho); cubic-spline error ~1e-10


@dataclass(frozen=True)
class PointDrive:
    """A point force (e.g. a piezo or striker) at polar position ``(r, theta_deg)``."""

    r: float
    theta_deg: float = 0.0


@dataclass(frozen=True)
class UniformPressure:
    """Uniform pressure over the head (e.g. a loudspeaker underneath). Excites m = 0 only."""


@dataclass(frozen=True)
class MembraneMode:
    m: int
    n: int
    alpha: float
    frequency_hz: float
    amplitude: complex
    """Dimensionless modal coefficient ``a_mn`` (multiply by ``level`` for metres)."""


@dataclass(frozen=True)
class Drumhead:
    radius: float = 0.1
    f01: float = 400.0
    q: float = 50.0
    drive: PointDrive | UniformPressure = field(
        default_factory=lambda: PointDrive(r=0.04, theta_deg=30.0)
    )
    center: tuple[float, float] = (0.0, 0.0)
    max_m: int = 8
    max_n: int = 6

    def __post_init__(self) -> None:
        if self.radius <= 0 or self.f01 <= 0 or self.q <= 0:
            raise ValueError("radius, f01 and q must be positive")
        if isinstance(self.drive, PointDrive) and not 0 <= self.drive.r < self.radius:
            raise ValueError("the point drive must lie inside the head (0 <= r < radius)")

    @property
    def _theta_d(self) -> float:
        return np.radians(self.drive.theta_deg) if isinstance(self.drive, PointDrive) else 0.0

    def mode_frequency(self, m: int, n: int) -> float:
        return self.f01 * jn_zeros(m, n)[-1] / jn_zeros(0, 1)[0]

    def modes(self, frequency_hz: float) -> tuple[MembraneMode, ...]:
        """Modal response at ``frequency_hz``, keeping modes that contribute noticeably."""
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        w = 2 * np.pi * frequency_hz
        w01 = 2 * np.pi * self.f01
        zeta = 1.0 / (2.0 * self.q)
        a01 = jn_zeros(0, 1)[0]
        out = []
        for m in range(self.max_m + 1):
            for n, alpha in enumerate(jn_zeros(m, self.max_n), start=1):
                wmn = w01 * alpha / a01
                norm = jv(m + 1, alpha) ** 2 * (1.0 if m == 0 else 0.5)
                if isinstance(self.drive, PointDrive):
                    force = jv(m, alpha * self.drive.r / self.radius)
                else:
                    force = 2 * jv(1, alpha) / alpha if m == 0 else 0.0
                a = w01**2 * force / (norm * (wmn**2 - w**2 + 2j * zeta * wmn * w))
                out.append(MembraneMode(m, n, float(alpha), float(wmn / (2 * np.pi)), complex(a)))
        biggest = max(abs(md.amplitude) for md in out)
        return tuple(md for md in out if abs(md.amplitude) >= _MODE_THRESHOLD * biggest)

    def complex_amplitude(
        self, xi: ArrayLike, eta: ArrayLike, frequency_hz: float, level: float
    ) -> NDArray[np.complex128]:
        """Complex out-of-plane amplitude ``W`` (metres) at object-plane points; 0 off the head."""
        x = np.asarray(xi, float) - self.center[0]
        y = np.asarray(eta, float) - self.center[1]
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        inside = r < self.radius
        w = np.zeros(np.broadcast(x, y).shape, complex)
        if level == 0 or not inside.any():
            return w
        rr, tt = r[inside] / self.radius, theta[inside]
        acc = np.zeros(rr.shape, complex)
        for m, spline in _radial_profiles(self, float(frequency_hz)).items():
            radial = spline(rr)
            acc += radial if m == 0 else radial * np.cos(m * (tt - self._theta_d))
        w[inside] = level * acc
        return w

    def calibrate(self, peak_amplitude: float, at_hz: float) -> float:
        """Drive level giving ``max|W| = peak_amplitude`` at ``at_hz``."""
        r = np.linspace(0.0, self.radius, 400, endpoint=False)
        t = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
        rr, tt = np.meshgrid(r, t)
        xi = self.center[0] + rr * np.cos(tt)
        eta = self.center[1] + rr * np.sin(tt)
        peak = np.abs(self.complex_amplitude(xi, eta, at_hz, 1.0)).max()
        return peak_amplitude / peak


@lru_cache(maxsize=4)  # reuse is back-to-back (sim grid, then pixel centres) per tone
def _radial_profiles(drum: Drumhead, frequency_hz: float) -> dict[int, CubicSpline]:
    """Per azimuthal order m, a spline of ``R_m(rho) = sum_n a_mn J_m(alpha_mn rho)``.

    Evaluating the Bessel sums once on a fine 1-D grid, then interpolating, is far cheaper than
    calling ``jv`` per mode at every 2-D point. Cached per (drumhead, frequency).
    """
    rho = np.linspace(0.0, 1.0, _RADIAL_SAMPLES)
    sums: dict[int, NDArray[np.complex128]] = {}
    for md in drum.modes(frequency_hz):
        prof = sums.setdefault(md.m, np.zeros(rho.size, complex))
        prof += md.amplitude * jv(md.m, md.alpha * rho)
    return {m: CubicSpline(rho, prof) for m, prof in sums.items()}
