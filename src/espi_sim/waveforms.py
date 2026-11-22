"""Time schedules for video simulation: tone-generator drive, reference piezo and camera timing.

Tone schedules answer ``tone_at(t) -> (frequency_hz, level)`` and ``rate_at(t)`` (Hz/s).
Each video frame uses the tone at its mid-exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Tone:
    """A constant tone at a fixed drive level."""

    frequency_hz: float
    level: float

    def __post_init__(self) -> None:
        if not self.frequency_hz > 0:
            raise ValueError("frequency_hz must be positive (use level=0 to switch the drive off)")

    def tone_at(self, t: float) -> tuple[float, float]:
        return self.frequency_hz, self.level

    def rate_at(self, t: float) -> float:
        return 0.0


@dataclass(frozen=True)
class SteppedSweep:
    """Tones held for ``dwell_s`` each. ``levels`` may be one value or one per step, so a
    level of 0 switches the drive off. Times past the last step hold the last tone."""

    frequencies_hz: tuple[float, ...]
    dwell_s: float
    levels: float | tuple[float, ...]

    def __post_init__(self) -> None:
        if self.dwell_s <= 0 or len(self.frequencies_hz) == 0:
            raise ValueError("need at least one frequency and a positive dwell")
        if not all(f > 0 for f in self.frequencies_hz):
            raise ValueError("frequencies must be positive (use a level of 0 to switch off)")
        if not np.isscalar(self.levels) and len(self.levels) != len(self.frequencies_hz):
            raise ValueError("levels must be a scalar or one per frequency")

    def _index(self, t: float) -> int:
        return int(np.clip(np.floor(t / self.dwell_s), 0, len(self.frequencies_hz) - 1))

    def tone_at(self, t: float) -> tuple[float, float]:
        i = self._index(t)
        level = self.levels if np.isscalar(self.levels) else self.levels[i]
        return float(self.frequencies_hz[i]), float(level)

    def rate_at(self, t: float) -> float:
        return 0.0


@dataclass(frozen=True)
class Sweep:
    """Continuous sweep from ``f_start`` to ``f_stop`` over ``duration_s`` (then held)."""

    f_start: float
    f_stop: float
    duration_s: float
    level: float
    spacing: str = "linear"

    def __post_init__(self) -> None:
        if self.spacing not in ("linear", "log"):
            raise ValueError("spacing must be 'linear' or 'log'")
        if self.duration_s <= 0 or self.f_start <= 0 or self.f_stop <= 0:
            raise ValueError("frequencies and duration must be positive")

    def _u(self, t: float) -> float:
        return float(np.clip(t / self.duration_s, 0.0, 1.0))

    def tone_at(self, t: float) -> tuple[float, float]:
        u = self._u(t)
        if self.spacing == "linear":
            f = self.f_start + (self.f_stop - self.f_start) * u
        else:
            f = self.f_start * (self.f_stop / self.f_start) ** u
        return float(f), self.level

    def rate_at(self, t: float) -> float:
        if not 0.0 <= t <= self.duration_s:
            return 0.0
        if self.spacing == "linear":
            return (self.f_stop - self.f_start) / self.duration_s
        f, _ = self.tone_at(t)
        return f * np.log(self.f_stop / self.f_start) / self.duration_s


@dataclass(frozen=True)
class TriangleWave:
    """Reference-arm piezo driven by a slow triangle wave.

    ``delta = offset + phase_pp * tri(f t + start_fraction)``, where ``tri`` rises 0 → 1 over
    the first half period and falls back over the second. Then the PZT nonlinearity
    ``delta + nonlinearity * delta**2`` is applied.
    """

    frequency_hz: float = 0.25
    phase_pp: float = 4 * np.pi
    offset: float = 0.0
    start_fraction: float = 0.0
    nonlinearity: float = 0.0

    def phase_at(self, t: ArrayLike) -> NDArray[np.float64]:
        u = np.mod(self.frequency_hz * np.asarray(t, float) + self.start_fraction, 1.0)
        tri = np.where(u < 0.5, 2 * u, 2 * (1 - u))
        d = self.offset + self.phase_pp * tri
        return d + self.nonlinearity * d**2

    @staticmethod
    def phase_for_travel(travel: float, wavelength: float) -> float:
        """Reference phase for mirror ``travel`` at normal incidence: ``4 pi travel / lambda``."""
        return 4 * np.pi * travel / wavelength


@dataclass(frozen=True)
class VideoTiming:
    """Free-running video: frame ``k`` exposes over ``[k/fps, k/fps + exposure_s]``."""

    frame_rate_hz: float = 30.0
    exposure_s: float = 0.03

    def __post_init__(self) -> None:
        if self.frame_rate_hz <= 0 or self.exposure_s <= 0:
            raise ValueError("frame rate and exposure must be positive")
        if self.exposure_s > 1.0 / self.frame_rate_hz + 1e-12:
            raise ValueError("exposure cannot exceed the frame interval")

    def frame_start(self, k: int) -> float:
        return k / self.frame_rate_hz
