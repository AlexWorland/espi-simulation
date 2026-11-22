"""Laser source model, with defaults for a helium-neon (HeNe) laser at 632.8 nm.

A free-running HeNe oscillates on the longitudinal modes of its cavity, spaced ``c / 2L``
apart, that sit above threshold under the Doppler-broadened neon gain curve (FWHM about
1.5 GHz). A 30 cm tube runs two or three modes. That limits the coherence length to the
order of the cavity length. As the tube warms, the comb sweeps across the gain curve,
which redistributes power between modes and shifts the interference phase wherever the
interfering arms are not path-matched.

The source is modelled as a set of mutually incoherent modes. Each arm ``a`` of the
interferometer has an optical path length ``L_a``, and the detected intensity is

    I = sum_m p_m |sum_a E_a exp(i 2 pi dnu_m L_a / c)|^2

where ``dnu_m`` is the mode's offset from the gain-line centre. The speckle fields
``E_a`` are taken to be identical across modes, because the relative wavelength change is
about 1e-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0
HENE_WAVELENGTH = 632.8e-9
_MIN_RELATIVE_POWER = 1e-9


@dataclass(frozen=True)
class LaserMode:
    offset_hz: float
    """Frequency offset from the gain-line centre."""
    power: float
    """Fraction of the total power (modes sum to 1)."""


@dataclass(frozen=True)
class Laser:
    wavelength: float = HENE_WAVELENGTH
    cavity_length: float | None = None
    """Cavity length in metres. ``None`` means an ideal single-frequency laser."""
    gain_fwhm_hz: float = 1.5e9
    gain_threshold: float = 0.5
    """Lasing threshold as a fraction of peak gain. Mode power is proportional to the
    gain in excess of threshold."""
    comb_offset_hz: float = 0.0
    """Offset of one comb line from the gain centre at frame 0."""
    sweep_hz_per_frame: float = 0.0
    """Comb drift per captured frame (thermal cavity growth during warm-up)."""
    relative_intensity_noise: float = 0.0
    """RMS frame-to-frame fluctuation of total output power (fraction)."""

    @classmethod
    def hene_multimode(cls, cavity_length: float = 0.3, **kwargs) -> Laser:
        """A free-running (unstabilised) multimode HeNe tube."""
        return cls(cavity_length=cavity_length, **kwargs)

    @property
    def mode_spacing_hz(self) -> float:
        if self.cavity_length is None:
            return float("inf")
        return SPEED_OF_LIGHT / (2.0 * self.cavity_length)

    def modes_at(self, frame: int = 0) -> tuple[LaserMode, ...]:
        if self.cavity_length is None:
            return (LaserMode(0.0, 1.0),)
        spacing = self.mode_spacing_hz
        offset = self.comb_offset_hz + self.sweep_hz_per_frame * frame
        reach = 3.0 * self.gain_fwhm_hz
        m = np.arange(np.floor((-reach - offset) / spacing), np.ceil((reach - offset) / spacing) + 1)
        nu = offset + m * spacing
        gain = np.exp(-4.0 * np.log(2.0) * (nu / self.gain_fwhm_hz) ** 2)
        excess = np.clip(gain - self.gain_threshold, 0.0, None)
        if excess.max() <= 0.0:  # everything below threshold: keep the strongest mode
            excess = (gain == gain.max()).astype(float)
        keep = excess > _MIN_RELATIVE_POWER * excess.max()
        powers = excess[keep] / excess[keep].sum()
        return tuple(LaserMode(float(f), float(p)) for f, p in zip(nu[keep], powers, strict=True))

    def coherence(self, opd: float, frame: int = 0) -> complex:
        """Complex degree of coherence between two arms differing in path by ``opd`` metres."""
        return complex(
            sum(
                m.power * np.exp(2j * np.pi * m.offset_hz * opd / SPEED_OF_LIGHT)
                for m in self.modes_at(frame)
            )
        )
