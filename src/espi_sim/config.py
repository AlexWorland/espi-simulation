"""Configuration of the simulated ESPI setup: geometry, optics, camera and phase shifter.

Conventions
-----------
* Object surface in the z = 0 plane. +z points toward the camera, and viewing is
  telecentric along +z.
* An illumination source sits at polar angle ``theta`` from the surface normal and azimuth
  ``phi``. Its light travels along ``k_ill = -(sin t cos p, sin t sin p, cos t)``.
* The phase change of an arm whose surface point moves by ``d`` is
  ``(2 pi / lambda) (k_ill - k_obs) . d``, the change in optical path. The measured
  interference phase is arm 0 minus arm 1, where arm 1 is the reference beam, or the
  second illumination beam in a dual-beam setup.
* Intensities are in photoelectrons per pixel per exposure, for a 100 %-fill-factor pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .laser import Laser

OBSERVATION = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class ObjectBeam:
    """A collimated beam illuminating the rough object surface."""

    theta_deg: float = 0.0
    azimuth_deg: float = 0.0
    mean_electrons: float = 2500.0
    path_length: float = 0.0
    """Optical path length of this arm (metres, relative). Only differences matter."""

    @property
    def direction(self) -> np.ndarray:
        t, p = np.radians(self.theta_deg), np.radians(self.azimuth_deg)
        return -np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

    def sensitivity(self, wavelength: float) -> np.ndarray:
        """Phase change per unit displacement (rad/m), ``(2 pi/lambda)(k_ill - k_obs)``."""
        return 2 * np.pi / wavelength * (self.direction - OBSERVATION)


@dataclass(frozen=True)
class ReferenceBeam:
    """A smooth (specular) reference wave superimposed on the image."""

    mean_electrons: float = 2500.0
    carrier: tuple[float, float] = (0.0, 0.0)
    """Spatial carrier (fx, fy) in cycles per camera pixel, from tilting the reference."""
    path_length: float = 0.0


@dataclass(frozen=True)
class Setup:
    """Interferometer layout. Use one object beam with a reference (out-of-plane), or two
    object beams with no reference (dual-beam in-plane)."""

    beams: tuple[ObjectBeam, ...]
    reference: ReferenceBeam | None = None

    def __post_init__(self) -> None:
        if self.reference is not None and len(self.beams) != 1:
            raise ValueError("a reference-beam setup takes exactly one object beam")
        if self.reference is None and len(self.beams) != 2:
            raise ValueError("a setup without a reference beam needs exactly two object beams")

    @classmethod
    def out_of_plane(
        cls,
        theta_deg: float = 0.0,
        azimuth_deg: float = 0.0,
        object_electrons: float = 2500.0,
        reference_electrons: float = 2500.0,
    ) -> Setup:
        return cls(
            beams=(ObjectBeam(theta_deg, azimuth_deg, object_electrons),),
            reference=ReferenceBeam(reference_electrons),
        )

    @classmethod
    def in_plane(
        cls, theta_deg: float = 45.0, azimuth_deg: float = 0.0, electrons_per_beam: float = 2500.0
    ) -> Setup:
        """Symmetric dual-beam illumination at +/- theta in the plane of ``azimuth``."""
        return cls(
            beams=(
                ObjectBeam(theta_deg, azimuth_deg, electrons_per_beam),
                ObjectBeam(theta_deg, azimuth_deg + 180.0, electrons_per_beam),
            )
        )

    def sensitivity(self, wavelength: float) -> np.ndarray:
        """Sensitivity vector of the measured phase (arm 0 minus arm 1), rad/m."""
        s0 = self.beams[0].sensitivity(wavelength)
        if self.reference is not None:
            return s0
        return s0 - self.beams[1].sensitivity(wavelength)


@dataclass(frozen=True)
class Optics:
    """Imaging lens: aperture and magnification (telecentric, aberration-free)."""

    f_number: float = 8.0
    magnification: float = 0.1

    def coherent_cutoff(self, wavelength: float) -> float:
        """Coherent cutoff frequency in the image plane (cycles/m): ``1 / (2 lambda (1+M) F#)``."""
        return 1.0 / (2.0 * wavelength * (1.0 + self.magnification) * self.f_number)

    def speckle_size(self, wavelength: float) -> float:
        """Conventional subjective speckle size ``1.22 lambda (1+M) F#`` (metres, image plane)."""
        return 1.22 * wavelength * (1.0 + self.magnification) * self.f_number


@dataclass(frozen=True)
class Camera:
    shape: tuple[int, int] = (512, 512)
    """(rows, columns) in pixels."""
    pixel_pitch: float = 3.45e-6
    fill_factor: float = 1.0
    full_well: float = 10_500.0
    read_noise: float = 2.5
    """RMS read noise in electrons."""
    bit_depth: int = 12
    gain: float | None = None
    """Electrons per ADU. Defaults to full well over the ADC range."""
    offset_adu: int = 0
    shot_noise: bool = True

    @property
    def electrons_per_adu(self) -> float:
        return self.gain if self.gain is not None else self.full_well / (2**self.bit_depth - 1)

    @property
    def max_adu(self) -> int:
        return 2**self.bit_depth - 1


@dataclass(frozen=True)
class PhaseShifter:
    """Temporal phase stepping by a piezo (PZT) mirror in arm 1.

    Actual step = ``delta (1 + step_error) + nonlinearity * delta**2 + jitter``, where the
    jitter is an independent Gaussian draw per frame (vibration, air turbulence).
    """

    steps: tuple[float, ...] = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    step_error: float = 0.0
    nonlinearity: float = 0.0
    """Quadratic PZT nonlinearity coefficient (1/rad)."""
    jitter_rms: float = 0.0

    def actual_steps(self, rng: np.random.Generator) -> np.ndarray:
        d = np.asarray(self.steps, float)
        actual = d * (1.0 + self.step_error) + self.nonlinearity * d**2
        if self.jitter_rms:
            actual = actual + rng.normal(0.0, self.jitter_rms, d.shape)
        return actual

    @staticmethod
    def mirror_displacement(phase_step: float, wavelength: float) -> float:
        """Mirror travel (metres) for a phase step at normal-incidence reflection."""
        return phase_step * wavelength / (4.0 * np.pi)


@dataclass(frozen=True)
class SimConfig:
    setup: Setup = field(default_factory=Setup.out_of_plane)
    laser: Laser = field(default_factory=Laser)
    optics: Optics = field(default_factory=Optics)
    camera: Camera = field(default_factory=Camera)
    phase_shifter: PhaseShifter = field(default_factory=PhaseShifter)
    seed: int = 0
    oversample: int | None = None
    """Simulation samples per camera pixel per axis. ``None`` picks the smallest value that
    keeps the speckle intensity alias-free."""
    margin_px: int | None = None
    """Border (camera pixels) simulated beyond the field of view and then cropped, to
    suppress FFT wrap-around. ``None`` means about 4 speckle sizes."""
