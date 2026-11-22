"""Coherent imaging of a rough surface: sampling grid, pupil, microstructure warp, FFT imaging.

The image-plane field of each beam is ``E = IFFT(FFT(U) * P)``, where ``U`` is the
object-plane field mapped into image coordinates and ``P`` is a circular pupil of radius
``f_c = 1 / (2 lambda (1+M) F#)``. The intensity ``|E|^2`` has twice the bandwidth of
``E``, so the grid is oversampled until ``f_c <= 0.25`` cycles per sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from .config import Camera, Optics

_MAX_CUTOFF = 0.25  # cycles/sample: keeps |E|^2 alias-free
_EPS = 1e-12


@dataclass(frozen=True)
class Grid:
    camera_shape: tuple[int, int]
    oversample: int
    margin: tuple[int, int]
    """Border in camera pixels on each side, (rows, cols)."""
    pixel_pitch: float
    magnification: float
    cutoff_cycles_per_sample: float

    @property
    def dx(self) -> float:
        """Sample spacing in the image plane (metres)."""
        return self.pixel_pitch / self.oversample

    @property
    def shape(self) -> tuple[int, int]:
        s = self.oversample
        return tuple((n + 2 * m) * s for n, m in zip(self.camera_shape, self.margin, strict=True))

    def object_coords(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Object-plane (xi, eta) of every simulation sample, in metres. eta points up."""
        ny, nx = self.shape
        scale = self.dx / self.magnification
        xi = (np.arange(nx) - (nx - 1) / 2) * scale
        eta = ((ny - 1) / 2 - np.arange(ny)) * scale
        return np.meshgrid(xi, eta)

    def pixel_object_coords(
        self, shape: tuple[int, int] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Object-plane (xi, eta) of camera pixel centres."""
        ny, nx = shape or self.camera_shape
        scale = self.pixel_pitch / self.magnification
        xi = (np.arange(nx) - (nx - 1) / 2) * scale
        eta = ((ny - 1) / 2 - np.arange(ny)) * scale
        return np.meshgrid(xi, eta)

    def crop(self, a: NDArray) -> NDArray:
        """Remove the simulation margin from a simulation-grid array."""
        s = self.oversample
        (ny, nx), (my, mx) = self.camera_shape, self.margin
        return a[..., my * s : (my + ny) * s, mx * s : (mx + nx) * s]


def _fast_margin(n: int, margin: int, s: int) -> int:
    while sfft.next_fast_len((n + 2 * margin) * s) != (n + 2 * margin) * s:
        margin += 1
    return margin


def make_grid(
    camera: Camera,
    optics: Optics,
    wavelength: float,
    oversample: int | None = None,
    margin_px: int | None = None,
) -> Grid:
    cutoff_px = optics.coherent_cutoff(wavelength) * camera.pixel_pitch  # cycles/pixel
    if oversample is None:
        oversample = max(1, math.ceil(cutoff_px / _MAX_CUTOFF - _EPS))
    elif cutoff_px / oversample > _MAX_CUTOFF + _EPS:
        need = math.ceil(cutoff_px / _MAX_CUTOFF - _EPS)
        raise ValueError(
            f"oversample={oversample} aliases the speckle intensity "
            f"(pupil cutoff {cutoff_px:.3f} cycles/pixel); use oversample >= {need}"
        )
    if margin_px is None:
        margin_px = max(4, math.ceil(4 * optics.speckle_size(wavelength) / camera.pixel_pitch))
    margin = tuple(_fast_margin(n, margin_px, oversample) for n in camera.shape)
    return Grid(
        camera_shape=tuple(camera.shape),
        oversample=oversample,
        margin=margin,
        pixel_pitch=camera.pixel_pitch,
        magnification=optics.magnification,
        cutoff_cycles_per_sample=cutoff_px / oversample,
    )


def circular_pupil(shape: tuple[int, int], cutoff_cycles_per_sample: float) -> NDArray[np.bool_]:
    """Binary circular pupil in unshifted FFT frequency order."""
    fy = sfft.fftfreq(shape[0])[:, None]
    fx = sfft.fftfreq(shape[1])[None, :]
    return fx**2 + fy**2 <= cutoff_cycles_per_sample**2


def rough_surface(shape: tuple[int, int], rng: np.random.Generator) -> NDArray[np.complex128]:
    """Unit-modulus, delta-correlated random phase: a surface much rougher than lambda."""
    return np.exp(2j * np.pi * rng.random(shape))


def warp(
    field: NDArray[np.complex128], d_rows: NDArray[np.float64], d_cols: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """Move the microstructure by (d_rows, d_cols) samples: ``out(x) = field(x - d(x))``.

    Periodic cubic-spline interpolation. Accurate to about 1-2 % for content inside a pupil
    of <= 0.25 cycles/sample, and exact for integer shifts.
    """
    rows, cols = np.indices(field.shape, dtype=float)
    coords = np.array([rows - d_rows, cols - d_cols])

    def interp(a):
        return map_coordinates(a, coords, order=3, mode="grid-wrap")

    return interp(field.real) + 1j * interp(field.imag)


def image_field(
    object_field: NDArray[np.complex128], pupil: NDArray[np.bool_]
) -> NDArray[np.complex128]:
    """Coherent imaging through the pupil: ``IFFT(FFT(U) * P)``."""
    return sfft.ifft2(sfft.fft2(object_field, workers=-1) * pupil, workers=-1)
