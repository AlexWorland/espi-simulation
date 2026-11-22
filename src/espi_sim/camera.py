"""Camera model: pixel-aperture integration, then shot noise, read noise and the ADC.

The pixel's light-sensitive area is a centred square of side ``sqrt(fill_factor) * pitch``.
Integration over it is a box filter. Because the speckle intensity is band-limited on the
simulation grid, the filter (``sinc`` MTF) and the half-sample shift to the pixel centre are
applied exactly in the Fourier domain before decimating to one sample per pixel.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from numpy.typing import NDArray

from .config import Camera
from .optics import Grid


def integrate_pixels(
    intensity: NDArray[np.float64], grid: Grid, fill_factor: float = 1.0
) -> NDArray[np.float64]:
    """Mean electrons collected per pixel from a simulation-grid intensity (with margin).

    ``intensity`` is in electrons per pixel area for a 100 % fill factor, so a uniform
    intensity ``I`` yields ``fill_factor * I``.
    """
    s = grid.oversample
    ny, nx = intensity.shape[-2:]
    fy = sfft.fftfreq(ny)[:, None]
    fx = sfft.fftfreq(nx)[None, :]
    width = np.sqrt(fill_factor) * s  # active aperture side, in samples
    centre = (s - 1) / 2  # pixel-centre offset from the pixel's first sample
    transfer = np.sinc(width * fy) * np.sinc(width * fx) * np.exp(2j * np.pi * centre * (fx + fy))
    filtered = sfft.ifft2(sfft.fft2(intensity, workers=-1) * transfer, workers=-1).real
    return fill_factor * grid.crop(filtered)[..., ::s, ::s]


def expose(
    electrons: NDArray[np.float64], camera: Camera, rng: np.random.Generator
) -> NDArray[np.uint16]:
    """Turn expected photoelectrons into a digitised frame (ADU, uint16)."""
    e = np.clip(np.asarray(electrons, float), 0.0, None)
    if camera.shot_noise:
        e = rng.poisson(e).astype(float)
    e = np.minimum(e, camera.full_well)
    if camera.read_noise:
        e = e + rng.normal(0.0, camera.read_noise, e.shape)
    adu = np.rint(e / camera.electrons_per_adu) + camera.offset_adu
    return np.clip(adu, 0, camera.max_adu).astype(np.uint16)
