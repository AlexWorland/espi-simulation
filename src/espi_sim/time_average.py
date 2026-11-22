"""Time averaging for time-averaged ESPI: over a vibration cycle and over a camera exposure.

The object vibrates much faster than the reference piezo moves, so the two averages
separate. Within one exposure:

    <I> = <|E_o|^2>_cycle + |E_r|^2 + 2 Re(<E_o>_cycle E_r* gamma <exp(-i delta)>_exposure)

``<E_o>_cycle`` is exact by linearity. The image of ``U * <exp(i Re(B e^{iwt}))>`` equals
the image of ``U * J0(|B|)`` (Jacobi-Anger), which is one FFT. ``<|E_o|^2>_cycle`` is
nonlinear, so it is averaged over ``N_t`` evenly spaced vibration phases. That keeps the
speckle smearing and decorrelation caused by the vibrating slope.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

from .optics import image_field


def cycle_samples(beta_max: float) -> int:
    """Vibration phases needed so that sampling aliasing (about ``J_N(beta)``) is negligible."""
    n = max(32, math.ceil(beta_max) + 16)
    return 4 * math.ceil(n / 4)


def cycle_averaged_fields(
    object_field: NDArray[np.complex128],
    phase_amplitude: NDArray[np.complex128],
    pupil: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Cycle-averaged image field and intensity for the object phase ``Re(B e^{iwt})``.

    ``object_field`` is the static object-plane field ``U`` (amplitude and rough surface).
    ``phase_amplitude`` is ``B``, the complex vibration phase amplitude in radians.
    """
    b = np.asarray(phase_amplitude)
    mean_field = image_field(object_field * j0(np.abs(b)), pupil)
    n = cycle_samples(float(np.abs(b).max(initial=0.0)))
    mean_intensity = np.zeros(object_field.shape)
    for j in range(n):
        phase = np.real(b * np.exp(2j * np.pi * j / n))
        mean_intensity += np.abs(image_field(object_field * np.exp(1j * phase), pupil)) ** 2
    return mean_field, mean_intensity / n


def exposure_average(
    phase_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    t0: float,
    t1: float,
    n: int = 64,
) -> complex:
    """Midpoint-rule mean of ``exp(i delta(t))`` over the exposure ``[t0, t1]``."""
    t = t0 + (np.arange(n) + 0.5) * (t1 - t0) / n
    return complex(np.mean(np.exp(1j * np.asarray(phase_fn(t), float))))
