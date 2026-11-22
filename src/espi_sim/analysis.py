"""Reference ESPI processing algorithms, used to validate the simulator end to end.

They are deliberately simple and textbook, a baseline for the real processing code to
beat. Every phase-stepping function uses the convention ``I_k = A + B cos(phi - delta_k)``,
which is what :mod:`espi_sim.simulator` produces.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import uniform_filter


class PhaseResult(NamedTuple):
    phase: NDArray[np.float64]
    """Wrapped phase in (-pi, pi]."""
    modulation: NDArray[np.float64]
    """Fringe amplitude B. Low values mark unreliable (dark or decorrelated) speckles."""
    background: NDArray[np.float64]
    """Mean intensity A."""


def wrap(phase: ArrayLike) -> NDArray[np.float64]:
    """Wrap to (-pi, pi]."""
    p = np.angle(np.exp(1j * np.asarray(phase, float)))
    return np.where(p == -np.pi, np.pi, p)


def phase_from_steps(frames: ArrayLike, steps: ArrayLike) -> PhaseResult:
    """Least-squares phase from three or more frames with known phase steps.

    Fits ``I_k = a + b cos(delta_k) + c sin(delta_k)`` per pixel, giving
    ``phi = atan2(c, b)``. For four steps of pi/2 this reduces to the classic four-step
    formula.
    """
    frames = np.asarray(frames, float)
    steps = np.asarray(steps, float)
    if steps.shape != frames.shape[:1] or len(steps) < 3:
        raise ValueError("need at least three frames, one phase step per frame")
    design = np.column_stack([np.ones_like(steps), np.cos(steps), np.sin(steps)])
    coef = np.linalg.pinv(design) @ frames.reshape(len(steps), -1)
    a, b, c = (x.reshape(frames.shape[1:]) for x in coef)
    return PhaseResult(np.arctan2(c, b), np.hypot(b, c), a)


def carrier_phase(
    frame: ArrayLike, carrier: tuple[float, float], radius: float
) -> NDArray[np.float64]:
    """Single-frame Fourier-transform (Takeda) phase for a spatial-carrier interferogram.

    ``frame = A + B cos(phi - 2 pi (fx*col + fy*row))``, where ``carrier = (fx, fy)`` is in
    cycles per pixel, matching a tilted reference in :class:`espi_sim.config.ReferenceBeam`.
    The side lobe at ``-carrier`` is isolated with a disk of ``radius`` cycles/pixel,
    inverse-transformed and demodulated.
    """
    f = np.asarray(frame, float)
    ny, nx = f.shape
    fy = np.fft.fftfreq(ny)[:, None]
    fx = np.fft.fftfreq(nx)[None, :]
    lobe = (fx + carrier[0]) ** 2 + (fy + carrier[1]) ** 2 <= radius**2
    side = np.fft.ifft2(np.fft.fft2(f) * lobe)
    rows, cols = np.indices(f.shape)
    return wrap(np.angle(side * np.exp(2j * np.pi * (carrier[0] * cols + carrier[1] * rows))))


def phase_difference(after: ArrayLike, before: ArrayLike) -> NDArray[np.float64]:
    """Wrapped phase change between two states."""
    return wrap(np.asarray(after) - np.asarray(before))


def subtraction_fringes(before: ArrayLike, after: ArrayLike) -> NDArray[np.float64]:
    """Classic real-time ESPI correlation fringes ``|I_before - I_after|``."""
    return np.abs(np.asarray(after, float) - np.asarray(before, float))


def sincos_filter(phase: ArrayLike, size: int = 5, iterations: int = 1) -> NDArray[np.float64]:
    """Mean-filter sin and cos of a wrapped phase map, which avoids smearing the 2pi jumps."""
    p = np.asarray(phase, float)
    for _ in range(iterations):
        s = uniform_filter(np.sin(p), size, mode="nearest")
        c = uniform_filter(np.cos(p), size, mode="nearest")
        p = np.arctan2(s, c)
    return p


def unwrap(phase: ArrayLike, mask: ArrayLike | None = None) -> NDArray[np.float64]:
    """2-D phase unwrapping (Herraez et al. 2002, via scikit-image).

    Requires the ``analysis`` extra. ``mask`` marks pixels to exclude (True = invalid).
    """
    try:
        from skimage.restoration import unwrap_phase
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("unwrap() needs scikit-image: install espi-sim[analysis]") from exc
    p = np.asarray(phase, float)
    if mask is not None:
        p = np.ma.masked_array(p, mask=np.asarray(mask, bool))
    out = unwrap_phase(p)
    return np.ma.getdata(out) if mask is None else out
