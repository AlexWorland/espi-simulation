"""Result types and steady-state checks for time-averaged video acquisition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .drumhead import Drumhead, MembraneMode
from .waveforms import SteppedSweep, VideoTiming

MIN_CYCLES_PER_EXPOSURE = 10.0
SWEEP_SAFETY = 0.25
RING_UP_DWELLS = 5.0


@dataclass(frozen=True)
class ToneState:
    """Ground truth for one distinct ``(frequency, level)`` drive state, at pixel centres."""

    frequency_hz: float
    level: float
    amplitude: NDArray[np.float64]
    """Out-of-plane vibration amplitude ``|W|`` (metres)."""
    phase: NDArray[np.float64]
    """Vibration phase ``arg W`` (rad), relative to the drive."""
    modulation: NDArray[np.float64]
    """Expected fringe-modulation factor ``J0(S_z |W|)`` (signed)."""
    modes: tuple[MembraneMode, ...]


@dataclass(frozen=True)
class VideoAcquisition:
    frames: NDArray[np.uint16]
    """Digitised frames, shape (n_frames, rows, cols)."""
    electrons: NDArray[np.float64]
    """Expected photoelectrons per pixel before shot/read noise and quantisation."""
    times: NDArray[np.float64]
    """Exposure start time of each frame (s), on the global timeline."""
    reference_phase: NDArray[np.float64]
    """``arg<exp(i delta)>`` over each exposure: the phase step to give phase fitting."""
    reference_visibility: NDArray[np.float64]
    """``|<exp(i delta)>|``: fringe-contrast loss from the reference moving during exposure."""
    frequency_hz: NDArray[np.float64]
    level: NDArray[np.float64]
    state_index: NDArray[np.int_]
    """Index into ``states`` for each frame."""
    states: tuple[ToneState, ...]
    sensitivity: NDArray[np.float64]
    frame_offset: int = 0
    metadata: dict = field(default_factory=dict)


def steady_state_warnings(
    tone, drum: Drumhead, timing: VideoTiming, mid_times: NDArray[np.float64]
) -> list[str]:
    """Reasons the steady-state, many-cycles-per-exposure model may not hold."""
    msgs: list[str] = []
    driven = []
    for t in mid_times:
        f, level = tone.tone_at(float(t))
        if level != 0:
            driven.append((float(t), f))
    if not driven:
        return msgs
    for t, f in driven:
        limit = SWEEP_SAFETY * np.pi * f**2 / drum.q**2
        rate = abs(tone.rate_at(t))
        if rate > limit:
            msgs.append(
                f"sweep rate {rate:.0f} Hz/s exceeds the steady-state limit {limit:.0f} Hz/s "
                f"at {f:.0f} Hz (Q={drum.q:g}); ring-up transients are not modelled"
            )
            break
    f_min = min(f for _, f in driven)
    if isinstance(tone, SteppedSweep):
        ring_up = drum.q / (np.pi * f_min)
        if tone.dwell_s < RING_UP_DWELLS * ring_up:
            msgs.append(
                f"dwell {tone.dwell_s:.3g} s is under {RING_UP_DWELLS:g} ring-up times "
                f"({ring_up:.3g} s); the steady-state response is assumed immediately"
            )
    cycles = f_min * timing.exposure_s
    if cycles < MIN_CYCLES_PER_EXPOSURE:
        msgs.append(
            f"exposure spans only {cycles:.1f} vibration cycles (< {MIN_CYCLES_PER_EXPOSURE:g}); "
            "the leftover partial cycle is not modelled"
        )
    return msgs
