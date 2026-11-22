"""Save and load acquisitions as ``.npz``, readable from any language with an npz reader."""

from __future__ import annotations

import dataclasses
import json
from os import PathLike

import numpy as np

from .config import SimConfig
from .drumhead import MembraneMode
from .simulator import Acquisition, GroundTruth
from .video import ToneState, VideoAcquisition

_FORMAT = "espi-sim/acquisition/1"
_VIDEO_FORMAT = "espi-sim/video/1"


def save_acquisition(
    path: str | PathLike, acquisition: Acquisition, config: SimConfig | None = None
) -> None:
    """Write frames, expected electrons, phase steps, ground truth and (optionally) config."""
    t = acquisition.truth
    meta = {
        "format": _FORMAT,
        "label": acquisition.label,
        "frame_offset": acquisition.frame_offset,
        "metadata": acquisition.metadata,
        "config": dataclasses.asdict(config) if config is not None else None,
    }
    np.savez_compressed(
        path,
        frames=acquisition.frames,
        electrons=acquisition.electrons,
        phase_steps=acquisition.phase_steps,
        nominal_steps=acquisition.nominal_steps,
        truth_u=t.u,
        truth_v=t.v,
        truth_w=t.w,
        truth_phase=t.phase,
        truth_sensitivity=t.sensitivity,
        meta_json=np.array(json.dumps(meta, default=float)),
    )


def load_acquisition(path: str | PathLike) -> tuple[Acquisition, dict | None]:
    """Read a file written by :func:`save_acquisition`. Returns (acquisition, config dict)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        if meta.get("format") != _FORMAT:
            raise ValueError(f"{path} is not an {_FORMAT} file")
        truth = GroundTruth(
            z["truth_u"], z["truth_v"], z["truth_w"], z["truth_phase"], z["truth_sensitivity"]
        )
        acq = Acquisition(
            frames=z["frames"],
            electrons=z["electrons"],
            phase_steps=z["phase_steps"],
            nominal_steps=z["nominal_steps"],
            truth=truth,
            frame_offset=int(meta["frame_offset"]),
            label=meta["label"],
            metadata=meta["metadata"],
        )
    return acq, meta["config"]


def save_video(
    path: str | PathLike, video: VideoAcquisition, config: SimConfig | None = None
) -> None:
    """Write a time-averaged video, its per-frame timeline and per-tone ground truth."""
    meta = {
        "format": _VIDEO_FORMAT,
        "frame_offset": video.frame_offset,
        "metadata": video.metadata,
        "config": dataclasses.asdict(config) if config is not None else None,
        "modes": [
            [
                [md.m, md.n, md.alpha, md.frequency_hz, md.amplitude.real, md.amplitude.imag]
                for md in st.modes
            ]
            for st in video.states
        ],
    }
    np.savez_compressed(
        path,
        frames=video.frames,
        electrons=video.electrons,
        times=video.times,
        reference_phase=video.reference_phase,
        reference_visibility=video.reference_visibility,
        frequency_hz=video.frequency_hz,
        level=video.level,
        state_index=video.state_index,
        sensitivity=video.sensitivity,
        state_frequency_hz=np.array([st.frequency_hz for st in video.states]),
        state_level=np.array([st.level for st in video.states]),
        state_amplitude=np.stack([st.amplitude for st in video.states]),
        state_phase=np.stack([st.phase for st in video.states]),
        state_modulation=np.stack([st.modulation for st in video.states]),
        meta_json=np.array(json.dumps(meta, default=float)),
    )


def load_video(path: str | PathLike) -> tuple[VideoAcquisition, dict | None]:
    """Read a file written by :func:`save_video`. Returns (video, config dict)."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        if meta.get("format") != _VIDEO_FORMAT:
            raise ValueError(f"{path} is not an {_VIDEO_FORMAT} file")
        states = tuple(
            ToneState(
                frequency_hz=float(z["state_frequency_hz"][i]),
                level=float(z["state_level"][i]),
                amplitude=z["state_amplitude"][i],
                phase=z["state_phase"][i],
                modulation=z["state_modulation"][i],
                modes=tuple(
                    MembraneMode(int(m), int(n), a, f, complex(re, im))
                    for m, n, a, f, re, im in meta["modes"][i]
                ),
            )
            for i in range(len(z["state_frequency_hz"]))
        )
        video = VideoAcquisition(
            frames=z["frames"],
            electrons=z["electrons"],
            times=z["times"],
            reference_phase=z["reference_phase"],
            reference_visibility=z["reference_visibility"],
            frequency_hz=z["frequency_hz"],
            level=z["level"],
            state_index=z["state_index"],
            states=states,
            sensitivity=z["sensitivity"],
            frame_offset=int(meta["frame_offset"]),
            metadata=meta["metadata"],
        )
    return video, meta["config"]
