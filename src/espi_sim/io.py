"""Save and load acquisitions as ``.npz``, readable from any language with an npz reader."""

from __future__ import annotations

import dataclasses
import json
from os import PathLike

import numpy as np

from .config import SimConfig
from .simulator import Acquisition, GroundTruth

_FORMAT = "espi-sim/acquisition/1"


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
