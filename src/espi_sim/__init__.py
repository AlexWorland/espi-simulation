"""ESPI simulator: physically grounded speckle interferometry frames with exact ground truth."""

from . import analysis, displacement
from .config import Camera, ObjectBeam, Optics, PhaseShifter, ReferenceBeam, Setup, SimConfig
from .drumhead import Drumhead, PointDrive, UniformPressure
from .io import load_acquisition, save_acquisition
from .laser import Laser
from .simulator import Acquisition, GroundTruth, Simulator
from .video import ToneState, VideoAcquisition
from .waveforms import SteppedSweep, Sweep, Tone, TriangleWave, VideoTiming

__all__ = [
    "Acquisition",
    "Camera",
    "Drumhead",
    "GroundTruth",
    "Laser",
    "ObjectBeam",
    "Optics",
    "PhaseShifter",
    "PointDrive",
    "ReferenceBeam",
    "Setup",
    "SimConfig",
    "Simulator",
    "SteppedSweep",
    "Sweep",
    "Tone",
    "ToneState",
    "TriangleWave",
    "UniformPressure",
    "VideoAcquisition",
    "VideoTiming",
    "analysis",
    "displacement",
    "load_acquisition",
    "save_acquisition",
]
