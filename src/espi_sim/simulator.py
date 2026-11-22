"""Orchestrates one ESPI acquisition: surface → imaging → interference → camera.

Per object state, the speckle field of each beam is computed once (warp + 2 FFTs). Each
phase-stepped frame then only recombines the two interferometer arms:

    I = |E0|^2 + |E1|^2 + 2 Re(E0 E1* gamma(L0 - L1, frame) exp(-i delta))

where ``gamma`` is the laser's complex coherence for the arms' path mismatch (which
includes HeNe mode-sweep drift), and ``delta`` is the actual PZT step applied to arm 1.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray
from scipy.special import j0

from .camera import expose, integrate_pixels
from .config import SimConfig
from .displacement import DisplacementField, Zero
from .optics import circular_pupil, image_field, make_grid, rough_surface, warp
from .rng import stream
from .time_average import cycle_averaged_fields, exposure_average
from .video import ToneState, VideoAcquisition, steady_state_warnings


@dataclass(frozen=True)
class GroundTruth:
    """Exact displacement and interference-phase change at camera pixel centres."""

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    w: NDArray[np.float64]
    phase: NDArray[np.float64]
    """Unwrapped phase change (rad) of arm 0 relative to arm 1: ``sensitivity . d``."""
    sensitivity: NDArray[np.float64]
    """Sensitivity vector (rad/m)."""


@dataclass(frozen=True)
class Acquisition:
    """A phase-stepped sequence of frames of one object state, plus its ground truth."""

    frames: NDArray[np.uint16]
    """Digitised frames, shape (n_steps, rows, cols)."""
    electrons: NDArray[np.float64]
    """Expected photoelectrons per pixel before shot/read noise and quantisation."""
    phase_steps: NDArray[np.float64]
    """Actual phase steps applied (including miscalibration and jitter)."""
    nominal_steps: NDArray[np.float64]
    truth: GroundTruth
    frame_offset: int = 0
    label: str = "state"
    metadata: dict = field(default_factory=dict)


class Simulator:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        c = self.config
        self.wavelength = c.laser.wavelength
        self.grid = make_grid(c.camera, c.optics, self.wavelength, c.oversample, c.margin_px)
        self.pupil = circular_pupil(self.grid.shape, self.grid.cutoff_cycles_per_sample)
        self._pupil_fraction = float(self.pupil.mean())
        self._xi, self._eta = self.grid.object_coords()
        self._surfaces: dict[int, NDArray[np.complex128]] = {}

    # -- fields --------------------------------------------------------------------------

    def _surface(self, beam_index: int) -> NDArray[np.complex128]:
        if beam_index not in self._surfaces:
            rng = stream(self.config.seed, "surface", beam_index)
            self._surfaces[beam_index] = rough_surface(self.grid.shape, rng)
        return self._surfaces[beam_index]

    def image_fields(self, displacement: DisplacementField) -> list[NDArray[np.complex128]]:
        """Image-plane speckle field of each object beam on the simulation grid (with margin).

        Scaled so that ``mean(|E|^2)`` equals the beam's ``mean_electrons``.
        """
        d = displacement(self._xi, self._eta)
        g = self.grid
        to_samples = g.magnification / g.dx
        fields = []
        for j, beam in enumerate(self.config.setup.beams):
            r = self._surface(j)
            if displacement.has_in_plane:  # the microstructure moves with the material
                r = warp(r, -d.v * to_samples, d.u * to_samples)
            s = beam.sensitivity(self.wavelength)
            phase = s[0] * d.u + s[1] * d.v + s[2] * d.w
            amplitude = np.sqrt(beam.mean_electrons / self._pupil_fraction)
            fields.append(image_field(amplitude * r * np.exp(1j * phase), self.pupil))
        return fields

    @cached_property
    def _reference_field(self) -> NDArray[np.complex128]:
        ref = self.config.setup.reference
        g = self.grid
        s = g.oversample
        rows, cols = np.indices(g.shape, dtype=float)
        # Camera-pixel coordinates, so that pixel centres fall on integers.
        col_px = (cols - g.margin[1] * s - (s - 1) / 2) / s
        row_px = (rows - g.margin[0] * s - (s - 1) / 2) / s
        carrier_phase = 2 * np.pi * (ref.carrier[0] * col_px + ref.carrier[1] * row_px)
        return np.sqrt(ref.mean_electrons) * np.exp(1j * carrier_phase)

    def _arms(self, displacement):
        setup = self.config.setup
        fields = self.image_fields(displacement)
        if setup.reference is not None:
            return (
                fields[0],
                self._reference_field,
                setup.beams[0].path_length - setup.reference.path_length,
            )
        return fields[0], fields[1], setup.beams[0].path_length - setup.beams[1].path_length

    # -- ground truth ----------------------------------------------------------------------

    def truth(self, displacement: DisplacementField) -> GroundTruth:
        xi, eta = self.grid.pixel_object_coords()
        d = displacement(xi, eta)
        s = self.config.setup.sensitivity(self.wavelength)
        return GroundTruth(d.u, d.v, d.w, s[0] * d.u + s[1] * d.v + s[2] * d.w, s)

    # -- acquisition -----------------------------------------------------------------------

    def acquire(
        self,
        displacement: DisplacementField | None = None,
        *,
        frame_offset: int = 0,
        label: str = "state",
    ) -> Acquisition:
        """Capture one phase-stepped sequence of the object in ``displacement``'s state.

        ``frame_offset`` is the index of the first frame in the experiment's timeline. It
        selects the laser, PZT-jitter and camera-noise draws, so give distinct sequences
        distinct offsets.
        """
        displacement = displacement or Zero()
        c = self.config
        nominal = np.asarray(c.phase_shifter.steps, float)
        steps = c.phase_shifter.actual_steps(stream(c.seed, "phase-shifter", frame_offset))
        e0, e1, opd = self._arms(displacement)
        base = np.abs(e0) ** 2 + np.abs(e1) ** 2
        cross = e0 * np.conj(e1)

        electrons, frames = [], []
        for k, delta in enumerate(steps):
            f = frame_offset + k
            gamma = c.laser.coherence(opd, frame=f)
            irradiance = base + 2.0 * np.real(cross * (gamma * np.exp(-1j * delta)))
            if c.laser.relative_intensity_noise:
                rin = stream(c.seed, "laser", f).normal()
                irradiance = irradiance * (1.0 + c.laser.relative_intensity_noise * rin)
            e = integrate_pixels(irradiance, self.grid, c.camera.fill_factor)
            electrons.append(e)
            frames.append(expose(e, c.camera, stream(c.seed, "camera", f)))

        return Acquisition(
            frames=np.stack(frames),
            electrons=np.stack(electrons),
            phase_steps=steps,
            nominal_steps=nominal,
            truth=self.truth(displacement),
            frame_offset=frame_offset,
            label=label,
            metadata={"oversample": self.grid.oversample, "margin": self.grid.margin},
        )

    def acquire_pair(self, displacement: DisplacementField) -> tuple[Acquisition, Acquisition]:
        """Reference (undeformed) sequence followed by the deformed sequence."""
        n = len(self.config.phase_shifter.steps)
        before = self.acquire(Zero(), frame_offset=0, label="before")
        after = self.acquire(displacement, frame_offset=n, label="after")
        return before, after

    # -- time-averaged video ---------------------------------------------------------------

    def acquire_video(self, drum, tone, n_frames: int, frame_offset: int = 0) -> VideoAcquisition:
        """Time-averaged video of a tone-driven drumhead with a slowly moving reference piezo.

        ``drum`` is a :class:`espi_sim.drumhead.Drumhead`. ``tone`` is a schedule from
        :mod:`espi_sim.waveforms` (``Tone``, ``SteppedSweep`` or ``Sweep``). Frame ``k``
        exposes over ``[(frame_offset + k)/fps, ... + exposure]`` and uses the tone at its
        mid-exposure. Cycle averages are computed once per distinct ``(frequency, level)``.
        """
        c = self.config
        setup = c.setup
        if setup.reference is None:
            raise ValueError("time-averaged video needs a reference-beam (out-of-plane) setup")
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1")
        timing = c.timing
        index = frame_offset + np.arange(n_frames)
        starts = index / timing.frame_rate_hz
        mids = starts + timing.exposure_s / 2
        tones = [tone.tone_at(float(t)) for t in mids]
        for msg in steady_state_warnings(tone, drum, timing, mids):
            warnings.warn(msg, UserWarning, stacklevel=2)

        s_vec = setup.sensitivity(self.wavelength)
        e_r = self._reference_field
        u = np.sqrt(setup.beams[0].mean_electrons / self._pupil_fraction) * self._surface(0)
        ff = c.camera.fill_factor
        px_xi, px_eta = self.grid.pixel_object_coords()

        cache: dict[tuple[float, float], int] = {}
        states: list[ToneState] = []
        terms: list[tuple[NDArray, NDArray]] = []
        state_index = np.empty(n_frames, int)
        for i, (f, level) in enumerate(tones):
            key = (float(f), float(level))
            if key not in cache:
                b = s_vec[2] * drum.complex_amplitude(self._xi, self._eta, f, level)
                gy, gx = np.gradient(b)
                if np.sqrt(np.abs(gx) ** 2 + np.abs(gy) ** 2).max() > np.pi / 2:
                    warnings.warn(
                        "vibration phase is under-sampled (> pi/2 rad per simulation sample); "
                        "reduce the amplitude or raise magnification/oversample",
                        UserWarning,
                        stacklevel=2,
                    )
                mean_field, mean_intensity = cycle_averaged_fields(u, b, self.pupil)
                base = integrate_pixels(mean_intensity + np.abs(e_r) ** 2, self.grid, ff)
                cross = integrate_pixels(mean_field * np.conj(e_r), self.grid, ff)
                w_px = drum.complex_amplitude(px_xi, px_eta, f, level)
                states.append(
                    ToneState(
                        frequency_hz=key[0],
                        level=key[1],
                        amplitude=np.abs(w_px),
                        phase=np.angle(w_px),
                        modulation=j0(s_vec[2] * np.abs(w_px)),
                        modes=drum.modes(f),
                    )
                )
                terms.append((base, cross))
                cache[key] = len(states) - 1
            state_index[i] = cache[key]

        opd = setup.beams[0].path_length - setup.reference.path_length
        drive = c.reference_drive
        electrons, frames, ref_avg = [], [], []
        for i, (k, t0) in enumerate(zip(index, starts, strict=True)):
            k = int(k)
            avg = exposure_average(drive.phase_at, t0, t0 + timing.exposure_s) if drive else 1 + 0j
            base, cross = terms[state_index[i]]
            gamma = c.laser.coherence(opd, frame=k)
            e = base + 2.0 * np.real(cross * (gamma * np.conj(avg)))
            if c.laser.relative_intensity_noise:
                e = e * (
                    1.0 + c.laser.relative_intensity_noise * stream(c.seed, "laser", k).normal()
                )
            electrons.append(e)
            frames.append(expose(e, c.camera, stream(c.seed, "camera", k)))
            ref_avg.append(avg)

        ref_avg = np.asarray(ref_avg)
        return VideoAcquisition(
            frames=np.stack(frames),
            electrons=np.stack(electrons),
            times=starts,
            reference_phase=np.angle(ref_avg),
            reference_visibility=np.abs(ref_avg),
            frequency_hz=np.array([f for f, _ in tones], float),
            level=np.array([lv for _, lv in tones], float),
            state_index=state_index,
            states=tuple(states),
            sensitivity=s_vec,
            frame_offset=frame_offset,
            metadata={
                "drumhead": dataclasses.asdict(drum),
                "tone": {"type": type(tone).__name__, **dataclasses.asdict(tone)},
                "oversample": self.grid.oversample,
                "margin": self.grid.margin,
            },
        )
