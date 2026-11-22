"""Time-averaged ESPI of a drumhead stepped through its first four resonances in one recording.

A slow triangle wave drives the reference piezo. Each tone is held for one half-period
(2 s = 60 frames), so the reference phase sweeps 4 pi through every tone. Fitting each dwell
with the known reference phases gives a fringe-modulation map proportional to
|J0(4 pi A / lambda)|: bright nodal lines, dark rings where A = 121 nm, 278 nm, ...

    uv run python examples/drumhead_sweep.py [output.png]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter

import espi_sim
from espi_sim import analysis as an

MODES = [(0, 1), (1, 1), (2, 1), (0, 2)]


def main(out: str = "outputs/drumhead_modes.png") -> None:
    n_px = 512
    drum = espi_sim.Drumhead(drive=espi_sim.PointDrive(r=0.025, theta_deg=30.0))
    cfg = espi_sim.SimConfig(
        camera=espi_sim.Camera(shape=(n_px, n_px)),  # 12-bit, shot + read noise
        optics=espi_sim.Optics(f_number=8, magnification=n_px * 3.45e-6 / (2.2 * drum.radius)),
        reference_drive=espi_sim.TriangleWave(frequency_hz=0.25, phase_pp=4 * np.pi),
        timing=espi_sim.VideoTiming(frame_rate_hz=30.0, exposure_s=0.03),
        seed=2,
    )
    freqs = tuple(drum.mode_frequency(m, n) for m, n in MODES)
    # Turn the tone generator up or down per step so every mode peaks near 0.5 um.
    levels = tuple(drum.calibrate(0.5e-6, at_hz=f) for f in freqs)
    tone = espi_sim.SteppedSweep(frequencies_hz=freqs, dwell_s=2.0, levels=levels)

    t = time.perf_counter()
    video = espi_sim.Simulator(cfg).acquire_video(drum, tone, n_frames=4 * 60)
    print(
        f"simulated {len(video.frames)} frames of {n_px}x{n_px} in {time.perf_counter() - t:.1f} s"
    )

    sim_grid = espi_sim.Simulator(cfg).grid
    xi, eta = sim_grid.pixel_object_coords()
    r = np.hypot(xi, eta)
    inside, rim = r < 0.95 * drum.radius, r > 1.05 * drum.radius

    fig, axes = plt.subplots(3, 4, figsize=(15, 11.5), constrained_layout=True)
    for s, ((m, n), state) in enumerate(zip(MODES, video.states, strict=True)):
        sel = video.state_index == s
        fit = an.phase_from_steps(video.frames[sel], video.reference_phase[sel])
        modulation = uniform_filter(fit.modulation, 3) / np.median(fit.modulation[rim])
        truth = np.abs(state.modulation)
        corr = np.corrcoef(modulation[inside], truth[inside])[0, 1]
        peak_nm = state.amplitude.max() * 1e9
        print(
            f"mode ({m},{n}) at {state.frequency_hz:6.1f} Hz: peak {peak_nm:5.0f} nm,"
            f" corr(modulation, |J0|) = {corr:.3f}"
        )
        axes[0, s].imshow(video.frames[np.flatnonzero(sel)[0]], cmap="gray")
        axes[0, s].set_title(f"raw frame, ({m},{n}) @ {state.frequency_hz:.0f} Hz")
        axes[1, s].imshow(modulation, cmap="gray", vmin=0, vmax=1.2)
        axes[1, s].set_title("fitted fringe modulation")
        axes[2, s].imshow(truth, cmap="gray", vmin=0, vmax=1.2)
        axes[2, s].set_title("ground truth |J0(4πA/λ)|")
    for ax in axes.flat:
        ax.set_axis_off()
    fig.suptitle("Simulated time-averaged HeNe ESPI: drumhead stepped through four resonances")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
