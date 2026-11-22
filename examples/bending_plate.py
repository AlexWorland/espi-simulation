"""Simulate a clamped plate bending under pressure (HeNe, out-of-plane ESPI) and plot each
processing stage next to the ground truth.

    uv run python examples/bending_plate.py [output.png]
"""

from __future__ import annotations

import sys
import time

import matplotlib.pyplot as plt
import numpy as np

import espi_sim
from espi_sim import analysis as an
from espi_sim.displacement import ClampedPlatePressure, RigidRotation

LAM = 632.8e-9


def main(out: str = "outputs/bending_plate.png") -> None:
    cfg = espi_sim.SimConfig(camera=espi_sim.Camera(shape=(512, 512)), seed=1)
    sim = espi_sim.Simulator(cfg)
    fov = 512 * cfg.camera.pixel_pitch / cfg.optics.magnification
    # Plate bulges 2.5 wavelengths (10 fringes) plus a small rigid tilt.
    load = ClampedPlatePressure(radius=0.42 * fov, center_deflection=2.5 * LAM)
    load = load + RigidRotation(rx=1.5e-5)

    t = time.perf_counter()
    before, after = sim.acquire_pair(load)
    print(f"simulated 2x{len(before.frames)} frames of 512x512 in {time.perf_counter() - t:.2f} s")

    pb = an.phase_from_steps(before.frames, before.phase_steps)
    pa = an.phase_from_steps(after.frames, after.phase_steps)
    wrapped = an.phase_difference(pa.phase, pb.phase)
    filtered = an.sincos_filter(wrapped, size=7, iterations=2)
    unwrapped = an.unwrap(filtered)
    truth = after.truth.phase
    resid = unwrapped - truth
    resid -= np.median(resid)
    print(f"RMS error vs ground truth: {np.sqrt(np.mean(resid[8:-8, 8:-8] ** 2)):.3f} rad")

    panels = [
        (before.frames[0], "raw frame (speckle + reference)", "gray"),
        (an.subtraction_fringes(before.frames[0], after.frames[0]), "subtraction fringes", "gray"),
        (wrapped, "wrapped phase change (4-step)", "twilight"),
        (filtered, "sin/cos filtered", "twilight"),
        (unwrapped - np.median(unwrapped - truth), "unwrapped", "viridis"),
        (truth, "ground truth", "viridis"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)
    for ax, (img, title, cmap) in zip(axes.flat, panels, strict=True):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Simulated HeNe ESPI: clamped plate under pressure (out-of-plane)")
    from pathlib import Path

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
