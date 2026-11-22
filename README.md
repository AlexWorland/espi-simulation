<img src="assets/icon.svg" width="64" height="64" alt="">

# ESPI Simulation

A physically grounded simulator for **Electronic Speckle Pattern Interferometry**, built to
develop and test ESPI processing code without an optical bench. Every simulated frame comes
with exact ground truth (displacement and phase change at each pixel), so phase extraction,
filtering, unwrapping and strain recovery can be scored quantitatively.

Defaults model a **HeNe laser (632.8 nm)** and a 12-bit machine-vision camera.

![Simulated bending plate](docs/images/bending_plate.png)

## Quick start

```bash
uv sync
uv run pytest                              # 70 physics/unit tests, < 1 s
uv run python examples/bending_plate.py    # writes outputs/bending_plate.png
```

```python
import espi_sim
from espi_sim import analysis as an
from espi_sim.displacement import ClampedPlatePressure

sim = espi_sim.Simulator(espi_sim.SimConfig(camera=espi_sim.Camera(shape=(512, 512)), seed=1))
before, after = sim.acquire_pair(ClampedPlatePressure(radius=7e-3, center_deflection=1.5e-6))

pb = an.phase_from_steps(before.frames, before.phase_steps)   # 4-step PZT by default
pa = an.phase_from_steps(after.frames, after.phase_steps)
recovered = an.unwrap(an.sincos_filter(an.phase_difference(pa.phase, pb.phase), size=7))
error = recovered - after.truth.phase                          # score against ground truth
```

## What it models

| Area | Model |
|---|---|
| Speckle | Rough surface → coherent 4f imaging through a circular pupil. Speckle size `1.22 λ (1+M) F#`, fully developed statistics. |
| Decorrelation | Emerges from the physics. In-plane motion moves the microstructure, and tilt or strain shifts the spectrum within the pupil. |
| Geometry | Out-of-plane (smooth reference) and dual-beam in-plane. Sensitivity vector `(2π/λ)(k̂_ill − k̂_obs)`. |
| Ground truth | Rigid translation/rotation, clamped circular plate (pressure, point load), uniform/thermal strain. Fields superpose with `+`. |
| Phase shifting | PZT steps with miscalibration, quadratic nonlinearity and per-frame vibration jitter. Spatial carrier via a tilted reference. |
| HeNe laser | Single-frequency, or multimode comb (`c/2L` spacing) under a Doppler gain curve: visibility vs path mismatch, mode-sweep phase drift, intensity noise. |
| Camera | Exact pixel-aperture (fill factor) integration, Poisson shot noise, read noise, full-well saturation, gain, bit depth. |
| Determinism | One integer seed, independent named streams per noise source. |

Output: `Acquisition` objects (frames, noiseless expected electrons, actual phase steps,
ground truth). `save_acquisition` writes them as `.npz` for processing code in any
language.

Design and rationale: [`docs/superpowers/specs/2026-11-21-espi-simulator-design.md`](docs/superpowers/specs/2026-11-21-espi-simulator-design.md).
