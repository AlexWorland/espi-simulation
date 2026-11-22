# ESPI Simulator — Milestone 1 Design

Date: 2026-11-21 · Status: approved to implement ("please begin implementing")

## Intent

Simulate Electronic Speckle Pattern Interferometry (ESPI) well enough to develop and
test processing and analysis code without an optical bench. The simulator is a **test
oracle**. Every frame comes with exact ground truth (displacement and phase change), so
phase extraction, filtering, unwrapping and strain recovery can be scored against it.
The target hardware is a **HeNe laser (632.8 nm)** with a machine-vision camera.

## Goals (M1)

- Physically grounded speckle: fully developed speckle formed by coherent imaging through
  a circular pupil. Speckle size comes from λ, F-number and magnification.
- Out-of-plane setup (smooth reference beam) and in-plane dual-beam setup. Sensitivity
  vector `S = (2π/λ)(k̂_ill − k̂_obs)`.
- Decorrelation emerges from the physics. In-plane displacement moves the surface
  microstructure, and local tilt moves the scattered spectrum within the pupil.
- Analytic displacement fields that give ground truth: rigid translation and rotation,
  a clamped circular plate under uniform pressure or a central point load, and uniform
  or thermal in-plane strain. Fields can be superposed.
- Temporal phase shifting (PZT) with step miscalibration, nonlinearity and vibration
  jitter. A spatial carrier is available through a tilted reference.
- HeNe laser model: single-frequency, or multimode with a longitudinal-mode comb under a
  Doppler gain curve. This gives coherence-limited visibility versus path mismatch,
  mode-sweep phase drift between frames, and relative intensity noise.
- Camera model: pixel pitch, fill factor (exact pixel-aperture MTF), shot noise, read
  noise, gain, bit depth, saturation.
- Deterministic: a single integer seed, with independent named random streams per noise
  source.
- Reference analysis functions (N-step phase, wrapped difference, subtraction fringes,
  sin/cos filter) to validate the simulator end to end.

## Non-goals (M1)

Diverging (point-source) illumination and non-telecentric viewing, so the sensitivity
vector is spatially uniform. Also out of scope: shearography, vibration/time-average
ESPI, pulsed addition mode, GPU, a GUI, and real-camera I/O formats beyond `.npz`.

## Anti-goals

- A simulator whose "truth" is the cheap formula `I = a + b·cos(ψ + Δφ)`. It can't
  decorrelate and would let processing code pass tests it should fail.
- Tests that compare exact random pixel values across NumPy versions.

## Approaches considered

1. **Phenomenological random-phase model**, `I = I_o + I_r + 2√(I_o I_r) cos(ψ+Δφ)`.
   Fast, but has no decorrelation, the wrong pixel-integration statistics, and no
   speckle-size physics. Kept only as a possible later fast path.
2. **Coherent 4f imaging of a rough-surface field (chosen).** A random-phase surface,
   warped by the in-plane displacement and multiplied by `exp(iΔφ)`, is passed through
   FFT → circular pupil → IFFT. Speckle statistics, size and decorrelation all follow from
   the physics. It costs about 2 FFTs per beam per state; on the M5 Max a 2048² FFT takes
   about 3 ms, so this is cheap.
3. **Fresnel/angular-spectrum propagation from surface to lens to sensor.** The most
   general option, but it needs much larger grids and adds nothing for a focused,
   telecentric imaging setup. Deferred.

Language: Python + NumPy/SciPy (agreed in conversation). Managed with `uv`, src layout,
pytest.

## Model

**Sampling.** The camera has `(ny, nx)` pixels at pitch `p`. The simulation grid
oversamples by an integer `s`, chosen so that the coherent cutoff
`f_c = 1 / (2 λ (1+M) F#)` is at most 0.25 cycles/sample. That keeps the intensity
`|E|²`, which has twice the bandwidth, alias-free. A margin of several speckle diameters
is added around the field of view and cropped after imaging, to suppress FFT wrap-around.

**Coordinates.** Object coordinates are image coordinates divided by `M`, with an erect
image and the origin at the field-of-view centre. The observation direction is telecentric
along +z. A beam's illumination propagation vector is `k̂_ill = −(sinθ cosφ, sinθ sinφ, cosθ)`.

**Object field per beam j.**
`U_j(ξ) = sqrt(I_j/η) · r_j(ξ − u_∥(ξ)) · exp(i·S_j·d(ξ))`, where `r_j` is a unit-modulus
random-phase field (one independent realization per beam) and `η` is the pupil power
fraction. Beams with large angular separation see uncorrelated speckle.

**Imaging.** `E_j = IFFT(FFT(U_j) · P)`. The reference beam is a smooth plane wave with
an optional carrier.

**Laser.** `I = Σ_m p_m |Σ_a E_a exp(i2π Δν_m L_a / c)|²`, where the sum runs over
longitudinal modes `m` and arms `a`. Each arm has a path length `L_a`.

**Phase step.** The step is applied as `exp(+iδ)` to the second arm (the reference, or
beam 1 in dual-beam). The measured phase is therefore `arm0 − arm1`, and the ground-truth
phase change is `(S_0 − S_1)·d`.

**Camera.** Box pixel aperture of side `√ff·p`, applied exactly in the Fourier domain and
then sampled at pixel centres. After that: Poisson shot noise, Gaussian read noise, gain,
rounding, and clipping to the bit depth.

## Module layout (`src/espi_sim/`)

`rng` (named streams) · `config` (dataclasses + HeNe defaults) · `laser` · `displacement` ·
`optics` (grid, pupil, warp, imaging) · `camera` · `simulator` (orchestration →
`Acquisition` with frames + truth) · `analysis` (reference processing) · `io` (npz).

## Test strategy

Physics invariants against analytic theory:
- speckle contrast ≈ 1 and exponential statistics
- intensity autocorrelation vs `|2J₁(x)/x|²`
- λ/4 rigid out-of-plane shift gives a π phase
- fringe counts for tilt and thermal strain match `S·d`
- a two-mode laser loses visibility at OPD = cavity length
- decorrelation vs in-plane shift follows the autocorrelation
- camera noise mean and variance
- seeded determinism and stream independence
- end to end: bending plate → 4-step → unwrap within tolerance of truth

## Left to the builder

Default parameter values (documented in code), numerical tolerances, file formats beyond
`.npz`.

## Implementation notes (M1)

- **Strain decorrelation is real and intended.** Under ±45° dual-beam illumination, a
  thermal strain of 4.6e-4 shifts each beam's spectrum by about 5.7 % of the pupil radius.
  The measured per-beam correlation (0.965) matches the pupil-overlap prediction (0.964),
  and the resulting phase noise (median about 0.24 rad) is unbiased. Tests encode the law
  itself rather than an arbitrary accuracy threshold.
- The spline microstructure warp is exact for integer shifts and accurate to about 1–2 %
  inside the pupil band for fractional shifts.
- Calls: camera defaults (3.45 µm pitch, 10.5 ke⁻ full well, 2.5 e⁻ read noise, 12-bit);
  beam defaults of 2500 e⁻ each (reference at about ¼ of saturation); auto margin of about
  4 speckle sizes, grown to an FFT-friendly size; image taken as erect (no inversion).

## Later milestones

Diverging illumination and spatially varying sensitivity; Gaussian beam profile;
shearography; vibration (time-average) ESPI; spatial phase shifting / multi-camera;
height-correlated rough surfaces (partially developed speckle); GPU backend if needed.
