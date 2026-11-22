# Time-Averaged ESPI of a Vibrating Drumhead — Design (M2)

Date: 2026-11-21 · Status: implemented (approved 2026-11-21) · Builds on
[`2026-11-21-espi-simulator-design.md`](2026-11-21-espi-simulator-design.md) (M1)

## Intent

Simulate the lab experiment the user runs:
- a drumhead driven by a tone generator, either at a fixed tone or swept across resonances during one recording;
- an out-of-plane ESPI camera recording **video** with **time-averaged** exposures, each
  spanning many vibration cycles;
- a **piezo mirror in the reference beam driven by a slow triangle wave**, so the reference
  phase ramps continuously across frames.

The simulator must output the video plus exact ground truth (vibration amplitude and phase
maps), so that processing code that turns such videos into mode-shape and amplitude maps
can be developed and scored.

## Goals

1. **Drumhead vibration model.** An ideal circular membrane, clamped at radius `a`. Modes
   `(m, n)` have shape `J_m(α_mn r/a)·cos(m(θ − θ_d))` and frequency
   `f_mn = f01 · α_mn / α_01`, where `α_mn` is the n-th zero of `J_m` and `f01` is a
   parameter. The steady-state complex response to a harmonic drive at frequency `f` is
   the sum over modes of `F_mn / (ω_mn² − ω² + 2iζ ω_mn ω) · φ_mn / N_mn`, with one modal
   damping `ζ = 1/(2Q)` for all modes.
   - **Drive options:**
     - *point force* at `(r_d, θ_d)`: `F_mn = J_m(α_mn r_d / a)`;
     - *uniform pressure* (a speaker under the head): only `m = 0` modes,
       `F_0n ∝ ∫ J_0(α r/a) dA`.
   - **Scaling: fixed drive strength (user decision).** The amplitude follows the
     resonance curve, as in the lab, so a tone far from resonance gives few fringes and a
     tone on resonance gives dense ones.
     - The modal sum is made dimensionless by multiplying by `ω01²`.
     - A `drive_level` in metres scales it.
     - `drumhead.calibrate(peak_amplitude, at_hz)` returns the `drive_level` that gives
       `max|W| = peak_amplitude` at frequency `at_hz`.
   - **Outside the head** (`r ≥ a`, the rim and surroundings) the surface is static and
     still rough. It provides an in-frame "no vibration" reference region.
2. **Tone generator: a drive schedule within one recording (user decision).**
   `tone_at(t) -> (frequency_hz, drive_level)`, with three implementations:
   - `Tone`: constant;
   - `SteppedSweep`: a list of frequencies and/or levels, each held for a dwell time, which
     allows drive on/off;
   - `Sweep`: continuous from `f_start` to `f_stop` over a duration, linear or log
     spacing.

   Each frame uses the tone at its mid-exposure. Vibration states are cached by
   `(frequency, level)`, so constant and stepped segments cost one state computation each.
   A continuous sweep costs one state per frame (about `N_t + 1` FFTs; measured about 0.6 s per
   512² frame on an M5 Max, so a 900-frame sweep takes about 9 minutes).

   Steady state is assumed. Ring-up transients are not modelled, and a `UserWarning` fires
   when the sweep rate exceeds `0.25·(f/Q)/τ` with `τ = Q/(πf)`. For the defaults that is
   about 50 Hz/s.
3. **Reference piezo waveform.** A triangle wave of frequency `f_pzt`, peak-to-peak phase
   `Δ_pp`, offset, start phase and the same quadratic nonlinearity as M1's
   `PhaseShifter`. There are real turnarounds at the peaks, and a helper converts mirror
   travel to phase (`4π·travel/λ`).
4. **Video timing.** Frame rate, exposure time and frame count. Frame `k` integrates over
   `[k/fps, k/fps + T_exp]`.
5. **Time averaging, cycle-averaged-fields method (chosen).** The vibration
   (≥ 100 Hz) is much faster than the piezo (≤ a few Hz), so the averages separate.
   Per exposure:
   `⟨I⟩ = ⟨|E_o|²⟩_cycle + |E_r|² + 2 Re(⟨E_o⟩_cycle · E_r* · γ · ⟨e^{−iδ(t)}⟩_exposure)`
   - `⟨E_o⟩_cycle` is computed exactly by linearity: it is the image of
     `U_o · J0(S·|W|)` (Jacobi–Anger, continuous time average). One FFT.
   - `⟨|E_o|²⟩_cycle` is a numerical average over `N_t` evenly spaced vibration phases,
     with `N_t ≥ max(32, ⌈max S·|W|⌉ + 16)` so that aliasing (about `J_{N_t}`) is
     negligible. This keeps the speckle smearing and decorrelation caused by the
     vibrating slope.
   - `⟨e^{−iδ}⟩` is averaged numerically over the exposure from the triangle waveform.
     This captures the integrating-bucket loss and the turnarounds.
   - Both cycle averages are computed once per vibration state, i.e. once per distinct
     `(frequency, level)` tone. The pixel-integrated
     (camera-resolution) base and the complex cross term are cached, so each frame costs
     `O(pixels)`.
   - Laser coherence and drift, relative intensity noise (RIN) and camera noise apply per
     frame, as in M1.
6. **Ground truth.** At pixel centres:
   - `amplitude = |W|` (m) and `phase = arg W` (rad);
   - the expected fringe-modulation factor `J0(S_z·|W|)` (signed);
   - the per-mode complex amplitudes;
   - per frame: timestamp, mean reference phase `arg⟨e^{iδ}⟩`, reference visibility
     `|⟨e^{iδ}⟩|`, tone frequency and drive level. Amplitude and phase maps are stored
     per distinct tone state, and each frame records the index of the state it used.
7. **Validation and honesty.**
   - **Partial cycles.** Raise a `UserWarning` when `f·T_exp < 10`: the exposure's partial
     vibration cycle is not modelled, and the error scales like `1/(f·T_exp)`.
   - **Setup restriction.** Raise `ValueError` for setups without a reference beam
     (dual-beam time averaging would need a non-separable cross term) and for vibrations
     with in-plane components.
8. **Analysis and example.** The existing least-squares `phase_from_steps`, fed each
   frame's reference phase, gives a modulation map proportional to `|J0|`. The example
   renders mode maps at drive frequencies near the (0,1), (1,1), (2,1) and (0,2)
   resonances, next to the `|J0|` ground truth.

## Non-goals (M2)

- **Partial-cycle exposure** (planned follow-up: Jacobi–Anger harmonics with `sinc`
  window weights).
- **Ring-up and ring-down transients** during fast sweeps or drive switching (steady state
  only; warned).
- **Stroboscopic and high-speed** capture.
- **Physical membrane effects:** air loading, inharmonic real-drum ratios, nonlinear
  large-amplitude behaviour, tension non-uniformity.
- **Other piezo behaviour:** hysteresis beyond the quadratic nonlinearity, and drive
  harmonics or multiple tones.
- **Dual-beam (in-plane) time averaging.**

## Anti-goals

- Producing the frames with the textbook `J0` formula alone. That would drop
  slope-induced speckle smearing and make the oracle too lenient.
- Silently accepting exposures that are too short for the cycle-average assumption.

## Approaches considered (user chose 2)

Separately, the user chose sweeps **within one recording** over one-tone-per-video, and
**fixed drive strength** so that amplitude follows the resonance curve.


1. **Brute-force time integration:** sum the field at about 32 instants per cycle over
   every exposure. Exact, including partial cycles, but thousands of FFTs per frame
   (minutes per video).
2. **Cycle-averaged fields (chosen):** about `N_t + 1` FFTs per vibration state, then
   `O(pixels)` per frame. Exact for steady-state vibration when exposures span many cycles.
3. **Textbook `J0` scaling:** instant, but no slope decorrelation.

## Defaults (builder's calls, documented in code)

- **Drumhead:** radius 0.1 m, `f01` = 400 Hz, Q = 50, point drive at `r = 0.4a`,
  `θ = 30°`. The example calibrates the drive to a 0.5 µm peak (about 4 dark Bessel
  rings) at `f01`.
- **Tone:** each frame uses the tone at its mid-exposure. The quasi-static warning
  threshold is `0.25·(f/Q)/τ`.
- **Video:** 30 fps, exposure 30 ms (12 cycles at 400 Hz).
- **Piezo:** triangle wave at 0.25 Hz, `Δ_pp = 4π` (mirror travel λ).
- **Optics:** magnification chosen in the example so the head plus rim fills the frame.

## Interfaces (sketch)

```python
drum = Drumhead(radius=0.1, f01=400, q=50, drive=PointDrive(r=0.04, theta_deg=30))
level = drum.calibrate(peak_amplitude=0.5e-6, at_hz=400)
tone = Sweep(f_start=300, f_stop=1000, duration_s=30, level=level, spacing="log")
cfg = SimConfig(..., reference_drive=TriangleWave(frequency_hz=0.25, phase_pp=4*pi),
                timing=VideoTiming(frame_rate_hz=30, exposure_s=0.03))
video = Simulator(cfg).acquire_video(drum, tone, n_frames=900)   # VideoAcquisition
fit = analysis.phase_from_steps(video.frames, video.reference_phase)  # fit.modulation ∝ |J0|
```

## Test strategy

- **Membrane physics:**
  - frequency ratios match the Bessel zeros;
  - the response is zero at the rim;
  - at the (0,1) resonance with high Q, the shape correlates with `J0(2.405 r/a)` above
    0.99;
  - an (m = 1) drive leaves a nodal diameter at `θ_d ± 90°`;
  - uniform pressure gives an axisymmetric response;
  - the phase lag is 0, then π/2 at resonance, then π as the drive frequency crosses it.
- **Time averaging:**
  - the cycle-averaged field equals the static field where `W = 0`;
  - the recovered modulation, normalised by the static rim, tracks `|J0(S·|W|)|` and
    dips at the J0 zeros (121 nm, 278 nm, …);
  - in vibrating regions the averaged intensity has speckle contrast ≤ 1.
- **Waveform:**
  - triangle values and turnarounds are correct;
  - for a linear ramp, the integrating-bucket visibility is `|sin(Δ/2)/(Δ/2)|`.
- **Sweeps:**
  - within one recording the amplitude peaks at the frames whose tone is nearest each
    resonance;
  - stepped segments reuse their cached state, so identical tones give identical
    noiseless electrons apart from the reference phase;
  - the sweep-rate warning fires.
- **Validation and plumbing:**
  - the short-exposure warning fires;
  - dual-beam and in-plane vibration raise errors;
  - same seed gives the same video, with correct shapes and timestamps.

## Implementation notes

- **Drumhead evaluation** uses cached per-azimuthal-order cubic-spline radial profiles
  (2049 samples, error about 1e-10), not per-mode Bessel calls at every point. That took
  the test suite from 95 s to 3.4 s.
- **Ground-truth check.** The cycle-averaged mean field is the image of `U·J0`, not
  `J0 × image(U)`. Where the J0 envelope varies over a speckle, the pixelwise modulation
  ratio deviates from `|J0|` by a zero-mean speckle term: about 0.07 at 96 px over the head,
  shrinking with resolution. The tests use a local least-squares complex estimator, which
  recovers the signed J0 to about 0.03.
- **Pixel integration of complex cross terms** routes the real and imaginary parts through
  the real filter. With an even oversample, the half-sample centre shift is not Hermitian
  at Nyquist.
- **Example** (`examples/drumhead_sweep.py`): 240 frames at 512² in about 4 s. The fitted
  modulation correlates 0.93–0.95 with `|J0|` truth under shot and read noise. The (2,1)
  pattern is a real mixed mode, because (0,2) is only 7 % higher in frequency at Q = 50.
