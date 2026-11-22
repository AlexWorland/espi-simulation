# Time-Averaged Drumhead ESPI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate time-averaged ESPI video of a tone-driven (optionally swept) drumhead, with
a slow triangle-wave piezo in the reference arm, plus exact ground truth.

**Architecture:**
- A membrane modal model (`drumhead.py`) gives the complex out-of-plane amplitude
  `W(x; f, level)`.
- Drive and timing schedules (`waveforms.py`) give each frame its tone and its reference
  piezo phase.
- Per distinct tone, `time_average.py` computes the cycle-averaged object field (one FFT,
  using `J0`) and the cycle-averaged intensity (`N_t` FFTs). Pixel integration is cached.
- `Simulator.acquire_video` then forms each frame in O(pixels) and returns a
  `VideoAcquisition` (`video.py`).

**Tech Stack:** Python ≥ 3.12, NumPy, SciPy (`scipy.special.jv/j0/jn_zeros`, `scipy.fft`),
pytest, ruff. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-11-21-time-averaged-drumhead-design.md` (builds on
`docs/superpowers/specs/2026-11-21-espi-simulator-design.md`).

## Global Constraints

- Runtime dependencies stay `numpy` and `scipy` only. Do not add new ones.
- Double precision throughout. All randomness comes from `espi_sim.rng.stream(seed, *names)`.
- Seeded tests assert tolerances or statistics. The only exact comparison is
  same-seed-same-run equality.
- Sign convention (M1): the reference beam carries `exp(+iδ)`, so a frame is
  `I = A + B cos(φ − δ)` and `analysis.phase_from_steps(frames, steps=δ)` applies unchanged.
- Vibration convention: `w(x, t) = Re(W(x) e^{iωt})`. The modal factor is
  `1/(ω_mn² − ω² + 2iζω_mn ω)`, so `arg` goes 0 → −π/2 → −π across a resonance.
- Steady-state warning when the sweep rate exceeds `0.25·π·f²/Q²` (Hz/s). Short-exposure
  warning when `f·T_exp < 10`. Both are `UserWarning`s.
- Defaults (spec): `Drumhead(radius=0.1, f01=400.0, q=50.0, drive=PointDrive(r=0.04,
  theta_deg=30.0))`, `VideoTiming(frame_rate_hz=30.0, exposure_s=0.03)`,
  `TriangleWave(frequency_hz=0.25, phase_pp=4π)`.
- Lint: `uv run ruff check src tests examples` and `uv run ruff format --check src tests
  examples` must be clean, with line length 100.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Drive point outside the head.** `PointDrive(r ≥ radius)` should raise `ValueError` at
   construction, not silently excite nothing. Test in Task 1.
2. **Non-physical parameters.** `q ≤ 0`, `f01 ≤ 0`, or a tone frequency ≤ 0 should raise
   `ValueError`, not divide by zero. Test in Task 1.
3. **Vibration slope too steep for the grid.** When the instantaneous phase gradient
   `max|∇(S_z·|W|)|` exceeds π/2 rad per simulation sample, warn ("under-sampled") rather
   than silently alias. Test in Task 4.
4. **Drive switched off** (`level == 0` segments in a `SteppedSweep`). Those frames must
   equal a static scene: state modulation exactly 1, amplitude exactly 0. Test in Task 4.
5. **Bad frame count.** `n_frames < 1` raises `ValueError`. Test in Task 4.

---

### Task 1: Membrane modal model (`drumhead.py`)

**Files:**
- Create: `src/espi_sim/drumhead.py`
- Test: `tests/test_drumhead.py`

**Interfaces:**
- Produces:
  - `PointDrive(r: float, theta_deg: float = 0.0)` and `UniformPressure()`, both frozen
    dataclasses.
  - `MembraneMode(m: int, n: int, alpha: float, frequency_hz: float, amplitude: complex)`.
    `amplitude` is the dimensionless modal coefficient `a_mn`.
  - `Drumhead(radius=0.1, f01=400.0, q=50.0, drive=PointDrive(0.04, 30.0),
    center=(0.0, 0.0), max_m=8, max_n=6)`, frozen, with methods:
    - `mode_frequency(m: int, n: int) -> float`
    - `modes(frequency_hz: float) -> tuple[MembraneMode, ...]`, which keeps modes with
      `|a| ≥ 1e-4·max|a|`
    - `complex_amplitude(xi, eta, frequency_hz: float, level: float) -> NDArray[complex128]`,
      in metres, 0 where `r ≥ radius`
    - `calibrate(peak_amplitude: float, at_hz: float) -> float`

- [ ] **Step 1: Write failing tests** in `tests/test_drumhead.py`:
  - `test_natural_frequency_ratios_follow_bessel_zeros`: `mode_frequency(1,1)/f01 ≈
    3.831706/2.404826`, `(2,1) ≈ 5.135622/2.404826`, `(0,2) ≈ 5.520078/2.404826`, all with
    rel 1e-6.
  - `test_response_vanishes_at_rim_and_outside`: at `r ∈ {radius, 1.2·radius}` and several θ,
    `|W| < 1e-9·max|W|`.
  - `test_on_resonance_fundamental_shape_is_j0`: `Drumhead(q=500, drive=PointDrive(r=0))`
    at `f = f01`. Along a radius, `corrcoef(|W|, j0(2.404826·r/a)) > 0.99`.
  - `test_off_centre_drive_at_f11_has_nodal_diameter`: `q=500`,
    `PointDrive(r=0.04, theta_deg=30)`, `f = mode_frequency(1,1)`.
    `max|W| along θ=120° < 0.05·max|W| along θ=30°`.
  - `test_uniform_pressure_is_axisymmetric`: `UniformPressure()` at `1.3·f01`. `W(r, θ)` at
    θ ∈ {0°, 90°, 200°} agree to rel 1e-9.
  - `test_modal_phase_lag_across_fundamental`: the (0,1) mode's `np.angle(a)` is about 0
    (abs 0.05) at `0.5·f01`, about −π/2 (abs 1e-3) at `f01`, and `|angle| > 0.95π` at
    `2·f01`.
  - `test_calibrate_sets_peak_amplitude`: `level = calibrate(0.5e-6, at_hz=f01)`. On a
    401×401 Cartesian grid over the head, `max|W| == approx(0.5e-6, rel=0.02)`.
  - `test_fixed_level_amplitude_follows_resonance`: with the same `level`, the peak `|W|` at
    `f01` is more than 5× the peak at `0.8·f01`.
  - `test_invalid_parameters_raise`: `PointDrive` with `r ≥ radius` (checked in
    `Drumhead.__post_init__`), `q ≤ 0`, `f01 ≤ 0`, and `modes(0.0)` each raise `ValueError`.
    *(Review Focus 1, 2.)*

- [ ] **Step 2: Run** `uv run pytest tests/test_drumhead.py -q`. Expected: collection error
  (module missing).

- [ ] **Step 3: Implement `src/espi_sim/drumhead.py`.**
  - Use `scipy.special.jn_zeros(m, max_n)` for `α_mn`, `m = 0..max_m`, and
    `f_mn = f01·α_mn/α_01`.
  - Mode shape: `φ_mn = J_m(α r/a)·cos(m(θ − θ_d))`, where `θ_d` is the drive angle (0 for
    `UniformPressure`).
  - Area-normalised norm: `N_mn = J_{m+1}(α)²` for `m = 0`, and `J_{m+1}(α)²/2` for `m > 0`.
  - Forcing projection:
    - point drive: `F = J_m(α r_d/a)`;
    - uniform pressure: `F = 2·J1(α)/α` for `m = 0`, else 0.
  - Modal coefficient:
    `a_mn = ω01²·F/(N·(ω_mn² − ω² + 2iζω_mn ω))`, with `ζ = 1/(2q)` and `ω = 2πf`.
  - Response: `W = level·Σ a_mn φ_mn`, set to 0 where `r ≥ a`. Coordinates are relative to
    `center`, and `θ = atan2(eta, xi)`.
  - `calibrate`: evaluate `|W|` with `level = 1` on a polar grid (r: 400 points in
    `[0, a)`, θ: 720 points), then return `peak_amplitude / max`.

- [ ] **Step 4: Run** `uv run pytest tests/test_drumhead.py -q`. Expected: all pass.

- [ ] **Step 5: Commit**:
  `git add src/espi_sim/drumhead.py tests/test_drumhead.py && git commit -m "Add drumhead
  membrane modal response model …"`

---

### Task 2: Drive and timing schedules (`waveforms.py`) and config fields

**Files:**
- Create: `src/espi_sim/waveforms.py`
- Modify: `src/espi_sim/config.py`. Add two fields to `SimConfig`:
  `reference_drive: TriangleWave | None = None` and
  `timing: VideoTiming = field(default_factory=VideoTiming)`.
- Test: `tests/test_waveforms.py`, and one test appended to `tests/test_config.py`.

**Interfaces:**
- Produces (all frozen dataclasses):
  - `Tone(frequency_hz: float, level: float)`.
  - `SteppedSweep(frequencies_hz: tuple[float, ...], dwell_s: float,
    levels: float | tuple[float, ...])`.
  - `Sweep(f_start: float, f_stop: float, duration_s: float, level: float,
    spacing: str = "linear")`, where `spacing` is `"linear"` or `"log"`.
  - Each schedule has `tone_at(t: float) -> tuple[float, float]`, returning
    `(frequency_hz, level)`, and `rate_at(t: float) -> float` in Hz/s. `rate_at` is 0 for
    `Tone` and `SteppedSweep`.
  - `TriangleWave(frequency_hz=0.25, phase_pp=4π, offset=0.0, start_fraction=0.0,
    nonlinearity=0.0)`:
    - `phase_at(t: ArrayLike) -> NDArray` gives the reference phase δ in rad.
    - `phase_for_travel(travel: float, wavelength: float) -> float` is a static method:
      `4π·travel/λ`.
  - `VideoTiming(frame_rate_hz=30.0, exposure_s=0.03)` with `frame_start(k: int) -> float`.
    `__post_init__` raises `ValueError` if `exposure_s > 1/frame_rate_hz` or either value
    is ≤ 0.

- [ ] **Step 1: Write failing tests** in `tests/test_waveforms.py`:
  - `test_tone_is_constant`.
  - `test_stepped_sweep_indexes_dwell_and_clamps`: frequencies `(100, 200, 300)`,
    `dwell_s=1.0`.
    - `tone_at(0.5)[0] == 100`, `tone_at(1.0)[0] == 200`, `tone_at(9.0)[0] == 300` (clamped).
    - With per-step `levels=(1.0, 0.0, 2.0)`, `tone_at(1.5)[1] == 0.0`.
  - `test_linear_and_log_sweep_endpoints_midpoints_and_rates`, for 100 → 400 Hz over 10 s:
    - linear: midpoint 250, rate 30;
    - log: midpoint 200, `rate_at(5) == approx(200·ln(4)/10)`;
    - `t` outside `[0, duration]` is clamped.
  - `test_triangle_wave_shape`: `TriangleWave(frequency_hz=1.0, phase_pp=4π, offset=0.5)`
    at t = 0, 0.25, 0.5, 0.75, 1.0 gives 0.5, 0.5+2π, 0.5+4π, 0.5+2π, 0.5. With
    `nonlinearity=0.01` the value is `δ + 0.01·δ²`.
  - `test_phase_for_travel`: `phase_for_travel(632.8e-9/2, 632.8e-9) == approx(2π)`.
  - `test_video_timing`: `frame_start(3) == 0.1` at 30 fps. `VideoTiming(30, 0.05)` raises
    `ValueError`.
  - In `tests/test_config.py`, `test_simconfig_video_defaults`: `SimConfig().reference_drive
    is None` and `SimConfig().timing == VideoTiming(30.0, 0.03)`.

- [ ] **Step 2: Run** `uv run pytest tests/test_waveforms.py tests/test_config.py -q`.
  Expected: failures (module or fields missing).

- [ ] **Step 3: Implement.**
  - Triangle: `u = frac(f·t + start_fraction)`, `tri = 2u` for `u < 0.5`, else `2(1 − u)`,
    then `δ = offset + phase_pp·tri`, then `δ + nonlinearity·δ²`.
  - Log sweep: `f0·(f1/f0)^(t/T)`.
  - `config.py` imports `TriangleWave` and `VideoTiming` from `.waveforms`. `waveforms`
    must not import `config`.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: the whole suite passes.

- [ ] **Step 5: Commit** (`waveforms.py`, `config.py`, the tests).

---

### Task 3: Cycle and exposure averaging (`time_average.py`) and complex pixel integration

**Files:**
- Create: `src/espi_sim/time_average.py`
- Modify: `src/espi_sim/camera.py`. In `integrate_pixels`, keep the result complex when the
  input is complex: apply `.real` only for real input.
- Test: `tests/test_time_average.py`, and one test appended to `tests/test_camera.py`.

**Interfaces:**
- Consumes: `optics.image_field(field, pupil)`, `optics.circular_pupil`,
  `optics.rough_surface`.
- Produces:
  - `cycle_samples(beta_max: float) -> int`: `max(32, ceil(beta_max) + 16)`, rounded up to a
    multiple of 4.
  - `cycle_averaged_fields(object_field: NDArray[complex], phase_amplitude:
    NDArray[complex], pupil: NDArray[bool]) -> tuple[NDArray[complex], NDArray[float]]`.
    It returns `(mean_field, mean_intensity)` on the simulation grid, where
    `phase_amplitude = B` and the phase at time t is `Re(B·e^{iωt})`.
  - `exposure_average(phase_fn: Callable[[NDArray], NDArray], t0: float, t1: float,
    n: int = 64) -> complex`: the midpoint-rule mean of `exp(iδ(t))` over `[t0, t1]`.

- [ ] **Step 1: Write failing tests** in `tests/test_time_average.py`:
  - `test_cycle_samples`: `cycle_samples(0) == 32`. `cycle_samples(40.2)` is ≥ 57 and a
    multiple of 4.
  - `test_static_vibration_returns_static_field`: with `B = 0`, `mean_field ==
    image_field(U)` and `mean_intensity == |image_field(U)|²`, to atol 1e-9 relative to the
    mean.
  - `test_uniform_vibration_scales_field_by_j0`: with uniform `B = 2.0`,
    `mean_field == approx(j0(2.0)·image_field(U))` and `mean_intensity ==
    approx(|image_field(U)|²)`, both rel 1e-6.
  - `test_oscillating_tilt_smears_speckle`: let `B = 2π·c·col_index`, where `c` is the
    pupil cutoff in cycles/sample (the spectrum swings by one pupil radius). The speckle
    contrast (`std/mean`) of `mean_intensity` is below 0.8, while the static field's is
    about 1.
  - `test_exposure_average_of_linear_ramp`: for `δ = 3.0·t` over `[0, 0.5]`,
    `|avg| == approx(|np.sinc(1.5/(2π))|, rel=1e-3)` and `angle(avg) == approx(0.75)`.
  - In `tests/test_camera.py`, `test_complex_integration_is_linear`:
    `integrate_pixels(a + 1j·b)` equals `integrate_pixels(a) + 1j·integrate_pixels(b)`.

- [ ] **Step 2: Run** `uv run pytest tests/test_time_average.py tests/test_camera.py -q`.
  Expected: failures.

- [ ] **Step 3: Implement.**
  - `mean_field = image_field(U·scipy.special.j0(|B|), pupil)`. This is the exact
    continuous average, by Jacobi–Anger.
  - `mean_intensity` is the mean over `j < N_t` of
    `|image_field(U·exp(i·Re(B·e^{2πij/N_t})), pupil)|²`, with
    `N_t = cycle_samples(max|B|)`.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.

- [ ] **Step 5: Commit**.

---

### Task 4: `Simulator.acquire_video` and the video result types

**Files:**
- Create: `src/espi_sim/video.py`, containing the `ToneState` and `VideoAcquisition`
  dataclasses and `steady_state_warnings`.
- Modify: `src/espi_sim/simulator.py`. Add the method `acquire_video`.
- Modify: `src/espi_sim/__init__.py`. Export `Drumhead`, `PointDrive`, `UniformPressure`,
  `Tone`, `SteppedSweep`, `Sweep`, `TriangleWave`, `VideoTiming` and `VideoAcquisition`.
- Test: `tests/test_video.py`

**Interfaces:**
- Consumes: Task 1 `Drumhead.complex_amplitude/modes`, Task 2 schedules and
  `SimConfig.timing/reference_drive`, Task 3 `cycle_averaged_fields/exposure_average`,
  complex `integrate_pixels`, `Simulator._surface(0)`, `Simulator._reference_field`,
  `Simulator._pupil_fraction`, `Simulator.pupil`.
- Produces:
  - `ToneState(frequency_hz, level, amplitude, phase, modulation, modes)`. Its maps are
    `(ny, nx)` at pixel centres: `amplitude = |W|`, `phase = arg W`,
    `modulation = j0(S_z·|W|)` (signed).
  - `VideoAcquisition(frames: uint16 (N,ny,nx), electrons: float (N,ny,nx), times (N,),
    reference_phase (N,), reference_visibility (N,), frequency_hz (N,), level (N,),
    state_index: int (N,), states: tuple[ToneState, ...], sensitivity (3,),
    frame_offset: int, metadata: dict)`.
  - `Simulator.acquire_video(drum: Drumhead, tone: Tone | SteppedSweep | Sweep,
    n_frames: int, frame_offset: int = 0) -> VideoAcquisition`.
  - `steady_state_warnings(tone, drum, timing, mid_times: NDArray) -> list[str]`.

- [ ] **Step 1: Write failing tests** in `tests/test_video.py`. Shared fixture:
  - camera `(96, 96)`;
  - optics `Optics(f_number=8, magnification=96·3.45e-6/(2.2·0.1))`, so the head plus rim
    fills the frame;
  - an ideal camera (`shot_noise=False, read_noise=0`) unless stated;
  - `reference_drive=TriangleWave(frequency_hz=30/16, phase_pp=2π)`, so 8 frames cover 2π
    on the rising half;
  - `timing=VideoTiming(30, 0.03)`.

  Tests:
  - `test_video_shapes_times_and_reference_phase`:
    - shapes and dtypes are correct;
    - `times == (frame_offset + k)/30`;
    - on the rising half, `reference_phase == approx(phase_at(t_mid))`;
    - `reference_visibility` is in `(0.9, 1]`.
  - `test_modulation_ratio_tracks_abs_j0`: compare two videos with the same seed and
    `Tone(f01, level)`:
    - one with `level = drum.calibrate(0.4e-6, f01)`, one with `level = 0`;
    - fit `phase_from_steps(electrons[:8], reference_phase[:8]).modulation` for each and
      take `ratio = mod_vib/mod_static` on pixels where `mod_static > 0.2·median`;
    - expect `median|ratio − |states[0].modulation|| < 0.05` inside `r < 0.9a`;
    - and the pixels with `|amplitude − 121.1 nm| < 5 nm` have median ratio < 0.12.
  - `test_drive_off_frames_equal_static_scene` *(Review Focus 4)*: `level = 0`. Then
    `states[0].amplitude` is all 0 and `states[0].modulation` all 1. The electrons equal
    those of a `level = 0` `Tone` at a different frequency (rel 1e-12).
  - The two sweep tests below use `Drumhead(q=10)` with
    `level = drum.calibrate(0.4e-6, f01)`. That keeps the short dwell and the 200 Hz/s sweep
    inside the steady-state limits, so they emit no warnings. They also keep every tone
    ≥ 360 Hz, so a 30 ms exposure spans ≥ 10 cycles.
  - `test_stepped_sweep_reuses_states_and_peaks_on_resonance`:
    - `SteppedSweep((0.9·f01, f01, 1.1·f01), dwell_s=4/30, levels=level)` over 12 frames;
    - `len(states) == 3` and `state_index == [0]*4 + [1]*4 + [2]*4`;
    - `states[1].amplitude.max()` exceeds the other two.
  - `test_continuous_sweep_makes_one_state_per_frame`: `Sweep(380, 420, duration_s=6/30,
    level)` over 6 frames gives `len(states) == 6`, with no warnings
    (`warnings.simplefilter("error")`).
  - `test_fast_sweep_warns_steady_state`: with the default `q=50`,
    `Sweep(300, 900, 0.2, level)` warns with a `UserWarning` matching "steady".
  - `test_short_exposure_warns_cycles`: `VideoTiming(30, 0.001)` at 400 Hz warns with a
    `UserWarning` matching "cycles".
  - `test_undersampled_vibration_warns` *(Review Focus 3)*: `Tone(f01,
    calibrate(10e-6, f01))` warns with a `UserWarning` matching "under-sampled". At about
    200 rad of peak phase, the gradient is above 1.5 rad/sample while `N_t` stays near 220.
  - `test_rejects_dual_beam_and_bad_frame_count` *(Review Focus 5)*: `Setup.in_plane()`
    raises `ValueError`, and `n_frames=0` raises `ValueError`.
  - `test_video_reproducible_and_frame_offset_continues`:
    - the same seed gives identical frames with the default noisy camera;
    - `acquire_video(..., n_frames=4, frame_offset=4)` has `times[0] == 4/30` and
      `reference_phase` equal to frames 4–7 of an 8-frame video.

- [ ] **Step 2: Run** `uv run pytest tests/test_video.py -q`. Expected: failures.

- [ ] **Step 3: Implement `acquire_video`.**
  1. **Validate:** a reference-beam setup is required, and `n_frames ≥ 1`.
  2. **Timing:** frame times `t_k = (frame_offset + k)/fps`, with mid-exposure
     `t_k + T/2`. Tones come from `tone.tone_at(mid)`.
  3. **Warnings:** emit each message from `steady_state_warnings` with
     `warnings.warn(..., UserWarning, stacklevel=2)`, before any FFT work. Cover:
     - **sweep rate:** `max rate_at(mid) > 0.25·π·f²/q²` → "steady";
     - **stepped dwell:** for `SteppedSweep`, `dwell_s < 5·q/(π·f_min)` → "steady";
     - **exposure length:** `min f·T_exp < 10` over the driven (level ≠ 0) frames →
       "cycles".
  4. **Per distinct `(f, level)` (dict cache):**
     - `W = drum.complex_amplitude(sim-grid ξ, η, f, level)` and `B = S_z·W`, where `S_z`
       is the z-component of `setup.sensitivity(λ)`;
     - if `max|∇|B|| > π/2` per sample (`np.gradient`), warn "under-sampled";
     - `U = sqrt(beam.mean_electrons/_pupil_fraction)·_surface(0)`;
     - `(mean_field, mean_intensity) = cycle_averaged_fields(U, B, pupil)`;
     - `base = integrate_pixels(mean_intensity + |E_r|², grid, ff)` and
       `cross = integrate_pixels(mean_field·conj(E_r), grid, ff)` (complex);
     - build a `ToneState` from `drum.complex_amplitude` at `grid.pixel_object_coords()`.
  5. **Per frame:**
     - `avg = exposure_average(reference_drive.phase_at, t_k, t_k + T)`, or 1 if there is
       no drive;
     - `γ = laser.coherence(L_beam − L_ref, frame=global index)`;
     - `e = base + 2·Re(cross·γ·conj(avg))`;
     - apply RIN as in `acquire`, then
       `frames = expose(e, camera, stream(seed, "camera", global index))`;
     - record `reference_phase = angle(avg)` and `reference_visibility = |avg|`.
  6. **Metadata:** `{"drumhead": asdict(drum), "tone": {"type": name, **asdict(tone)}}`.

- [ ] **Step 4: Run** `uv run pytest -q`. Expected: the whole suite passes. Then run
  `uv run ruff check src tests && uv run ruff format --check src tests`.

- [ ] **Step 5: Commit**.

---

### Task 5: Video `.npz` I/O

**Files:**
- Modify: `src/espi_sim/io.py`. Add `save_video` and `load_video`.
- Modify: `src/espi_sim/__init__.py`. Export both.
- Test: `tests/test_io.py`, by appending one test.

**Interfaces:**
- Produces:
  - `save_video(path, video: VideoAcquisition, config: SimConfig | None = None) -> None`;
  - `load_video(path) -> tuple[VideoAcquisition, dict | None]`, using format tag
    `"espi-sim/video/1"`.
  - Per-state maps are stacked as `state_amplitude` / `state_phase` / `state_modulation`
    `(S, ny, nx)`, with `state_frequency_hz` and `state_level` `(S,)`. Modes go into the
    meta JSON as `[m, n, alpha, frequency_hz, re, im]`.

- [ ] **Step 1: Write the failing test** `test_video_npz_round_trip`. A 4-frame `(32, 32)`
  video round-trips:
  - frames are exactly equal;
  - electrons, times, `reference_phase`, `state_index` and every `ToneState` map (and its
    modes) match to allclose;
  - `config_dict["timing"]["frame_rate_hz"] == 30.0`;
  - loading an M1 acquisition file with `load_video` raises `ValueError`.
- [ ] **Step 2: Run** `uv run pytest tests/test_io.py -q`. Expected: failure.
- [ ] **Step 3: Implement**, mirroring `save_acquisition` (compressed npz, JSON meta,
  `allow_pickle=False`).
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit**.

---

### Task 6: Example, README and spec status

**Files:**
- Create: `examples/drumhead_sweep.py`
- Create: `docs/images/drumhead_modes.png`, generated by the example.
- Modify: `README.md`. Add a "Time-averaged vibration (drumhead)" section with a usage
  snippet and the image.
- Modify: `docs/superpowers/specs/2026-11-21-time-averaged-drumhead-design.md`. Set the
  status to "implemented" and add any implementation notes.

**Interfaces:**
- Consumes: the public API from Tasks 1–5.

- [ ] **Step 1: Write `examples/drumhead_sweep.py`.**
  - Camera `(512, 512)`, `magnification = 512·3.45e-6/(2.2·0.1)`, `f_number=8`, the default
    noisy camera.
  - `SteppedSweep` over the (0,1), (1,1), (2,1) and (0,2) resonance frequencies, with
    `dwell_s = 2.0` (one rising half of the 0.25 Hz triangle).
  - Calibrate the level to a 0.5 µm peak at `f01`.
  - For each dwell, fit the frames on the rising ramp with `phase_from_steps` and divide by
    the median static-rim modulation.
  - Plot each normalised modulation map above its `|ToneState.modulation|` ground truth,
    and print each state's max amplitude.
  - Write `outputs/drumhead_modes.png`, or the path given by `argv[1]`.
- [ ] **Step 2: Run** `uv run python examples/drumhead_sweep.py`. Then `Read` the PNG and
  confirm:
  - the four mode patterns match the ground truth (nodal lines bright, J0 dark rings);
  - the (1,1) and (2,1) patterns show 1 and 2 nodal diameters;
  - the run finishes in under about 60 s.
- [ ] **Step 3: Copy the PNG** to `docs/images/drumhead_modes.png`, update the README and the
  spec status, and run `uv run ruff check src tests examples` and `uv run pytest -q`.
- [ ] **Step 4: Commit**.
