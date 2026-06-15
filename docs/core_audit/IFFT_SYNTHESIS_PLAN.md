# IFFT Synthesis (Rodet–Depalle) — F-stream plan

Decision basis (closed): P6/D4 priced the fork structurally; pass 255 measured
it — one CMSIS-DSP inverse RFFT-512 (Q31, Daisy flags) = **43,910 insns**
[measured: qemu-tcg] vs **6,528 insns per oscillator-bank partial per
256-sample block** [measured] ⇒ crossover ≈ **7 dense partials** (4–16 across
the IPC band). Dense material is ~8× cheaper at 64 partials and scales. The
oscillator bank stays the exact path and the default; IFFT is an
approximation and ships **only behind a gate + golden sign-off** (AI_CANON
rule 3).

## Method (one paragraph)

Per output frame of N samples (hop H = N/2): for each partial, add its
**spectral motif** — the sampled spectrum of the synthesis window, precomputed
oversampled — into the frame's complex half-spectrum at the partial's
fractional bin, scaled by amplitude and rotated by the phase at frame center;
one inverse real FFT; multiply-free overlap-add (the synthesis window is baked
into the motif). The synthesis window is **periodic Hann at 50% overlap** ⇒
exact COLA (gain 1, verified by `spectral_overlap_add_is_constant`). Window
convention note: this is the IFFT path's INTERNAL synthesis window — the
analysis-window convention decision (REVIEWER_HANDOFF_2 §4.4) stays untouched
and no analysis golden moves.

## Error budget (the gate criteria)

Approximation sources, each bounded in the F1 harness before integration:
1. **Motif truncation** (K bins per side): MEASURED (F1, pass 257, double
   precision, N=512): frame-level max error K=4 → −43 dBFS, K=8 → −55,
   K=12 → −63 (Hann decays ~1/d³; the original −80 dB@K=8 guess was wrong —
   measurement replaced it). Stream-level after OLA: −68…−76 dBFS max,
   **−83…−92 dBFS RMS** at K=8–12. K (and/or a faster-decaying motif window
   pair) is an explicit F2 operating-point decision; per-partial cost is
   only ~2K complex MACs, so K barely moves the crossover.
2. **Motif interpolation** (oversample O, linear interp): MEASURED — O=16
   is within 1 dB of O=64 at every K (truncation dominates); small tables
   are fine.
3. **Intra-frame stationarity**: frequency/amp ramps (df/da) are held to
   frame-center values; error grows with |df|·H — the hybrid sends fast
   chirps/fades to the oscillator bank (F2 routing rule, measured threshold).
Parity ladder: (a) frame-level — built spectrum iFFT'd vs time-domain
windowed cosine; (b) stream-level — OLA output vs the exact oscillator sum
over stationary partials; (c) golden — full pipeline render, maintainer signs.

## File organization (pass-256 maintainer ruling)

One contract header `synth/spectral_synth_ifft.h` in OUR types; per-backend
TUs in port dirs, never macro mono-files: `port/host/spectral_ifft_vdsp.c`
(+ FFTW variant), `port/cmsis/spectral_ifft_cmsis.c` (Q31). Third-party
headers do not leak through the contract; every port pair gets a parity test
(`window_backend_parity` pattern). The motif table is generated (SSOT
generator, committed artifact — resource-hash pattern).

## Phases

- **F1 — math harness (no engine code).** `tests/core_math/harnesses/
  ifft_synth_sweep.c`: motif build + frame builder + reference DFT + OLA;
  measures the error-budget numbers above; pytest driver asserts the ladder
  (a)+(b) within the measured-then-frozen bounds. Done-when: error floor
  numbers recorded here and the harness is green.
- **F2 — desktop reference path.** `spectral_synth_ifft.c` (float, host FFT
  backend via the contract header), opt-in flag (no default change), hybrid
  density router (oscillator bank below crossover / fast chirps; IFFT above),
  stream parity test vs osc-bank in ctest. Done-when: ctest parity green +
  measured desktop speedup on a dense fixture.
- **F3 — golden sign-off.** Maintainer listens/diffs a dense render;
  thresholds frozen into the golden set. Done-when: signed.
- **F4 — embedded (Cortex-M7) port + DEFAULT-ON for embedded (maintainer
  directive 2026-06-15).** The compat audit (see "Embedded integration" below)
  ruled **float-on-M7-FPU over Q31**: the M7 has a hardware single-precision FPU,
  the CMSIS oscillator already uses `arm_sin_f32`, and a Q15 motif table would
  lose ~12–15 dB SNR at dense densities. So F4 = a float embedded port of
  `spectral_synth_ifft.c`, NOT a Q31 rewrite. Three sub-pieces:
  - **F4a — static allocation.** `spectral_ifft_synth_create` mallocs 6×+callocs
    2× (spectral_synth_ifft.c:39-79) and the ref backend computes twiddles at init
    — all forbidden on the libc-free embedded build. Add a pre-sized/pool-backed
    create + a portable no-libm FFT (CMSIS `arm_rfft_fast_f32` or a static-twiddle
    ref) + LUT/`spectral_fast_math` trig instead of `cosf`/`sinf`/`fmod`.
  - **F4b — STREAMING OLA.** The embedded render is block-by-block
    (`spectral_arm32_process`, audio callback), but `spectral_ifft_synth_render`
    renders a whole buffer with internal OLA. A streaming variant must carry the
    half-frame OLA tail + frame phase across `process()` calls in the ctx.
  - **F4c — route + default-on.** Eligibility gate in `spectral_arm32_process`
    (num_active ≥ threshold, all sustain/no-fade-this-block, no chirp, in-domain),
    an arm32 IFFT wrapper, and the m7 baseline REGENERATED (flipping the default
    render path moves codegen + cycles — that IS the new gate). Density-gated +
    K-chosen so the floor is inaudible (see the accuracy note). Skip dual-MAC
    pair-eligible voices. Done-when: rig/qemu cycles show the speedup + gate
    re-frozen + parity oracle green + maintainer golden (F3) for the embedded set.
- **F5 — capacity republish.** Re-run the capacity table with the hybrid;
  update M7_PERF_MODEL_PLAN + the published guarantees.
- **F6 — CPU parallelization (SIMDe host / vDSP / CMSIS-DSP).** The FFT⁻¹ pipeline
  is data-parallel in all three stages (see "Parallelization" below): partial→bin
  motif placement, the inverse FFT (already vDSP/cuFFT-class), and OLA. Host:
  SIMDe-vectorize `place_partial`'s contiguous tap scatter + the OLA add (bit-
  parity, build-agnostic SSE2→NEON); use vDSP vector ops for the OLA on Apple.
  Embedded: the M7 has NO NEON — use CMSIS-DSP `arm_*_f32` + the dual-issue FPU.
  Done-when: parity unchanged + measured per-backend speedup in the bench.
- **F7 — GPU FFT⁻¹ synthesis (Metal / CUDA).** The big lever (Savioja et al. —
  additive synthesis is "embarrassingly parallel", ~1000× CPU on GPU). Parallel
  partial→bin scatter (one thread per partial or per bin), batched inverse FFT
  (Metal Performance Shaders / cuFFT), parallel OLA across many frames. Reuses the
  existing Metal/CUDA backend plumbing + the GPU-parity ctest pattern (#26).
  Done-when: GPU↔CPU parity within the IFFT floor + measured GPU speedup.

## Status

- F0 (this plan): committed pass 257.
- F1 CLOSED (pass 257): `tests/core_math/harnesses/ifft_synth_sweep.c` +
  `test_ifft_synth_sweep.py` — COLA machine-exact; frame/stream parity
  ladder green; measured floors frozen into the pytest gate (+3 dB
  headroom). The frame-builder construction (centered motif, (−1)^k
  twiddle, Hermitian placement) is validated and is the SSOT formulation
  for F2.
- F2 core CLOSED (pass 258): contract `synth/spectral_synth_ifft.h` +
  frame renderer `synth/spectral_synth_ifft.c` (float port of the F1
  formulation, K=8/O=16 measured operating point) + two host iFFT ports
  (`core/port/host/spectral_ifft_vdsp.c`, `spectral_ifft_ref.c` — exactly
  one live per build, selected on SPECTRAL_USE_VDSP). ctest
  `ifft_synth_parity` (#20): port-vs-reference-iDFT 5.6e-9; stream parity
  vs the exact oscillator sum **−72.8 dBFS max / −87.5 dBFS RMS** at 64
  dense partials; deterministic; **MEASURED 7.5× over the naive oscillator
  loop on desktop** (matches the embedded pricing's ~8× at 64 partials).
- F2b IN PROGRESS — the hybrid density router into the engine dispatch (segment
  semantics: stationary sine interiors → IFFT, fades/chirps/non-sine →
  oscillator bank; opt-in flag, no default change), then F3 golden. **Step 1
  LANDED** (renderer per-frame-activity foundation + ctest rung 4); **Step 2 v1
  LANDED** (opt-in `spectral_synth_hybrid_try_render` fast path for the all-eligible
  dense case + ctest #25). Remaining: the mixed-case seam correction, stretch/pitch,
  and the dispatch wiring (rides F3 golden). See the F2b build log below.

## F2b — implementation design (characterized 2026-06-15, NOT yet built)

Characterized against the current code so the build is mechanical, not exploratory.
F2b is correctness-critical (a mis-routed fade is an audible click), so it is its
own focused unit, not a tack-on.

**The two facts the router must respect (verified in code):**
1. `spectral_ifft_synth_render(s, partials, n, out, total)` (spectral_synth_ifft.c:118)
   renders a *stationary* partial set over the WHOLE `total`: every frame evaluates
   ALL `n` partials at frame center (`phi = omega*center + phase0`, :154). It has no
   notion of a partial starting/ending. Real segments are time-localized (start,
   length).
2. The fade envelope is applied at RENDER time by the oscillator path (per-segment
   fade-in/out over `fade_len`), not stored in the Segment. The IFFT renders a
   partial as steady → routing a faded span to IFFT DROPS the fade (click). So a
   segment must be split: stationary interior → IFFT, fade edges → osc bank.

**Design:**
- **Renderer extension (per-frame activity).** Add a dynamic entry beside
  `spectral_ifft_synth_render` that drives the same frame loop + OLA (:141-167) but
  pulls the active partials PER FRAME from a caller callback
  `(frame_center_sample) -> {SpectralIfftPartial[], count}`. The framing/OLA/motif
  stay owned by the renderer (the subtle part); the engine owns "which partials are
  live this frame". Keep the existing static API (its parity test #20 stays the
  oracle for the dynamic path on a stationary fixture: dynamic-with-constant-set ==
  static).
- **Segment classifier** `is_stationary_sine_segment(seg)`: sine timbre AND
  |da|·length below the F1 intra-frame-stationarity threshold (38-40) AND |df|≈0.
  Non-sine / chirped / amplitude-ramped → never IFFT.
- **Interior/edge split.** For an eligible segment, only `[start+fade_len,
  end-fade_len]` is IFFT-routed; `[start, start+fade_len]` and `[end-fade_len, end]`
  go to the osc bank (which already applies the fade ramp). The per-frame callback
  emits a partial for a segment only when the frame center lies in its interior;
  `bin = omega·n_fft/2π`, `amp = seg.amp`, `phase0 = seg.phase − omega·seg.start`
  (so the renderer's `omega·center + phase0` reproduces the segment phase model).
- **Sum.** IFFT interiors OLA into the output buffer; the osc bank renders edges +
  all non-eligible segments into the SAME buffer (additive). One shared output.
- **Crossover gate.** Only engage IFFT for frames with ≳7 simultaneous eligible
  partials (the F1/P6 measured crossover); below that the osc bank is cheaper — so
  the router counts per-frame eligible density and falls back when sparse.
- **Opt-in.** A compile gate (default 0) at the synth_cpu seam (around
  `synth_cpu_driver`, spectral_synth_cpu.c:431): default OFF → the IFFT code is not
  compiled, the default render path is byte-identical, m7 perf gate untouched.

**Test (done-when):** a tolerance stream-parity ctest — IFFT-routed engine render vs
pure osc-bank render — within the F1-frozen floor (≈ −72 dBFS max / −87 dBFS RMS at
64 dense partials, +3 dB headroom), on a dense stationary fixture AND a mixed fixture
(dense interiors + faded/chirped segments) that exercises the split seam (assert no
discontinuity at the interior↔edge handoff). Plus an informational desktop speedup.

**Risks to verify during build:** (a) phase continuity at the interior↔edge seam
(IFFT and osc must agree on the partial's phase at the boundary sample); (b) OLA
coverage at interior boundaries (the renderer needs frames overlapping the interior
edges); (c) the crossover fallback must not double-count a partial (IFFT or osc, never
both, for a given span). Default-on acceptance rides F3 (golden, maintainer).

### F2b build log

- **Step 1 DONE (pass 259, this commit-stream).** The renderer per-frame-activity
  foundation: `spectral_ifft_synth_render_dynamic(s, fn, ctx, out, total)` with the
  framing/OLA/motif extracted into one shared `ifft_render_frame()` + `ifft_partial_valid()`;
  the static `spectral_ifft_synth_render` now delegates (stream parity unchanged at
  −72.81/−87.46 dBFS, byte-for-byte the pre-refactor numbers; backend rung 5.6e-9).
  ctest rung 4 added: constant set == static **bit-exact**, time-localized activity keeps
  energy to active frames (far edges exactly 0.0), out-of-domain partials skipped. The
  IFFT TU still has no production caller — the Step-2 router is its first.

- **Step 2 — the segment→partial mapping is verified; the boundary is the hard part.**
  Segment model (spectral_common.h:23): `start,length,phase,omega,df,amp,da,width` +
  cubic c2/c3 in pad. The IFFT partial for an eligible segment is
  `bin = omega·n_fft/2π`, `amp = seg.amp`, `phase0 = seg.phase − omega·seg.start`
  (matches the osc `phase_at_cubic(phase, omega, …, j)` at absolute `t = start+j`).
  Eligibility = **global timbre == sine** (the motif is a single cosine bin; non-sine
  timbres have harmonics the IFFT can't place) AND `|df|≈0` AND `|da·length|` below the
  intra-frame threshold AND no cubic AND `bin ∈ (1, n_fft/2−1)` (out-of-domain → osc,
  never dropped). **The boundary subtlety (the real Step-2 work):** the fade is
  per-segment (`fade_params_init(length, SPECTRAL_FADE_SAMPLES_ACTIVE)` in every
  `segment_fn_*`, spectral_synth_cpu.c:381+). The IFFT activity boundary does NOT yield
  a sharp edge — COLA coverage ramps 0→1 over ~one frame (n_fft) as successive frames
  start including the partial, and `fade_len` is generally narrower than n_fft. So a
  simple interior/edge truncation leaves a COLA ramp, not the intended fade. The correct
  decomposition is **additive**: IFFT renders the full steady span `[start,end]`
  (coverage = COLA_coverage(t)); the osc renders the seam correction
  `(fade(t) − COLA_coverage(t))·steady` over the boundary region (width ≈ max(fade_len,
  n_fft)), which is 0 in the deep interior. COLA_coverage(t) = Σ_active-frames
  window(t − m·hop) — exact and computable, but it MUST match or the seam steps
  audibly. The tolerance test must assert seam continuity, not just bulk parity. This
  boundary DSP is why Step 2 is its own focused unit.

- **Step 2 v1 LANDED (pass 259).** `core/spectral_synth_hybrid.{c,h}` —
  `spectral_synth_hybrid_try_render()` as an opt-in FAST PATH (returns RENDERED or
  DECLINED-untouched; the caller keeps synth_cpu as the fallback, so the default path
  is byte-for-byte unchanged). The boundary problem is sidestepped for v1 by the
  COLA-as-fade insight: the activity-boundary ramp IS a smooth fade, so requiring
  `length ≥ 4·n_fft` makes it a negligible fraction of the partial life and no seam
  correction is needed. Classifier `spectral_synth_hybrid_segment_eligible` (no chirp /
  no cubic / ~const amp / in-domain bin / long enough); global gates reject non-sine
  timbre + non-identity stretch/pitch; map `phase0 = phase − omega·start`. ctest
  `synth_hybrid_parity` (#25): deep-interior parity −66.6/−79.5 dBFS at 16 partials,
  eligibility + decline-untouched contracts. Test-target-only (no production caller; F3
  wires it). **Step-2 sequel (NOT built):** (a) the mixed case — additive seam
  correction `(fade − COLA_coverage)·steady` so dense interiors route to IFFT while the
  same buffer's faded/chirped/non-sine partials stay on the osc bank; (b) stretch/pitch
  support (apply the driver's param transform to the segment→partial map); (c) wire the
  fast path into the synth dispatch behind a runtime opt-in (rides F3 golden).

## Architecture clarifications (maintainer Q&A, 2026-06-15)

Recorded verbatim because they correct the mental model:

1. **Is the IFFT under the "fast math" path only?** No. It is NOT tied to the
   fast-math compile mode (`SPECTRAL_CUSTOM_FAST_MATH_MODE`). It is a separate
   *synthesis algorithm* exposed as an opt-in entry (`spectral_synth_hybrid_try_render`)
   with no production caller yet. It is opt-in because it is an APPROXIMATION whose
   default-on flip needs a listening test (F3), not because of any fast-math flag.

2. **Is the IFFT an optimization for the oscillator (synthesis) stage?** Yes,
   exactly. Same input (partials/segments), same output (audio samples), a faster
   *approximate* algorithm: instead of N per-partial oscillators, build the frame's
   half-spectrum (place each partial's window main-lobe motif), ONE inverse FFT, then
   overlap-add. It replaces the resynthesis inner loop. The ANALYSIS stage (STFT →
   peaks → segments) is untouched.

3. **Can it work on any build path?** The ALGORITHM is build-agnostic — that is the
   point of the port contract (`spectral_synth_ifft.h`: one surface, one inverse-FFT
   primitive implemented per backend). The current IMPLEMENTATION is desktop-float
   only (host vDSP/ref FFT, `malloc`, `cosf`/`sinf`). Each path needs its FFT primitive
   ported: desktop ✓; embedded = F4 (float-on-M7, static alloc, CMSIS/ref FFT);
   GPU = F7 (a GPU FFT kernel). Designed for any path, realized so far only on desktop.

4. **Does it need to work on streamed partials with no embedded control surface?**
   The embedded render IS streaming — `spectral_arm32_process` renders block-by-block
   from the audio callback into a Q63 accumulator. So the IFFT needs a STREAMING
   variant (F4b): persistent half-frame OLA tail + frame phase carried in the ctx
   across blocks (the current renderer is whole-buffer). The control surface DOES
   exist — `spectral_arm32_process` + the ctx — we extend it with an eligibility gate
   and OLA state. It is integration work, not a missing surface.

**On "is the quality degradation really so bad it can't be default?"** — Likely not,
and it is tunable. Measured floors at K=8 taps: −55 dBFS frame-peak / −83 dBFS stream
RMS. −83 dB RMS is very quiet; the −55 dB frame PEAKS are the risk in sparse/quiet
passages. On Q15 (16-bit, ~−90 dBFS quantization floor) the IFFT floor sits ABOVE the
fixed-point floor, so the IFFT becomes the dominant error vs the exact oscillator. The
dials that make on-by-default sound: **(i)** only engage at high partial density (the
router already does — the speed win and the error-averaging both need density);
**(ii)** raise K (~−8 dB per +4 taps, ~2K MACs/partial) until the floor drops below the
Q15 floor for the target material; **(iii)** maintainer A/B golden listen (F3). So
"opt-in until F3" is process caution + a tunable tradeoff, not a verdict that the
quality is bad. The embedded default-on directive is the maintainer making the F3 call
for embedded — it should still be density-gated + K-chosen, not a blanket flip.

## Embedded integration (compat audit, 2026-06-15)

- **Port decision: float-on-M7-FPU, not Q31.** The M7 has hardware single-precision
  float; `arch/arm/spectral_oscillator_cmsis.c` already uses `arm_sin_f32`; a Q15
  motif would lose ~12–15 dB SNR at dense densities (the motif interp is linear-float).
- **Blockers to clear (F4a):** `spectral_ifft_synth_create` heap-allocates (6×
  malloc + 2× calloc, spectral_synth_ifft.c:39-79); both FFT backends `calloc` state
  and the ref backend computes twiddles at runtime; `cosf`/`sinf`/`fmod`/`floorf`
  throughout. All forbidden libc-free. → static/pool buffers, a no-libm FFT
  (`arm_rfft_fast_f32` or static-twiddle ref), LUT trig.
- **Streaming (F4b):** persistent OLA tail + frame phase in the ctx across blocks.
- **Perf gate:** flipping the default arm32 render path moves codegen + cycles;
  regenerate the m7 baseline — the speedup BECOMES the gate. llvm-mca alone can't model
  the OLA/cache; a qemu trace of the IFFT kernel is needed for honest M7 cycles.
- **Accuracy gating (the accuracy reader's verdict — "conditional, not free"):**
  require a density profile gate (>80% of frames with ≥7 eligible partials in target
  material for the win to be real), K chosen so K-tap floor is inaudible in context,
  and the F3 golden audition for the embedded set.

## Parallelization — SIMDe / vDSP / GPU (F6/F7)

**Yes, the FFT⁻¹ synthesis is highly parallelizable** — all three pipeline stages are
data-parallel:

1. **Spectral construction** (place each partial's motif into bins): parallel across
   partials (scatter-add) or across bins (gather). Per partial the 16-tap write
   (K=8) is *contiguous* in `re[]`/`im[]`, so it SIMD-vectorizes (load 16 motif taps,
   ×(−1)^k sign vector, ×cr/ci broadcast, add to the contiguous bin slice). The
   per-partial trig (`cr=½·amp·cosf φ`, `ci=½·amp·sinf φ`) vectorizes across partials
   with a SIMD sincos.
2. **Inverse FFT** — already the canonical parallel primitive (vDSP on Apple, a
   batched FFT on GPU). O(N log N) per frame regardless of partial count.
3. **Overlap-add** — per-sample independent; a pure vector add (`vDSP_vadd` / SIMD).

**Literature.** Base method: **Rodet & Depalle (1992), "Spectral Envelopes and Inverse
FFT Synthesis"** (AES 93rd) — the FFT⁻¹ additive method, ~15× over oscillators on CPU;
**Freed, Rodet, Depalle (1992/93), "Synthesis and Control of Hundreds of Sinusoidal
Partials on a Workstation"** — the first real-time transform-domain synthesizer.
Parallel/GPU technique: **Savioja, Välimäki & Smith (2010), "Real-Time Additive
Synthesis with One Million Sinusoids Using a GPU"** (AES 128th) + their **JAES 2011
"Audio Signal Processing Using Graphics Processing Units"** — additive synthesis is
"embarrassingly data parallel"; ~1000× CPU by computing many partials AND many adjacent
output samples in parallel. For dense spectra the GPU FFT⁻¹ (parallel bin placement +
batched FFT + parallel OLA) beats brute-force GPU oscillators because the FFT is
O(N log N) independent of partial count. (Add these to `reference/ACADEMIC_SOURCES.md`.)

**Bottleneck note (measured):** on desktop the inverse FFT is already vDSP, so the
remaining CPU hot spot is the placement — specifically the per-partial `cosf`/`sinf` and
the 16-tap scatter. F6 targets exactly those; F7 moves the whole pipeline to the GPU.

**Sequencing:** F6a — SIMDe-vectorize the placement tap-scatter (bit-parity vs scalar;
the existing parity ctest #24 pins it) + the OLA → bench the placement speedup. F6b —
vectorized sincos across partials (the larger CPU lever). F6c — embedded uses CMSIS-DSP
`arm_*_f32` (no NEON on M7). F7 — GPU Metal/CUDA FFT⁻¹ with a GPU↔CPU parity ctest in
the #26 pattern.
