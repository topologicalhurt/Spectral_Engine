# IFFT Synthesis (Rodet–Depalle) — F-stream plan

**PREMISE REVISED 2026-06-21 — the "512 voices needs the IFFT" motivation is REFUTED.** The
embedded-perf campaign's high-word-multiply `coupled_step` (commit d02c817) fits 512 voices in ~98%
of the 400 MHz budget on the EXACT per-partial oscillator bank — so the IFFT is no longer *required*
for the 512 capacity target. The IFFT is **not dead**: the crossover-pricing argument below still
holds (it remains a ~7.5–8× density optimization for dense stationary material), and it is now the
**substrate for the renderer abstraction's wavetable + subtractive frequency-domain paths** (renderer
commits 078fee2 / 6232044 / 9be448f / c40c5c18; see `RENDERER_ABSTRACTION_PLAN.md` + AI_CANON #31).
Status: F1 / F2 / F2b-Step1 / F6 DONE; **F3 golden is the single open maintainer gate**; F4 / F5 / F7
and the F2b mixed-case ride behind F3 (dead code with no production caller until F3 wires one). So its
urgency shifted from "required for 512" to "optional density optimization + renderer substrate."

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

## File organization (pass-256 ruling; paths trued to the post-KERNEL-LAYOUT tree)

One contract header `core/spectral_synth_ifft.h` in OUR types; per-backend TUs
in their layer, never macro mono-files: `drivers/vdsp/spectral_ifft_vdsp.c`
(+ FFTW variant), the portable radix-2 `arch/ref/spectral_ifft_ref.c`, and the
F4 CMSIS embedded port (float-on-M7, NOT Q31). Third-party headers do not leak
through the contract; every port pair gets a parity test (`window_backend_parity`
pattern). The motif table is generated (SSOT generator, committed artifact —
resource-hash pattern). [Earlier drafts named `synth/`, `port/host/`, `port/cmsis/`
— those dirs were dissolved by the L2/L3 kernel-layout refactor (CHANGELOG pass 262).]

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
  - **GPU caveat (when precise-phase/cubic is wired here).** The additive GPU
    path packs the 32-byte `SegmentGpu` (omega/df only — quadratic phase; no
    cubic c2/c3, `spectral_common.h:99`), so a cubic-annotated render dispatched
    to Metal/CUDA would silently drop to the quadratic model and diverge from the
    CPU. There is no guard today (the dispatch gate, `gpu_check_timbre_or_fallback`,
    checks timbre only) — harmless while `SPECTRAL_PRECISE_PHASE=0` (cubic dormant,
    c3≡0). Wiring precise-phase must therefore EITHER widen `SegmentGpu` + the
    codegen'd MSL to carry c2/c3 (port the cubic DSP onto the GPU) OR add a
    dispatch guard that defers cubic-annotated renders to CPU. `gpu_backend_parity`
    would catch the divergence under a `PRECISE_PHASE=1` build with a
    non-stationary fixture (speech), which is the natural regression gate.
- **F4 — embedded (Cortex-M7) port + DEFAULT-ON for embedded (maintainer
  directive 2026-06-15).** The compat audit (see "Embedded integration" below)
  ruled **float-on-M7-FPU over Q31**: the M7 has a hardware single-precision FPU,
  the CMSIS oscillator already uses `arm_sin_f32`, and a Q15 motif table would
  lose ~12–15 dB SNR at dense densities. So F4 = a float embedded port of
  `spectral_synth_ifft.c`, NOT a Q31 rewrite. Three sub-pieces:
  - **F4a — static allocation: SYNTH STRUCT done (pass 259); 3 blockers remain.**
    LANDED: `spectral_ifft_synth_pool_bytes(n_fft)` + `spectral_ifft_synth_init(pool,
    bytes, n_fft, backend)` carve the struct + all scratch from a caller pool (16-byte
    aligned, no malloc); `create()` refactored to ONE pool malloc + `#if !SPECTRAL_EMBEDDED`
    (embedded uses init); `destroy()` is a no-op on init pools. ctest rung 6: init render
    == create render BIT-IDENTICAL + undersized/null rejects. The F6 trig/fmod work already
    removed the hot path's `sinf`/`fmod` libm calls (`floorf` remains but lowers to a single
    round intrinsic, no libc call). REMAINING F4a blockers (each its own step): (i) the motif/
    window build calls double `cos` (libm) at init — needs a COMMITTED precomputed table
    (the plan's "generated motif", resource-hash pattern) or the no-libm poly; (ii) the
    render's `memset(re/im)` → `spectral_mem_zero`; (iii) the FFT BACKEND — the ref backend
    still mallocs + runtime-computes twiddles, so a static-alloc backend init (CMSIS
    `arm_rfft_fast_f32` or static-twiddle ref) is needed (the init() already takes a
    caller-made backend, so this is a per-port `backend_pool_bytes`/`backend_init`).
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
- **F6 — CPU optimization — LANDED (pass 259), measured ~2.6× (24.0 → 9.1 ns/sample;
  IFFT-vs-osc 9.0× → 23.7×).** Three wins (trig, fmod, motif) after two measure-first
  declines. **Win C — motif scatter:** the per-tap `motif_eval` recomputed the table index
  + lerp weight, but the index advances by exactly `O` per tap and the weight is
  tap-independent, so `place_partial_cri` hoists the lookup setup and strides the lerp —
  bit-identical (gated `frac>0`: an exact-integer bin would index one past the motif table,
  so it falls to the per-tap path; ctest rung 5 covers it). 13.2 → 9.1: the scatter
  arithmetic WAS on the critical path (not load-bound as feared). Wins A+B below:
  (1) tap-scatter SIMD = no win (~2%, declined); (2) scalar poly swap = no win (the
  build's `-ffast-math` libm `sinf` is already ~2.4 ns/eval). **Win A — SIMD trig:** a
  microbench showed a 4-wide minimax sin at **6.7× the scalar sinf** (the renderer's trig
  was scalar AND interleaved with the scatter, so the compiler could not vectorize it).
  `ifft_render_frame` now runs **Phase A** (blocked phase reduction, then `ifft_fast_sin4`
  SIMD `cr/ci`) + **Phase B** (scatter via `place_partial_cri`). 24.0 → 18.5. **Win B —
  fmod kill:** the per-partial double libm `fmod` was a HIDDEN ~5 ns/sample (not ~2);
  replacing it with an inline round-subtract reduction to [−π,π] (libm-free, and tighter
  than fmod's [0,2π) so the minimax sin stays in its 1.4-ULP range) gave 18.5 → 13.2.
  #24 parity unchanged at −72.81/−87.46 (SIMD minimax ~3 ULP, far below the floor).
  Host-only + `APPROX_TRIG`-gated (exact/embedded take the scalar SSOT path; the inline
  reduction is libm-free so it serves F4 too). What remains: the **motif-scatter
  (`motif_eval` gather, now ~9 ns, the largest term)** — gather-bound, hard on CPU (the
  SIMD-add was already declined); it is the GPU's domain (F7). The inverse FFT (~3.7 ns)
  is the next-largest and already vDSP.
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
- F2 core CLOSED (pass 258): contract `core/spectral_synth_ifft.h` +
  frame renderer `core/spectral_synth_ifft.c` (float port of the F1
  formulation, K=8/O=16 measured operating point) + two iFFT ports
  (`drivers/vdsp/spectral_ifft_vdsp.c`, `arch/ref/spectral_ifft_ref.c` — exactly
  one live per build, selected on SPECTRAL_USE_VDSP). ctest
  `ifft_synth_parity`: port-vs-reference-iDFT 5.6e-9; stream parity
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

## F2b — implementation design (characterized 2026-06-15; Step 1 + Step 2 v1 LANDED, mixed case NOT yet built — see the build log below)

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
  live this frame". Keep the existing static API (its parity test ifft_synth_parity stays the
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

**Bottleneck note (MEASURED, pass 259 — `bench_ifft_synth` cost breakdown).**
[This is the PRE-F6 deep-research snapshot. F6 (Win A/B/C) subsequently landed the full
24.0 → 9.1 ns/sample cascade — SIMD trig, fmod kill, AND the motif-scatter arithmetic
hoist (Win C); see the F6 phase. The ~9 ns residual below is the motif GATHER (table
loads), which remains the largest term and is the GPU's domain (F7).] Hard
numbers at 64 partials, N_FFT=512, desktop host port (the FFT is data-independent, so it
was isolated by timing the backend on a random spectrum):
- full IFFT render: **23.9 ns/sample**
- inverse FFT alone: **3.76 ns/sample → only ~16%**
- ⇒ placement + per-frame spectrum clear + OLA = **~84%**.
This INVERTS the naive "the FFT dominates" intuition — the FFT is already vDSP-parallel
and is the *small* part. **CORRECTION (post-F6 measurement):** an early estimate here put
the trig at ~71% off a ~35 ns/trig guess — WRONG. The build's `-ffast-math` libm `sinf`
is ~2.4 ns/eval, so the scalar-but-interleaved trig was ~6.5 ns/sample (~27%); the F6 SIMD
batching removed ~5.5 of it (24.0 → 18.5, see the F6 phase). The genuinely largest term is
the **motif scatter (`motif_eval` per tap, ~10 ns/sample)** — that, not the trig, is the
next CPU lever. Corollaries that held: (i) SIMDe-vectorizing the 16-tap scatter ADD was
bit-identical but zero-win (~2%) — **DECLINED**; (ii) the scalar poly swap was also zero-win
(libm already fast) — only the SIMD-across-partials batching paid off, because the
interleaved scalar trig was un-vectorizable in place.

**Sequencing (revised by the measurement):**
- **F6a — per-partial trig (the real lever).** Either a SIMD `sincos` evaluating 4
  partials' `cr/ci` per call (needs a vectorizable poly — check `spectral_fast_math`),
  or per-partial phase recursion (rotate `(cr,ci)` by `ω·hop` across frames instead of
  recomputing — removes the per-frame trig; tolerance-tested, not bit-exact, drift ≪
  the IFFT floor over ~16 frames). Restructures the frame loop to process partials in
  groups. Bench-gated: must show a real ns/sample drop or it is declined too.
- **F6b — embedded** uses CMSIS-DSP `arm_*_f32` + the dual-issue FPU (no NEON on M7).
- **F7 — GPU Metal/CUDA FFT⁻¹** (the big lever, Savioja): parallel bin scatter +
  batched FFT + parallel OLA, with a GPU↔CPU parity ctest in the #26 pattern.
- (The contiguous-scatter SIMD + OLA-vadd remain available but unmotivated until a
  profile shows them mattering — recorded here so they are not re-attempted blind.)

## Parallelization roadmap (deep-research pass 259 — 5-angle workflow + adversarial critique)

A 7-agent dig (frame-parallel, pipeline/divide-conquer, GPU end-to-end, SIMD trig, other-
engine survey → synthesize → adversarial critique that **independently re-measured**).
Headline reframe, MEASURED: full IFFT 25.1 ns/sample, **inverse FFT only 3.7 (≈15%)**,
trig+fmod ≈ 10 of the 17.5 ns placement. **The FFT is not the bottleneck — the per-partial
trig is.** So parallelizing the *FFT* (GPU/batched) barely helps; the lever is the trig.

**The load-bearing axis is real-time vs offline:**
- **Real-time / streaming** (embedded M7 `spectral_arm32_process`, ~48–256-sample blocks =
  ~1 hop = ONE IFFT frame, no lookahead): the ONLY applicable levers are **within-frame** —
  fmod-fold + SIMD/CMSIS sincos. Frame-threading, batched FFT, and GPU all have ZERO value
  (no frames to batch; OpenMP fork/join and ~30–60 µs GPU launch blow a 1–5 ms budget).
- **Offline / bulk file render** (whole timeline known): everything applies — but see the
  blocker below.

**Validated sequence (re-ranked by the critique):**
1. **THE shippable win — within-frame trig on the SHARED `ifft_render_frame` — LANDED
   (pass 259), MEASURED ~1.3× (24.0 → 18.5 ns/sample, IFFT-vs-osc 9.0× → 11.6×).** Phase-A
   (blocked scalar+double phase reduction, then `cr/ci` via a SIMD minimax `ifft_fast_sin4` —
   SSOT coefficients, host-only, `APPROX_TRIG`-gated) + Phase-B (`place_partial_cri` scatter).
   #24 parity unchanged (~3 ULP, tolerance not byte parity). NOTE vs the original spec: the
   double `fmod` was KEPT (it is load-bearing for precision — phi is thousands of rad and the
   poly fold would lose ~1e-3 on the float); only the sin/cos was vectorized, which was the
   whole win because the interleaved scalar trig was un-vectorizable in place. A standalone
   `ifft_fast_sin4` was used rather than instantiating the templated `osc_vfastsin` (lighter;
   same SSOT coefficients). Target was ~1.4–1.5×; delivered ~1.3× (the scatter eats the rest).
2. **Phase recursion (coupled-form rotor) — DEPRIORITIZED.** Elegant and trig-killing, but
   it carries per-partial state across frames, which only the STATIC path can do — and the
   static path has **zero production callers**. The dynamic/hybrid path rebuilds the partial
   set per frame, so a rotor would have to key state to segment identity inside the hybrid
   (a bigger change). Revisit only if/when the hybrid grows persistent per-segment state.
3. **Offline tier — ALL blocked behind F2b/F3 pipeline wiring (no production caller today),
   so it is dead code until the hybrid is wired into the render path.** In order, after wiring:
   chunked-OLA frame-parallel (clone the analysis-fused OMP loop + synth_cpu's span/thread-
   buffer reduction; **prerequisite: hoist `s->re/im/frame` to per-thread scratch**; halo =
   exactly `n_fft`; reproduce the ascending float-add order) → batched `vDSP_fftm_zrip` (near-
   worthless alone: ~1.1× net on a 15% slice) → GPU FFT⁻¹ backend (privatized per-frame
   scatter + batched C2R + gather-OLA; the incremental win over the EXISTING GPU oscillator
   bank is only the ~8–15× algorithmic factor on dense stationary material; hot-bin scatter
   contention + packing-parity risk; sequence last + golden-gated).
4. **Split OUT as an independent subsystem: GPU-batched ANALYSIS forward-STFT.** The single
   CPU-only FFT stage (grep: ZERO cuFFT/MPS/VkFFT in-tree despite a full Metal+CUDA *synthesis*
   backend). Textbook batched-1D-FFT GPU workload, gated on the existing `STFT_CHUNK_THRESHOLD`,
   offline/transfer-bound — no dependency on the IFFT work; highest other-engine ceiling.

**Other untapped CPU SIMD (measure-first conditionals):** `segment_fn_wavetable_float`
(spectral_synth_cpu.c) is still scalar per-sample while the algebraic timbres are SIMD
(~1.3–1.6×, but NEON has no native gather — measure before building); per-candidate peak
estimator SIMD (sparse → low payoff); multi-file resource hashing (build-time, low value).

**Risks to honor:** tolerance-not-bit-exact (both trig moves); coupled-form amplitude drift
over thousands of streaming frames (periodic renorm); per-thread-scratch data race + halo
off-by-one + float-add-order in chunked-OLA; GPU real-complex packing/scale parity vs the
vDSP zrip convention; GPU hot-bin scatter contention; **and the overarching one — none of
the offline/GPU tier delivers value until F2b/F3 gives the IFFT a production caller.**
