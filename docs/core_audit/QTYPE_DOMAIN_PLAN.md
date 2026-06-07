# Q-Type Domain & SIMD-Width Plan — fixed-point as a first-class, desktop-available domain

**Status:** IN PROGRESS — **Q0 SHIPPED (PASS206)**; **Q1 CLOSED BY AUDIT** (storage
already Q15-clean + CI-pinned; no free packing wins exist — see §5); **Q2 SHIPPED
(PASS207)** (float L1 width-parameterized over `SIMDE_NATURAL_FLOAT_VECTOR_SIZE`; Mac
byte-identical, x86 8-wide within budget; `osc_width_parity` CTest landed); **Q3 is the
live decision, golden-gated** (opt-in Q15 compute — needs maintainer sign-off per path).
The unification track
(U1c → U1d → U3 → U2) completed first, so this lands on the already-unified L1
oscillator and L1 is extracted **once**, not twice (maintainer ordering,
2026-06-03 "finish unification first"). Four design forks below are signed off.

**Q0 — DONE (PASS206, ctest 6/6 + 5 builds green).** `spectral_q15.h` is the
documented canonical Q-domain header (two-layer storage-vs-compute model +
boundary-macro rule + `SPECTRAL_Q_DOMAIN BEGIN/END` region convention, with the
live Q15-op block wrapped). New `q_domain_contract` CTest (pure `cmake -P` scan,
no compiler/Python) enforces both no-mixing directions: RULE 1 confines the raw
scale constants to allowlisted boundary sites; RULE 2 forbids float/double inside
a Q-domain region. One real leak fixed (`wavetable.c:254` → `SPECTRAL_SAMPLE_TO_FLOAT`,
byte-identical). Tripwire proven both directions. See CHANGELOG Pass 206.

## What the user asked for (verbatim intent, 2026-06-03)

> (1) Q15 etc. must be accessible on desktop builds as well and not just embedded
> ones; every path that could benefit on desktop should also leverage packing into
> Q15. (2) Clean separation between the types — readability, maintainability,
> robustness — but ALSO we don't want to mix floats/doubles/ints and q types unless
> absolutely necessary. (3) Refactor out how we're doing this currently — major
> source of tech debt. (4) Abstract/generalize, DON'T BE MYOPIC: in theory SIMD
> could pack 2x as much information. (5) Would it be more convenient to evaluate
> 'q16' as a short on desktop? … the exponent can be small (sin amplitude [-1,1],
> phase truncates to [-pi,pi] / [-2pi,2pi]). Custom type? Portability? Industry
> standard? (6) Does SimDe have a way to use the maximum width available? We use
> 128-wide by default but would want AVX-512/AVX2 where present.

## 1. Current landscape (measured 2026-06-03)

- **Q-types already exist and already compile on desktop.** `synth/math/spectral_q15.h`
  defines the CMSIS-style set `q15_t`/`q31_t`/`uq16_t`/`uq32_t` (lines 13-16) with a
  portable C fallback and an `__ARM_FEATURE_DSP` intrinsic path (`__qadd16`/`__smlad`/
  `__ssat`). Scales live in `core/spectral_consts.h:41-44`. So ask (1) is **not**
  "make Q15 compile on desktop" — it's "make desktop hot paths *choose* it."
- **No vectorized Q15 exists anywhere.** The embedded `core/port/embedded/oscillator_simd.c`
  is `float32_t` NEON (not packed Q15); the arm32 Q15 oscillator has no `__smlad`
  lane-packing. Ask (4) "SIMD could pack 2x" is unrealized on **every** backend — greenfield.
- **SIMD width is hard-pinned to 128-bit.** Every host kernel is `simde__m128` (4 float
  lanes) while `cmake/host-config.cmake:20-23` compiles `-march=native -mavx2
  -mno-avx512f`. On AVX2 x86 the hardware has 256-bit registers and we use half. On
  the maintainer's Mac (NEON) 128 is the ceiling, so no loss there — the width payoff
  is x86 desktop/CI only.

## 2. The precision crux (drives the whole "layered" design)

Q15 has 15 fractional bits → quantization step 2^-15 ≈ 3.05e-5 → SNR ceiling
≈ 6.02·15 + 1.76 ≈ **92 dB**. The float oscillator path just reached **-155 dBFS**
(PASS201 minimax sine). **Computing audio *in* Q15 throws away ~60 dB of headroom
we just fought for.** Therefore Q15 is a *throughput/storage* win at a *precision*
cost, and "every path that could benefit" must split:

- **Storage/transport packing** — boundaries that are *already* int16 (final PCM
  out, segment storage, sine LUT). Q15 here is free / lossless at the boundary.
- **Compute-in-Q15 intermediates** — lossy; only where 15-bit is *provably* enough
  and the path is throughput- or bandwidth-bound. Opt-in, float stays default.

This honors the standing note *don't force-unify Q15 with the float domain*: Q15 is
a **second domain selected per-path**, not a blanket replacement.

## 3. The packing synergy (asks 1 + 4 + 6 are one lever)

A 128-bit register holds **4× float32 OR 8× Q15**. Packing to Q15 is itself a 2×
SIMD-throughput win at the same register width; going native 256-bit (AVX2) is
another 2×. Stacked: 16× Q15 in a 256-bit register = **4×** today's 4× float32@128.
That is why (1)/(4)/(6) are facets of one decision.

## 4. Signed-off design forks (2026-06-03)

### F1 — Q15 reach: **both, layered**
Storage packing wherever the boundary is already int16, *plus* opt-in Q15 compute
for throughput-bound kernels behind a flag, float default. Needs the cleanest
domain separation (see §5 type discipline).

### F2 — Type safety: **transparent CMSIS typedef + boundary macros + CI contract test**
Keep the transparent integer typedef (zero-cost, SIMD load/store-friendly,
CMSIS-aligned — already the embedded convention). Enforce "no mixing" by *structure*:
all float↔Q conversions go through named boundary macros (`FLOAT_TO_Q15`/`Q15_TO_FLOAT`
/`PHASE_RAD_TO_Q15` already exist), and a CI **contract test** greps that the Q
kernels contain no `float`/`double` arithmetic and the float kernels no raw `q15_t`
math outside the boundary macros. **No newtype/struct wrapper in the kernels** — in C
it kills arithmetic operators and SIMD ergonomics in the hot loop. (A C++ newtype is
allowed *only* in desktop tooling/analysis if ever wanted; not in the kernels.)

Rationale on (5): a bare `short` discards the Q-format contract (the binary-point
location *is* the type), so you always need a carrier. The CMSIS typedef + convention
**is** the industry standard (ARM CMSIS-DSP `q7/q15/q31/q63`, TI, ADI). For **phase**
specifically, an *unsigned* `uq16_t`/`uq32_t` accumulator is the textbook NCO idiom —
wrap is free (mod 2^16/2^32), no `fmod` — and the repo already does this
(`spectral_q15.h:204-217` `phase_acc` is `uq32_t`). Generalize that; don't reinvent it.

### F3 — SIMD width: **`SIMDE_NATURAL_*_VECTOR_SIZE`-parameterized kernels**
SIMDe auto-*detects* max native width via `SIMDE_NATURAL_VECTOR_SIZE`
(`simde-features.h:560-608`): 512 (AVX-512F) / 256 (AVX2) / 128 (SSE2, NEON, …),
with `_GE(x)`/`_LE(x)` predicates. With `-march=native` that macro **is** "the widest
this CPU has." Author one kernel body per width tier (typically just 128 and 256)
selected by `#if SIMDE_NATURAL_FLOAT_VECTOR_SIZE_GE(256)`.

Two hard nuances baked into this header:
- **Width is type-dependent.** On plain AVX, float is 256 but **int is 128**; 256-bit
  *integer* lanes need **AVX2**. So Q15 kernels key off `SIMDE_NATURAL_INT_VECTOR_SIZE`,
  float kernels off `SIMDE_NATURAL_FLOAT_VECTOR_SIZE` — they can differ on one CPU.
- SIMDe has **no** auto-scaling single vector type (no Highway `ScalableTag`) and **no**
  runtime CPU dispatch. A portable *binary* that picks AVX-512-vs-AVX2 at startup would
  need FMV (`__attribute__((target_clones))`) or Highway — **out of scope** for the
  native build; only revisit if shipping a portable binary.
- `-mno-avx512f` (`host-config.cmake:23`) currently caps detection at 256 even on
  AVX-512 silicon — a deliberate knob to revisit during execution.
- Note: the *scalar fallback* loops already auto-vectorize to max width under
  `-march=native -ftree-vectorize`. Only the hand-written 128-bit intrinsic kernels
  are pinned — those are the sole target of the width work.

### F4 — Sequencing: **after unification** (U1c → U1d → U3 → U2 first).

## 5. Coarse phase outline (refine at execution time, post-unification)

> Kept high-level on purpose: U1c extracts the shared L1 oscillator kernel, which
> reshapes exactly the loop bodies this phase parameterizes. Detail after U1c lands.

- **Q0 — domain doc + type discipline. ✓ DONE (PASS206).** `spectral_q15.h` documented
  as the canonical Q-domain header (two layers + boundary-macro rule + `SPECTRAL_Q_DOMAIN`
  region markers). CI contract test `q_domain_contract` landed (RULE 1 scale-constant
  confinement, RULE 2 region float-purity) — pins "no float/Q mixing outside boundary
  macros" before any kernel moves. `wavetable.c:254` routed through the boundary macro.
- **Q1 — storage/transport packing. ✓ CLOSED BY AUDIT (2026-06-03, no code change).**
  The "free desktop packing wins" premise did **not** survive contact with the code —
  the int16 boundaries that exist are embedded-only and already Q15-clean/macro-routed,
  and the desktop simply has no int16 storage to pack. Per-site:
  - **Segment store** (`Segment` float → `SpectralSegmentQ15`): both conversion sites,
    `convert_segments.c:340-347` and the sim's `segment_to_q15`
    (`spectral_synth_simulation.c:142-148`), **already** route every field through the
    boundary macros (`OMEGA_TO_Q88`/`PHASE_RAD_TO_Q15`/`FLOAT_TO_Q15`). Already packed.
  - **LUT** (`spectral_lut_flash`/`spectral_lut_sin`): already `q15_t`, but consumed
    ONLY by arm32 + sim. No desktop-float path reads it (desktop uses the PASS201 minimax
    poly). "Pack the LUT on desktop" = compute-in-Q15 = **Q3** (lossy/golden), not a free win.
  - **PCM out**: desktop writes 32-bit FLOAT WAV (`SF_FORMAT_FLOAT`/`sf_writef_float`,
    `spectral_out.c:202,215`) — no existing int16 boundary to repack. Embedded/sim output
    is already int16/Q15 via `FLOAT_TO_SPECTRAL_SAMPLE`. A 16-bit desktop output is a NEW
    opt-in lossy feature, also Q3. (The lone desktop Q15-file→float load,
    `wavetable.c:254`, was macro-routed in Q0.)

  Net: the storage layer is already clean and, as of Q0, CI-pinned. Nothing byte-neutral
  left to repack. The remaining packing opportunities are lossy and fold into Q3.
- **Q2 — width parameterization of the *existing float* L1 kernel. ✓ DONE (PASS207,
  ctest 7/7 + 5 builds green).** The unified L1 SIMD sustain body was factored into a
  width-templated `core/port/host/oscillator_simd_kernel.inc` (includer defines `OSC_VW`
  4|8 + `OSC_VSUF`), with the width-independent scalar fade lanes split into
  `oscillator_simd_scalar_waves.h`. `oscillator_simd.c` instantiates the kernel at the
  machine's natural width: **8-wide `__m256` only on an `__AVX2__` + `SIMDE_NATURAL_FLOAT_
  VECTOR_SIZE_GE(256)` x86 target; 4-wide `__m128` everywhere else (SSE2, NEON).** New
  `osc_width_parity` CTest (test #6) force-instantiates BOTH widths in one TU via SIMDe's
  portable `__m256` (NEON-emulated on Apple Silicon) and asserts 8-wide == 4-wide ==
  scalar within the FMA-contraction budget (1e-5/sample). **Mac is byte-identical by
  construction** — the W=4 macro expansion is token-equivalent to the pre-Q2 hand-written
  `simde_mm_*` kernel; `osc_parity` reports the captured baseline to every digit (sine
  5.960e-08, quantized 2.384e-07, RMS -155.1 dBFS). x86-only speedup; no precision change.
  See CHANGELOG Pass 207.
- **Q3 — opt-in Q15 *compute* domain for throughput-bound kernels.** Add the Q15 L1
  twin behind a per-path flag, keyed off `SIMDE_NATURAL_INT_VECTOR_SIZE`, with `__smlad`/
  `__qadd16` packing (8×Q15@128 / 16×Q15@256). Each enabled path carries a measured
  precision justification (dBFS vs float reference) — the same "measure don't assert"
  bar as PASS200/201. Float stays default; Q15 is opt-in per path.
  - **Q3a — measure-first (DONE, PASS208, ctest #8 `q15_compute_precision`).** New
    golden-neutral characterization harness renders the L1 waveforms in pure Q15 and
    against the float L0 formulas (same phase, so it isolates Q15 *waveform-eval*
    precision — NCO phase resolution is an orthogonal axis, deferred). Measured RMS
    error vs float (amp 0.8, fs 48k, full-scale local sine LUT so the production
    32700-headroom gain doesn't mask the floor):

    | timbre   | RMS err (dBFS) | peak err |
    |----------|----------------|----------|
    | sine     | **-85.1**      | 1.50e-4  |
    | saw      | **-91.5**      | 6.10e-5  |
    | square   | **-90.0**      | 4.27e-5  |
    | triangle | **-92.6**      | 6.10e-5  |
    | parabola | **-91.0**      | 6.10e-5  |

    All five sit at the Q15 quantization floor (~-90 dBFS); sine is ~7 dB higher purely
    from the 12-bit-LUT + 8-bit-interp residual. **Verdict:** every one of the five clears
    a generous throughput-bound bar (all ≥ 25 dB below the -60 dBFS audibility-margin line
    and ~85–93 dB down absolute) — i.e. Q15 *compute* is precision-viable for the whole
    algebraic + LUT-sine L1 set. (quantized/pwm/asin excluded — width/transcendental, not
    Q15-compute candidates.) **Gate:** Q3b (production opt-in flag + SIMD `__smlad`/`__qadd16`
    packing) is the byte-moving step and waits for maintainer per-path sign-off (§7).
  - **Q3b sign-off: ALL 5 timbres AUTHORIZED (2026-06-04, AskUserQuestion).** Float stays
    default; Q15 opt-in per path; each path carries a production dBFS parity lock.
  - **Q3b slice 1 SHIPPED (PASS209, ctest 9/9 + desktop/simulate/simulate-daisy green; cuda
    env-blocked — no nvcc).** Per-path opt-in flag (`osc_set_q15_enable`/`OSC_Q15_BIT`/
    `osc_q15_available`, a per-timbre bitmask orthogonal to the float `OscDispatchWord`,
    default 0) + scalar Q15 sustain path `synth_segment_q15` wired into `timbre_synth_segment`
    after the band-limited return. Q15 evaluators promoted to the single-source header
    `core/spectral_osc_q15.h` (SPECTRAL_Q_DOMAIN-marked); Q3a `q15_compute_precision` refactored
    onto it (byte-identical PASS208 numbers — test==ship). New CTest **q15_production_parity**
    (#9) renders real segments through the production dispatch Q15-off vs Q15-on, locking each
    path's RMS error under a per-path budget ~12-13 dB above the Q3a floor (measured, with
    envelopes: sine -88.2, saw -98.1, square -96.4, triangle -98.0, parabola -98.8 dBFS — all
    below the floor since fades scale both paths down). Waveform-in-Q15 only; phase NCO stays
    float (deferred). Scalar path is the oracle for the SIMD kernel. **Default byte-identical
    by construction** (opt-in default-0, hot-path branch is control-flow only, no CLI wires it).
    See CHANGELOG Pass 209.
  - **Q3b slice 2 CLOSED — measure-first, SIMD Q15 kernel (#73) DECLINED on data (PASS210).**
    Did #75 first: `bench_q15_throughput` (EXCLUDE_FROM_ALL manual probe) renders the real
    production path under float-scalar / q15-scalar / float-simd. Result (ns/sample, Apple
    Silicon): Q15-scalar beats float-scalar ONLY for sine (0.51x, LUT vs poly); the four
    algebraic timbres LOSE (1.17–1.30x — the per-sample float→Q15→float round-trip costs more
    than the cheap float eval). Against the shipping float-SIMD default, the Q15 scalar oracle
    is 3.3–4.9x slower. **A Q15 SIMD kernel over FLOAT phase cannot beat float-SIMD:** the eval
    isn't the bottleneck (float-SIMD does it 4-wide, zero conversions); the only structural
    Q15 win is 8×int16 vs 4×float32 lane density, which is cancelled by the float→Q conversion
    unless the phase accumulator is integer — i.e. gated on the deferred integer-NCO axis. Sine
    (the lone scalar winner) loses hardest vectorized (8-way LUT gather, no efficient NEON
    form). So #73 is declined; the bench stays in-tree as reproducible evidence + the gate for
    any future integer-NCO attempt. ctest 9/9 + desktop/simulate/simulate-daisy green; default
    byte-identical (bench is EXCLUDE_FROM_ALL). See CHANGELOG Pass 210.
  - **Re-scoped follow-up — integer-NCO phase axis (deferred):** the prerequisite for any real
    Q15 throughput win (integer phase → 8-wide Q15 packing, no float↔Q conversions).
    `bench_q15_throughput` is its measurement gate if/when scoped. Footprint refinement still
    open: guard the ~8 KB `g_osc_q15_sine_lut` out of embedded (`#if !SPECTRAL_EMBEDDED`).
- **Q4 — verify + document.** 5 builds + ctest green; per-path speedup + dBFS tables;
  PATCH_NOTES; update this doc's status to shipped.
- **Q5 — integer-NCO phase axis (ACTIVE; the deferred prerequisite PASS210 surfaced).** The
  reason desktop Q15 exists at all is **double-lane packing** (8×Q15@128 = 2× the 4×float32
  density); PASS210 proved that win is unreachable over a *float* phase accumulator (the
  per-sample float→Q15 conversion cancels the density). Q5 makes the oscillator phase
  **integer**, so 8 Q15 phase indices come from integer adds with no conversion and the packed
  kernel can finally beat float-SIMD. This is a *port-and-widen*, not greenfield: the embedded
  backend already runs an integer NCO — `SpectralActiveSegQ15{ uq32_t phase_acc; q31_t
  freq_inc; q31_t freq_delta; }` and `spectral_phase_batch4()` in `spectral_synth_arm32.c`
  compute phases by integer add. Q5 lifts that model to a shared header and widens it.
  - **Scope decision (2026-06-04, AskUserQuestion): FULL CUBIC in integer — no float
    fallback.** Integer-NCO Q15 must carry the cubic `c3` (MQ track-linkage) phase too, via
    integer **forward-differencing** (embedded does only linear+quadratic via `freq_delta`;
    Q5 adds the third difference).
  - **Math (cubic forward-difference).** `θ(t) = phase0 + alpha·t + c2·t² + c3·t³`. Init at
    `t=0`: `p=phase0`, `Δ1=alpha+c2+c3`, `Δ2=2·c2+6·c3`, `Δ3=6·c3`. Per sample: emit `p`;
    `p+=Δ1`; `Δ1+=Δ2`; `Δ2+=Δ3` — the exact cubic in **3 integer adds/sample**, zero float on
    the hot path.
  - **Fixed-point precision (the real risk).** Phase domain = `uq32` (2π ≡ 2³², so the int
    wrap IS the mod-2π). But `Δ3=6·c3·scale` is sub-unit for realistic chirp → it would round
    to 0 and drop the cubic. So the accumulators carry fractional bits: **64-bit `Q32.32`** (`p`,
    `Δ1`, `Δ2`, `Δ3`), emit index = top bits of `p>>32`. The only error is the initial
    coefficient rounding to Q32.32; Q5a *characterizes* it (measure-first, not asserted).
  - **Throughput risk (why this stays measure-first).** The cubic costs 3 adds/sample in
    64-bit; whether packing still wins depends on that phase cost not erasing the 8-wide Q15
    eval-density advantage. SIMD phase has a sub-fork (8 parallel lane-offset cubic chains vs
    scalar-unroll-then-vector-eval) — settled by measurement at Q5c, **decline-on-data** if it
    loses, exactly as PASS210.
  - **Q5a — measure-first, golden-neutral (LANDED, PASS211):** `core/spectral_phase_nco.h` —
    the scalar cubic integer forward-difference phase primitive (uint64, full circle == 2^64,
    top 16 bits == signed Q15 index; `SPECTRAL_Q_DOMAIN`-marked step) — plus the
    `phase_nco_precision` CTest. **Gate PASSED, with margin.** Part 1 (index drift vs a double
    truth): the integer NCO holds **≤2 LSB** on every worst-case segment while the float-cubic
    path drifts further (**5 LSB** near Nyquist, 2 LSB on long-cubic) — integer phase is *tighter*
    than float exactly where total phase grows. Part 2 (cost of swapping the Q15 path's phase
    source, same evaluator): **sine −86.7, saw −93.3, square −300 (bit-identical), triangle
    −87.5, parabola −91.8 dBFS** — all at/below the PASS208 Q15-eval floor, i.e. the swap adds
    no audible error. The error budget of the axis is now *measured*, not asserted; Q5b is a GO.
  - **Q5b — LANDED (PASS212).** `synth_segment_q15` now phases by the integer NCO (init once per
    segment, `spectral_phase_nco_step` per sample) — the float cubic phase + per-sample
    float→Q15 conversion are gone from the opt-in Q15 path. The existing `q15_production_parity`
    CTest now exercises the NCO and stays inside budget (sine −90.1, saw −104.2, square −96.4
    bit-identical, triangle −92.1, parabola −96.7 dBFS). Default byte-identical (opt-in, mask
    defaults 0, no CLI). `bench_q15_throughput` re-measure: removing the conversion flipped the
    4 algebraic timbres from 1.17–1.30x *slower* than scalar-float to **0.88–0.97x** (tie/beat);
    still 2.27–3.61x off 4-wide float-SIMD — that gap is Q5c's packing, now feasible because the
    phase is integer. 5 builds + ctest 10/10 green.
  - **Q5c — LANDED (PASS213).** `core/port/host/oscillator_simd.c` gains `osc_simd_q15_segment` —
    the packed 8×Q15 kernel: 8 `int16` Q15 lanes evaluated per 128-bit register, 1-op
    `cvtepi16_epi32` widen, unchanged float amp ramp + accumulate; integer NCO stepped 8× per
    block for phase. Wired behind the opt-in Q15 mask AND SIMD dispatch (sine excluded — its
    serial LUT loses, 0.88 vs 0.71 ns/sample; stays scalar Q15). New CTest `q15_simd_parity` locks
    it to the scalar Q15 oracle: saw/square/triangle **bit-identical**, parabola ≤1 LSB (mulhrs
    round vs scalar truncate). Measured (`bench_q15_pack8`) **~1.2–1.4× over production float-SIMD**
    on the four algebraic timbres — the double-lane density win, realised. Short of the theoretical
    2× (needs a full-Q15 accumulate chain or native 16×Q15 codegen, neither on this 128-bit-NEON
    host). Two probe correctness fixes folded into production: triangle `MAX−2|pq|` via
    double-subtract (the probe clamped the upper half to 0), and a `pq` pre-clamp to `[−32767,32767]`
    so abs/mulhrs can't overflow at the −π corner. Default byte-identical, host-only.
  - **Bv — LANDED (PASS214), scoped on `c3 == 0`.** The deferred vectorized uint32 8-wide NCO phase
    (`core/spectral_phase_nco8.h`) replaces the kernel's serial scalar phase (8 sequential steps per
    block, ~31–35% of the kernel per PASS213) with 8 stride-8 cubic forward-difference chains in
    uint32 lanes — 8 consecutive Q15 indices per `simde__m128i` per step. Precision re-validation
    (`phase_nco_precision` Part 3) found uint32's 16 fractional bits hold the index **≤2 LSB** of the
    scalar uint64 NCO when **c3 == 0** (linear/quad — the default model, `c3` nonzero only under
    `SPECTRAL_PRECISE_PHASE` + track linkage), but drift to 44 LSB / −68 dBFS on long cubic. So
    `osc_simd_q15_segment` uses the vec NCO when `lp->c3 == 0.0f` (seeded once from the scalar NCO,
    sole phase source across SIMD-block + tail + fade via an 8-index buffer) and falls back to the
    bit-identical scalar pack8 when `c3 != 0`. `c3 == 0` is an EXACT boundary — zero precision
    regression. Measured **~1.42–1.57× over production float-SIMD** (saw 1.57, square 1.54, triangle
    1.56, parabola 1.42), up from the PASS213 1.25–1.39×. The `q15_simd_parity` cubic-stress row
    (c3=3e-7) is the CI lock on the scope gate: a misroute would jump to −25.9 dBFS, far past the
    −84 budget. Default byte-identical, host-only.
    - **Deferred follow-ups (decline-on-data as warranted):** (1) **LANDED (PASS216)** — the
      ~8 KB `g_osc_q15_sine_lut` `#if !SPECTRAL_EMBEDDED` footprint guard. `oscillator.c`'s whole
      desktop Q15 body (LUT storage + init, the scalar `osc_q15_eval`/`synth_segment_q15`, and the
      packed-SIMD dispatch) is now host-compiled only; real embedded firmware (which does Q15 in
      `spectral_synth_arm32.c`) carries none of it — 8194 bytes of `.bss` + the segment text removed.
      `osc_q15_available`/`osc_set_q15_enable`/`osc_get_q15_enable` stay in every build (the restricted
      CLI's `run_synthesis` references them). Desktop object code unchanged; `simulate` + `simulate_daisy`
      (EMBEDDED, +RESTRICTED) compile/link clean; ctest 11/11. (2) **still open** — with the vec phase,
      sine's pack8 (0.761) now edges production float-SIMD (0.876); sine stays excluded
      (`osc_simd_q15_available`) pending its own serial-LUT SIMD-Q15 re-validation.
  - **`--q15` CLI exposure — LANDED (PASS215).** The Q15 kernels were compiled into every desktop
    binary but unreachable from the shipped CLI (dispatch gated by `g_osc_q15_enable`, default 0,
    set only by harnesses). PASS215 adds the `--q15` opt-in flag (`SpectralCliOptions.enable_q15`,
    parsed in `spectral_cli.c`, wired in `run_synthesis`): when set and `osc_q15_available(timbre)`,
    it calls `osc_set_q15_enable(OSC_Q15_BIT(timbre))` and forces the CPU backend (Q15 lives on the
    CPU float-synth dispatch; GPU ignores the mask). Routes the packed 8-wide SIMD Q15 path on
    saw/square/triangle/parabola, scalar Q15 under `--scalar`, float on the no-Q15 timbres
    (asin/quantized/pwm). Pure plumbing over the CI-locked kernel — no kernel/precision change,
    float stays the desktop default, byte-identical without the flag. README documents it.

## 6. Open questions to resolve at execution (not now)

- Exact set of "throughput-bound" paths that clear the 15-bit precision bar for Q3
  (additive accumulation across many partials is the risky one — likely float-accumulate
  even if Q15-multiply).
- Whether to lift `-mno-avx512f`, and whether any AVX-512 path is worth a separate tier
  vs. stopping at 256.
- Whether the desktop tooling layer ever wants the optional C++ Q newtype (F2) — defer
  until a real mixing bug motivates it.

## 7. Standing constraints inherited

Golden-gated (maintainer is the golden authority; behavior/precision re-baselines need
sign-off). Measure-don't-assert. KISS. Don't force-unify Q15 with float. Plan-first;
maintainer sets order. See `OSCILLATOR_UNIFICATION_PLAN.md`, `ULTRAPLAN.md`, and the
memory notes `faster-path-should-default`, `avoid-assumptions`,
`algorithm-before-microopt`.
