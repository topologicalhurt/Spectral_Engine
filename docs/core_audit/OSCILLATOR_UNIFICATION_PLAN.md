# Oscillator Unification Plan — design pattern for one coherent oscillator system

**Status:** U-TRACK COMPLETE — design + ordering signed off by the maintainer.
U1a (doc), U1b (SIMD default, PASS200), the SIMD-vs-scalar parity ctest, the
default minimax-sine re-baseline (PASS201), U1c (L1 spectral_osc_eval, PASS202),
U1d (Metal MSL codegen, PASS203), U3 (oversample optimize, PASS204), and U2
(adversarial audit of the optimized band-limited file, PASS205) all done. All 5
P-problems addressed (P1 PASS200, P2 U1c, P3 vocab, P4 U1d, P5 = Q-type island,
handled by the QTYPE_DOMAIN phase next). See §4/§5.

## What the user asked for (verbatim intent)

> (1) We have so many different oscillator implementations now? It's getting
> confusing and we need to unify how the system works. Needs to be as
> maintainable / readable as possible. This goes for CPU/GPU sharing too.
> (3) Optimiz[e] your implementation to be as fast as possible; also related to
> (1) We have SIMD oscillator implementations too. We really need to have a
> design pattern that clearly separates all the implementations as cleanly as
> possible. (2) Audit your implementation in a separate pass, again — this should
> come *after* (3).

**Explicit ordering:** (1) unify → (3) optimize the band-limited file → (2) audit
the band-limited file. This plan honours that order.

---

## 1. The current landscape (inventory, measured this pass)

The float-domain oscillator math is **already centralized** in two headers; the
mess is not duplicated *math*, it's duplicated *loop structure*, an overloaded
*"dispatch"* vocabulary, and one dead execution path. Honest map:

```text
LAYER / FILE                                    ROLE                       SHARES VIA
─────────────────────────────────────────────────────────────────────────────────
core/spectral_osc_formulas.h                    8 waveform shapes,         #include
  (VERSION 5)                                   normalize_phase,           (CPU, CUDA,
                                                 fast_sin, fade envelope    bandlimited)
core/spectral_segment_math.h                    segment evolution          #include
  (VERSION 1)                                   (alpha/beta/d_amp/          (CPU, CUDA)
                                                 phase_at/amp_at)
─────────────────────────────────────────────────────────────────────────────────
core/oscillator.c                               CPU scalar driver:         calls L0
  synth_segment_scalar                          3 fade regions, per-       headers
  timbre_synth_segment (entry)                  sample loop body
  timbre_table[TIMBRE_COUNT]                    fn-ptr table over L0
  oscillator_metal_source (MSL string)          Metal MIRROR of L0          manual,
                                                 waveforms                  _Static_assert
                                                                            (VERSION 5)
core/oscillator.h                               oscillator_cuda() switch    calls L0
  (__CUDACC__)                                  CUDA MIRROR = thin
─────────────────────────────────────────────────────────────────────────────────
core/oscillator_dispatch.h                      per-timbre exec selector    DEAD (see §2)
  OscDispatchWord / OSC_GET_MODE                (scalar/SIMD/native)
core/port/host/oscillator_simd.c                host SIMDe oscillator       DEAD
  osc_simd_segment_*                            (cubic, SPECTRAL_PRECISE)
core/port/embedded/oscillator_simd.c            CMSIS-DSP oscillator        DEAD
  osc_simd_segment_*                            (quadratic-only)
─────────────────────────────────────────────────────────────────────────────────
core/spectral_osc_bandlimited.{h,c}             quality modes: polyblep/    calls L0;
  osc_bandlimited_synth_segment                 additive/oversample;        re-rolls its
  g_osc_quality                                 own per-sample loop ×3      own loop body
─────────────────────────────────────────────────────────────────────────────────
synth/backends/gpu/cuda/spectral_synth_cuda.cu  GPU tile kernel             #include L0 +
  synthesize_tile_kernel                        inner loop body             oscillator_cuda
synth/backends/gpu/metal/spectral_synth_metal.m GPU tile kernel (MSL):      manual MIRROR
  metalKernelCode (segment helpers MSL)         segment-math MIRROR +       of segment_math
  synthesize_tile_parallel                      inner loop body             _Static_assert
                                                                            (VERSION 1)
─────────────────────────────────────────────────────────────────────────────────
synth/backends/arm/spectral_synth_arm32.c       Q15 fixed-point, SINE-only  SEPARATE
  synth_segment_m7 / spectral_lut_sin           LUT. Different numeric      numeric domain
synth/backends/sim/spectral_synth_simulation.c  Q15 simulation              (cannot share
                                                                            float formulas)
```

### Three orthogonal axes are tangled under two names ("dispatch")

```text
AXIS                 VALUES                                   SELECTOR              STATE
─────────────────────────────────────────────────────────────────────────────────────
Backend / domain     CPU-float · Metal · CUDA · ARM-Q15      SynthBackend          live
Exec strategy (CPU)  scalar · host-SIMD · embedded-SIMD      OscDispatchWord       DEAD
                                                             ("dispatch")
Quality (anti-alias) naive · polyblep · additive · oversample SpectralOscQuality   live
                                                             ("quality")
```

"dispatch" means **backend** in `SynthBackend`/pipeline talk, but **per-timbre
SIMD selection** in `oscillator.c`. That collision is the single biggest source
of "it's getting confusing."

---

## 2. The real problems (separated from the inherent ones)

**P1 — Dormant per-timbre SIMD path (the "why isn't it being used?" question).**
`g_osc_dispatch` is statically `OSC_DISPATCH_ALL_SCALAR`; `osc_set_dispatch()`
has **zero callers** (verified — only doc-comment mentions). So `OSC_GET_MODE`
always returns `CPU_SCALAR`, the `OSC_MODE_CPU_SIMD` branch in
`timbre_synth_segment` is never taken, and the entire `osc_simd_segment_*` host +
embedded implementation (≈620 lines) is compiled but **unreachable** in every
current build.

It is *not* abandoned code — it is **complete, hardened, and deliberately written
to be op-for-op bit-identical to scalar** (passes 152, 157–193; cubic phase added
PASS198). `simde_fast_sin_ps` uses per-lane `sinf`; the phase wrap is "bit-for-bit"
the scalar formula; each `wave_*_4` "matches the scalar oscillator lane-for-lane"
(triangle even reorders ops to kill a 1-ULP seam). The **only** thing missing is
the *last mile*: nothing ever throws the switch. The switch was built but never
flipped.

**Why it was never flipped — measured this pass (golden gate):** the host build
compiles with `-ffast-math -ffp-contract=fast` (`host-config.cmake`). Under that,
the scalar Horner `phase0 + j*(…)` contracts to a hardware FMA while the SIMD
intrinsics (`add_ps`∘`mul_ps`) emit separate NEON mul+add. So despite identical
*source* arithmetic, the codegen differs by one rounding step. Scratch experiment
(`OSC_DISPATCH_ALL_SIMD`, desktop, CPU, 1 thread, saw on `sin_440hz.wav`):

```text
samples 88200   differing 37.3%   max|Δ| 1.192e-7 = EXACTLY 1.00 ULP@1.0
                                  rms|Δ| 2.34e-8  (sub-ULP on average)
```

So SIMD is **1-ULP-equivalent** to scalar — musically/numerically identical, but
**not byte-identical**, so turning it on **changes the golden bytes**. That is
almost certainly why it was left dormant: enabling it is golden-affecting and
needs a fresh *signed* golden, which the prior passes could not self-authorise.

**P2 — Per-sample loop body copied 4×+ (accidental).** The shape
`phase_at → amp_at → fade → osc → accumulate` is hand-written in:
`synth_segment_scalar` (3 fade regions), the CUDA kernel, the Metal kernel, and
*each* band-limited mode. A formula tweak means editing it in N places.

**P3 — Overloaded "dispatch" vocabulary (accidental).** See §1. No single doc
states the axes, so the layering is implicit and must be reverse-engineered from
~10 files.

**P4 — Metal string mirror (INHERENT, not a bug).** MSL is compiled from a
runtime string, so Metal *cannot* `#include` the C contract. It re-expresses L0
(`oscillator_metal_source`) and segment-math (`metalKernelCode`) as strings,
guarded by `_Static_assert(VERSION ==…)`. This duplication is structural; it can
only be *managed* (version guard, or codegen), not deleted.

**P5 — Q15 is a different numeric domain (INHERENT).** ARM/sim is fixed-point,
sine-LUT-only. It legitimately cannot share float formulas. The clean answer is
to *document the separation*, not force a merge.

---

## 3. Proposed design pattern — a 4-layer oscillator architecture

One documented contract, four layers, three named axes. Each backend is a thin
driver; math lives in exactly one place per numeric domain.

```text
L0  MATH CONTRACT   (pure, header-only, backend-agnostic)
    spectral_osc_formulas.h   — waveform shapes, phase, fast_sin, fade
    spectral_segment_math.h   — segment evolution
    → single source of truth for the FLOAT domain.
    → Metal carries a "projection" of L0 as MSL strings, version-locked.

L1  PER-SAMPLE KERNEL   (one inline, the thing every loop calls)
    spectral_osc_eval(seg-params, j, timbre, width) -> amp*fade*osc(phase)
    → CPU scalar + CUDA share by #include; Metal mirrors this ONE function
      instead of a scattered loop. Kills P2.

L2  EXECUTION STRATEGY   (how samples are iterated; thin drivers over L1)
    scalar · host-SIMD · embedded-SIMD · GPU-tile
    → WIRE the dormant SIMD selector so CPU actually uses SIMD (maintainer:
      "SIMD would be approached for CPU"). Rename the per-timbre selector so it
      stops colliding with "backend". Kills P1/P3.
      GOLDEN GATE: enabling SIMD shifts output by ≤1 ULP (see §2 P1), so it needs
      a NEW signed golden. Until that golden is signed, SIMD ships OFF by default
      (scalar stays the golden path) and is reachable via an explicit opt-in.

L3  QUALITY   (anti-alias, orthogonal opt-in, CPU-float only)
    naive · polyblep · additive · oversample   (spectral_osc_bandlimited)
    → sits ABOVE L2, unchanged in spirit; its modes are re-expressed over L1
      so they stop re-rolling the loop body.

L4  BACKEND / NUMERIC DOMAIN
    float: CPU · Metal · CUDA  (share L0–L3)
    fixed: ARM-Q15 · sim       (separate domain — documented, not merged)
```

The win: a newcomer reads **one** architecture block and knows that math is L0,
the per-sample contribution is L1, "which loop" is L2, "how musical" is L3, and
"which chip" is L4 — and that Q15 is a deliberately separate island.

---

## 4. Phased work (golden-safe, one change per pass, in the user's order)

Every phase keeps the **default build bit-identical** (the scalar naive path is
the golden path and is not reshaped in a way that changes float ops; L1
extraction is the *same* operations inlined).

### Phase U1 — UNIFY (request #1)   [maintainer chose: doc + dead-code first]

```text
U1a  Architecture doc + header banner. Write the L0–L4 / 3-axis model into one
     place (this doc + a banner in oscillator.h). NO code change.            [doc]

U1b  WIRE the dormant SIMD path as the DEFAULT (Decision A = wire, NOT delete;
     superseded sub-decision: maintainer made SIMD the *default*, not opt-in).
     DONE — PASS200. g_osc_dispatch default flipped to OSC_DISPATCH_ALL_SIMD in
     oscillator.c; pipeline calls osc_set_dispatch() from opts; CLI flags
     --scalar (force reference) and -S/--simd (affirm default) mirror the
     -q/quality plumbing. Verified: 5 builds + 4 ctest green; 1.83× faster on
     saw; default == --simd byte-identical, --scalar diverges by max 2.4e-7 abs
     (-132 dBFS) / 1.1e-8 RMS (-159 dBFS) on the full mix — sub-quantization.
     DONE (follow-ups): the SIMD-vs-scalar parity ctest shipped (osc_parity:
     per-sample drift + aggregate RMS-dBFS budgets through the real dispatch).
     The sine pathology it exposed — SIMD sine only ~1.1× because the kernel
     spilled to 4×sinf — is resolved in PASS201: a degree-9 quadrant-folded
     minimax sine is now the default (SPECTRAL_ENABLE_APPROX_TRIG default 0→1),
     shared bit-for-bit by scalar + SIMD, lifting sine to 1.88× end-to-end. See
     PATCH_NOTES_PASS201. The per-timbre "dispatch" rename still remains for U1c.
                                                          [biggest readability +
                                                           unblocks the perf win]
     → STOP and re-evaluate L1 with the maintainer (their chosen scope boundary).

U1c  ✓ DONE (PASS202). Extracted L1 spectral_osc_eval(); host single-sample API +
     CUDA kernel route through it; hot host loop uses a byte-identical hoisted
     specialization (osc_waveform_fn) of the same X-macro map. Bit-identical.
                                                                      [killed P2]

U1d  ✓ DONE (PASS203). CODEGEN the Metal MSL from the C contract (Decision B).
     tools/spectral_tools/generators/metal_osc.py emits core/spectral_osc_metal_
     generated.h from spectral_osc_formulas.h / spectral_segment_math.h /
     oscillator.h; a build-time `verify_metal_osc` makes drift a build failure,
     replacing the manual mirror + _Static_assert version locks. 16/16 Metal
     outputs byte-identical; FMA-contraction parity lesson recorded in the patch
     notes.                                                           [killed P4 drift]
```

### Phase U3 — OPTIMIZE the band-limited file (request #3, after U1)  ✓ DONE (PASS204)

CPU-float quality path only; opt-in; **not** a golden contract. Measure first.

```text
- oversample mode: ✓ FIR built once per thread (was 130 transcendentals/segment);
  ✓ malloc-per-call → reusable thread-local scratch; ✓ FIR symmetry fold + a
  branch-free interior / clamped-edge decimation split.  Result: saw 4.73x,
  quant 2.70x. Fold delta max 2.98e-7 / RMS -157 dBFS (~1 ULP, inaudible).
- additive: profiled (24x naive) — already O(N) via Chebyshev; left untouched
  (byte-identical).
- polyblep: profiled (5.4x naive) — already branch-light per-sample; left
  untouched (byte-identical).
- Default NAIVE byte-identical; 5 builds + ctest 5/5 green; MT-safe (thread-local).
  See PATCH_NOTES_PASS204.
```

### Phase U2 — AUDIT the band-limited file (request #2, after U3)  ✓ DONE (PASS205)

Separate adversarial-correctness pass over the *optimized* file (NaN/Inf,
boundary phase, fallback coverage matrix, malloc-fail, decimation alignment).
Result: one finite-preserving NaN hardening in osc_bl_norm_freq (a NaN chirp
slope slipped the [1e-6,0.5] clamp → poisoned floorf(0.5/dt) in additive); the
thread-local-scratch retention tradeoff documented; decimation index/alignment,
alloc-failure, fallback matrix, overflow, and division paths all confirmed sound.
Finite-path output byte-identical after the fix; 5 builds + ctest 5/5.
See PATCH_NOTES_PASS205.

---

## 5. Decisions — RESOLVED by maintainer (2026-06-02)

```text
A. Dormant SIMD path  → WIRE it, do NOT delete. ("Don't delete it; why isn't it
   being used??? SIMD would be approached for CPU.")  Answered in §2 P1: the
   switch was never thrown, and flipping it is golden-affecting (≤1 ULP under
   -ffast-math). → SUPERSEDED: maintainer made SIMD the DEFAULT, not opt-in
   ("So long as the SIMD implementation is faster — why wouldn't it be the
   default?") and authorized the re-baseline as the golden authority. Shipped
   PASS200; --scalar restores the reference.

B. Metal mirror       → CODEGEN the MSL from the C contract. → U1d.

D. First-step scope   → Doc + dead-code (SIMD-wiring) FIRST, then re-evaluate L1
   extraction (U1c) with the maintainer. → execute U1a + U1b, then STOP.

C. Pass accounting (still open, minor): slot U1a/U1b as the next monotonic
   passes vs a labelled side-track — confirm next pass number at commit time.
```

### Golden sign-off — RESOLVED

Enabling SIMD as the **default** CPU path changes the golden by a
sub-quantization amount (§2 P1; measured max 2.4e-7 abs / -132 dBFS on the full
mix). The maintainer — the golden authority — authorized this re-baseline
directly, so the SIMD golden is signed. The scalar reference remains one flag
away (`--scalar`) and the parity ctest (osc_parity) now pins the equivalence.

A SECOND re-baseline (PASS201) was authorized the same way: the default sine is
now a degree-9 minimax poly (was libm sinf), clearing the EXACT_TRIG guarantee
bit by default (SPECTRAL_ACTIVE_GUARANTEES 0x3e→0x3c). It is ~1.4 ULP /
-155 dBFS RMS vs libm and the exact path is one flag away
(`SPECTRAL_ENABLE_APPROX_TRIG=0`). See [[faster-path-should-default]].

**Execution order:** U1a → U1b ✓(PASS200) → parity ctest ✓ → minimax sine
✓(PASS201) → (STOP, re-evaluate L1) → U1c ✓(PASS202) → U1d ✓(PASS203) → U3
✓(PASS204) → U2 ✓(PASS205) → **U-TRACK COMPLETE; next effort = Q-type domain
(QTYPE_DOMAIN_PLAN.md)**. From U1c onward the default build is byte-identical at every step
(L1 extraction is the same float ops inlined; U1d's Metal output is 16/16 byte-
identical); the two intentional re-baselines were U1b's SIMD default and
PASS201's minimax-sine default.
