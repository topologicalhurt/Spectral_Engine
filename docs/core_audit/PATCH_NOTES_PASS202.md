# Patch notes — Pass 202: U1c — single L1 oscillator kernel (spectral_osc_eval)

## Scope

Oscillator-unification step **U1c** (`docs/core_audit/OSCILLATOR_UNIFICATION_PLAN.md`):
extract the per-sample timbre→waveform dispatch into ONE place so the host
single-sample API, the CUDA tile kernel, and (next, U1d) Metal stop carrying
their own copies. Resolves **P2** (per-sample dispatch duplicated across
backends). Default build stays byte-identical; this is a structural change, not
a behavior change.

## The single source of truth

`core/oscillator.h` now defines an X-macro list — the canonical timbre→L0-waveform
map — and expands it in the two places that previously diverged:

```c
#define SPECTRAL_OSC_TIMBRE_LIST(X)             \
    X(TIMBRE_SINE,      spectral_osc_sine)      \
    X(TIMBRE_SAW,       spectral_osc_saw)       \
    ... X(TIMBRE_PWM,   spectral_osc_pwm)
```

1. **`spectral_osc_eval(phase, timbre, width)`** (oscillator.h) — the L1 kernel:
   `spectral_normalize_phase(phase)` then a switch generated from the list.
   Compiles as `static inline` in C and `__device__ __forceinline__` under nvcc
   (via `OSC_FORMULA_FUNC`). Used by the single-sample `timbre_oscillator()` and
   by `oscillator_cuda()`.
2. **`osc_waveform_fn(timbre)`** (oscillator.c) — a host pointer selector
   generated from the SAME list. Resolves the timbre to its fixed L0 waveform
   ONCE per segment; the hot `synth_segment_scalar()` loop then calls that one
   function pointer.

One list, two expansions → the host and device dispatch can no longer drift.

## What changed

- **`core/oscillator.h`**: added `SPECTRAL_OSC_TIMBRE_LIST`; added
  `spectral_osc_eval()` (list-driven switch); `oscillator_cuda()` reduced to a
  one-line wrapper `return spectral_osc_eval(phase, timbre, 0.0f)` (was a bespoke
  6-case switch); `spectral_osc_formulas.h` now included unconditionally (it was
  CUDA-only before) so the L1 kernel is available to every includer.
- **`core/oscillator.c`**: removed the standalone `timbre_table[]`; added the
  list-driven `osc_waveform_fn()`; `timbre_oscillator()` now routes through
  `spectral_osc_eval()`; `timbre_synth_segment()` passes `osc_waveform_fn(timbre)`
  to the scalar loop. `synth_segment_scalar()` keeps its hoisted-`osc_fn` form.
- **CUDA `.cu`**: unchanged source — it already calls `oscillator_cuda(p,timbre)`,
  which now delegates to the shared kernel.

## Why the host segment loop does NOT call spectral_osc_eval per sample

The first cut routed `synth_segment_scalar()` through `spectral_osc_eval()` per
sample. Output was byte-identical, but the **scalar / asin** synth stage regressed
~1.7× (shakespeare, internal Synth ms): the loop-invariant 8-way switch was not
hoisted/unswitched out of the inner loop, and the heavy `quantized`/`pwm` cases
(isfinite + division) bloated the body.

Fix: hoist the dispatch out of the loop. `osc_waveform_fn(timbre)` selects the
waveform pointer once per segment; the loop calls that single function (inlinable,
no per-sample branch). This is the exact code shape Pass 200/201 measured. Both
forms are byte-identical; `spectral_osc_eval` is the portable single source,
`osc_waveform_fn` is its host-hoisted specialization. The decision is documented
inline at the kernel.

This only affects timbres that run the scalar loop: the `--scalar` reference path,
and **asin** in the default build (asin has no SIMD kernel). The default SIMD path
for every other timbre never touches `synth_segment_scalar`.

## Behavior delta

- **CPU default + `--scalar`**: byte-identical (verified, see below).
- **CUDA** (compile-unverifiable here — no nvcc on the dev Mac; source-reviewed):
  `oscillator_cuda` now handles all 8 timbres via the shared kernel. The only
  change vs the old bespoke switch is that **quantized/pwm** are now evaluated
  (with width 0 — quantized→0, pwm→1, since the CUDA tile kernel passes no width)
  instead of folding to sine. This is a faithful unification of the dispatch, not
  a tuned change; the CUDA tile path is not in any production target.

## Verification (measured, not asserted)

```text
- 5 production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 5/5 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity).

- Byte-identity (SHA-256 of output/out_c.wav vs the pre-U1c working-tree binary),
  all 8 timbres x {sin_440hz, shakespeare} x {SIMD default, --scalar} = 32 cases:
    SIMD default : IDENTICAL (16/16)
    --scalar     : IDENTICAL (16/16)

- Perf (shakespeare, internal Synth ms, best of 3, single thread):
    SAW  SIMD   535.7 ms   (default path, unchanged)
    SAW  scalar 1389.4 ms  (was 2387 ms in the per-sample-eval first cut — regression removed)
    ASIN SIMD   3736 ms / ASIN scalar 3807 ms  (asin has no SIMD kernel → both run
                the scalar loop; asinf()-bound, equal as expected, no regression)
```

## Status

Single L1 `spectral_osc_eval` kernel extracted and shared by the host single-sample
API and the CUDA device path; the hot host segment loop uses a byte-identical
hoisted specialization of the same X-macro map. Default build byte-identical, 5
builds + ctest green, scalar regression from the first cut removed. Metal still
carries its hand-written MSL mirror — that is U1d (codegen from the C contract),
which the `_Static_assert(SPECTRAL_OSC_FORMULAS_VERSION == 6)` still guards.

## Proposed next pass

**U1d** — codegen the Metal MSL oscillator + segment-math from
`spectral_osc_formulas.h` / `spectral_segment_math.h` at build time, replacing the
manual mirror string in oscillator.c and its version `_Static_assert`, so Metal
can no longer drift from the C contract. Then **U3** (optimize the band-limited
file, measure-first) and **U2** (adversarial audit of the optimized file).
