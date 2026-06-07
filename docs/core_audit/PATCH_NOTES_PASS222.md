# Patch notes — Pass 222: oscillator backend matrix + Q15 contract versioning (Oscillator-Backend-Contract Phase 0 + 1a/1d)

## Problem

Maintainer directive (2026-06-06): implement the oscillator across backends
(Scalar / SIMDe / CMSIS / GPU) under one anti-drift design contract per compute
domain, document **which paths prohibit Q15** and **what oscillator paths an
embedded build can take**, and **unify all Q15 first**. Plan captured in
`docs/core_audit/OSCILLATOR_BACKEND_CONTRACT_PLAN.md`. This pass executes the
parts of the maintainer-chosen "Q15 first" order that are deliverable
autonomously and byte-identically: the documentation/program-design matrix
(Phase 0), the Q15-divergence characterization (Phase 1a), and the contract
version stamp (Phase 1d). It also records an evidence-based **decline** of the
Phase-1b physical merge, surfaced to the maintainer.

## Change

```text
1. Backend x domain x target matrix in PROGRAM DESIGN  (Phase 0)
   core/oscillator_dispatch.h  (new authoritative header-comment block)
   Codifies, at the backend-selection locus, the full matrix:
     Scalar  : float + Q15, host + embedded            (universal)
     SIMDe   : float + Q15 (pack8 8xQ15), HOST ONLY
     CMSIS   : float today; Q15 = Phase 2, EMBEDDED ONLY
     GPU     : float-only* (Metal/CUDA)                 (*Q15 double-pack = Ph.5)
     vDSP    : NOT an oscillator -- FFT/window/math-accel only (Ph.4)
   Plus the two prose asks: WHICH PATHS PROHIBIT Q15 (vDSP=n/a, GPU=float-only,
   CMSIS-Q15=Phase 2) and WHAT AN EMBEDDED BUILD CAN TAKE (Scalar + CMSIS-float +
   arm32-Q15; SIMDe is host-only and the dispatch swaps it for CMSIS on real
   Cortex-M). Folds in the SIMDe-is-host premise correction. No behavior change.

2. Q15 oscillator contract versioned  (Phase 1d)
   core/spectral_osc_q15.h
     +#define SPECTRAL_OSC_Q15_VERSION 1   (with bump-rule comment; the Q15 twin
      of SPECTRAL_OSC_FORMULAS_VERSION on the float side)
   core/oscillator.c  (scalar-Q15 consumer)
   core/port/host/oscillator_simd.c  (pack8 8xQ15 consumer, inside OSC_SIMD_GENERIC)
     +_Static_assert(SPECTRAL_OSC_Q15_VERSION == 1, ...)
   A silent edit to any Q15 evaluator / boundary helper / LUT builder now fails
   the consumers' build until they are re-validated and the pin is bumped. This
   is the concrete "enforce a pattern so they don't drift" mechanism. Additive,
   compile-time-only -> zero runtime bytes change.
```

## Finding

Phase 1a characterization — the "three Q15 worlds" are **far less divergent**
than the planning framing assumed, and the residual divergence is **deliberate**:

```text
- SINE is already unified at the primitive level. All three worlds evaluate sine
  through the SAME interpolator spectral_lut_sin: the canonical
  spectral_osc_q15_sine (osc_q15.h:72) wraps it; arm32 calls it directly
  (spectral_synth_arm32.c ~587). arm32 is sine-ONLY -- it implements no Q15
  saw/square/triangle/parabola, so there is no non-sine divergence to reconcile.

- The ONLY real divergences are:
  (i)  LUT amplitude scale: embedded/arm32 build the sine LUT via
       spectral_lut_init_sine at SPECTRAL_LUT_AMP_SCALE=32700 (spectral_consts.h:38);
       desktop/canonical build via spectral_osc_q15_init_sine_lut at full-scale
       Q15_MAX=32767. Uniform ~-0.0178 dB gain (20*log10(32700/32767)). This is
       DELIBERATE and already cross-documented at osc_q15.h:47-52 -- full-scale
       gain-matches the Q15 sine to float so a parity test reads pure quantization,
       not gain.
  (ii) Phase representation: arm32 = 32-bit unsigned NCO accumulator (phase>>16 ->
       uq16 index); canonical = signed-Q15 pq, reinterpreted (uint16_t)pq. Different
       conventions feeding the same primitive.
```

Phase 1b is **DECLINED on evidence** (surfaced to maintainer; full rationale in
the plan doc Phase 1):

```text
- Collapsing the two LUT builders into one shared algorithm has NO byte-identical
  extraction site. As a static inline in spectral_lut.h it pollutes a clean
  integer-only header (its inlines never touch sinf/SPECTRAL_TWO_PI) across every
  includer, incl. embedded TUs. As a non-inline in spectral_lut.c it forces the two
  DELIBERATELY zero-engine-link precision harnesses (q15_compute_precision,
  phase_nco_precision -- "header-only inlines plus the test TU, nothing linked from
  the engine", per their cmake comments) to link spectral_lut.o. Both options damage
  a deliberate design property to de-dup ~10 lines of standard, already-cross-
  documented sine-table code. The drift risk is closed instead by 1d (version stamp)
  without touching either builder.
- Routing arm32's sine through the canonical evaluator is NOT byte-identical: it
  changes arm32's phase convention (32-bit NCO -> signed-Q15) and LUT scale
  (32700 -> 32767), altering shipped embedded output (the arm32_process golden).
  Per the north star (don't gratuitously change embedded output) the numeric
  convergence is a MAINTAINER decision, not an autonomous refactor.
- Phase 1c (cross-backend bit-parity gate) is DEFERRED behind that decision: a true
  bit-parity test cannot pass while the 32700/32767 + phase-convention gaps stand
  by design. The bit-exact claim that IS provable today (scalar-Q15 == pack8-SIMDe-Q15)
  is already gated by q15_simd_parity.
```

## Verification

```text
- five standard production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mavx2 / -mno-avx512f
  unused-arg notes on host). The _Static_assert pins compile in every build.
- ctest: 12/12 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity, osc_width_parity, q_domain_contract,
  q15_compute_precision, q15_production_parity, q15_simd_parity, phase_nco_precision,
  full_fused_parity). The full parity set empirically confirms no drift across the
  scalar/SIMD/Q15/arm32 paths.
- default desktop render byte-identical BY CONSTRUCTION: every change this pass is a
  header comment, a #define, or a _Static_assert -- all compile-time-only, zero
  runtime bytes.
- The non-standard embedded_arm_restricted target is independently broken
  (_analyze_audio / _perf_print undefined from spectral_cli_pipeline.c.o in the
  analysis=NO config). Confirmed PRE-EXISTING by stashing this pass's four edits and
  reproducing the identical link error -- unrelated to oscillator work.
```

## Scope (Oscillator-Backend-Contract Phase 0 + 1a/1d)

Documentation/program-design matrix + Q15 contract versioning only. No behavior
change on any path; no embedded output moved. The byte-changing parts of Phase 1
(LUT-scale convergence, arm32->canonical reroute) and the Phase-1c bit-parity gate
are surfaced as a maintainer decision. Remaining phases unstarted: Phase 2
(CMSIS-Q15, hardware-gated), Phase 3 (backend contract hardening + unified parity
matrix), Phase 4 (vDSP math-accel audit, measure-first), Phase 5 (GPU Q15
double-pack, measure-first).
