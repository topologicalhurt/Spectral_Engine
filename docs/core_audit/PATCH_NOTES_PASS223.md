# Patch notes — Pass 223: CMSIS-Q15 oscillator kernel on the canonical contract (Oscillator-Backend-Contract Phase 2)

## Problem

Maintainer directive (2026-06-06): implement the oscillator in CMSIS in
addition to SIMDe/Scalar, propagate the Q15 path into CMSIS, and enforce one
anti-drift design contract so the backends can't diverge —
"every embedded path must be able to support purely q15 or a q15 + float math
that would be designed to maximally saturate the FPU and the integer unit."
Plan: `docs/core_audit/OSCILLATOR_BACKEND_CONTRACT_PLAN.md` Phase 2. Phase 1
(PASS222) versioned the canonical Q15 waveform contract (`spectral_osc_q15.h`,
`SPECTRAL_OSC_Q15_VERSION`) and pinned the two existing Q15 consumers (scalar +
SIMDe pack8). This pass adds the THIRD consumer — the CMSIS embedded Q15
oscillator — on that same contract, and surfaces the part of Phase 2 that is a
maintainer decision rather than an autonomous refactor.

## Change

```text
1. CMSIS-Q15 oscillator kernel  (Phase 2a)
   core/port/embedded/oscillator_simd.c  (inside #if defined(OSC_SIMD_CMSIS))
     +#include "spectral_osc_q15.h"
     +_Static_assert(SPECTRAL_OSC_Q15_VERSION == 1, ...)   (3rd consumer pin)
     +osc_cmsis_q15_eval(pq, timbre, lut)   switch over the canonical v1
        spectral_osc_q15_<timbre> evaluators (sine/saw/square/triangle/parabola)
        -- the embedded sibling of oscillator.c osc_q15_eval and the host
        osc_q15_wave_scalar. NOT a 4th Q15 copy.
     +osc_simd_q15_available(timbre)  -> the 5 canonical Q15 timbres
        (matches the host pack8 set; square included -- the Q15 square is a cheap
        integer select, unlike the float path's scalar-only sign test).
     +osc_simd_q15_segment(dst, lp, timbre, sine_lut):
        float quadratic phase (spectral_segment_phase_at_f32 + phase_to_rads, FPU)
        -> spectral_osc_q15_phase_from_rads (the lone float->Q boundary)
        -> Q15 waveform via the canonical evaluators (INTEGER unit)
        -> arm_q15_to_float widen + arm_mult_f32 amp + arm_add_f32 accumulate
           (CMSIS-DSP, FPU) over 4-sample blocks + scalar tail.
        This literally realises the directive's "Q15 + float math to maximally
        saturate the FPU and the integer unit": integer-unit Q15 waveform eval
        running alongside FPU float phase/fade/widen/amp/accumulate.

2. Backend-uniform contract surface  (Phase 2)
   core/oscillator_dispatch.h
     +#if defined(OSC_SIMD_CMSIS) block declaring osc_simd_q15_{available,segment}
      with the SAME signature as the OSC_SIMD_GENERIC (SIMDe) twin, so the Q15
      dispatch is backend-uniform -- the "one contract per backend" mechanism.
     Matrix comment updated: CMSIS Q15 PLANNED -> present (shares canonical
     contract), with explicit hardware-gated + not-yet-dispatched caveats.
```

## Finding

The plan's Phase-2 framing ("add Q15 to CMSIS, parity by construction, verify on
hardware") was correct but understated two realities this pass nailed down:

```text
- arm_sin_q15 is DECLINED for the sine waveform -- on ANTI-DRIFT grounds, not a
  perf bake-off. arm_sin_q15 is CMSIS-DSP's own Q15 sine table: routing sine
  through it would create a SECOND Q15 sine forked from the canonical
  spectral_lut_sin, i.e. exactly the cross-backend drift this whole initiative
  exists to prevent. So CMSIS-Q15 sine (and every timbre) goes through the
  canonical spectral_osc_q15_* evaluators, making CMSIS-Q15 bit-parity with
  scalar-Q15 and SIMDe-Q15 BY CONSTRUCTION. The host q15_simd_parity /
  q15_compute_precision CTests therefore already pin the CMSIS waveform numerics;
  only the CMSIS-DSP float plumbing + phase pre-pass (identical in structure to
  the already-shipping float CMSIS path) are CMSIS-specific.

- The CMSIS-Q15 kernel has NO LIVE CALLER yet, and wiring one is a MAINTAINER
  decision. oscillator.c's entire Q15 dispatch block is #if !SPECTRAL_EMBEDDED
  (PASS216, deliberate): on embedded, Q15 oscillator synthesis is owned by
  spectral_synth_arm32.c (sine-only NCO), and g_osc_q15_sine_lut + the Q15 segment
  helpers are compiled out. Promoting CMSIS-Q15 into the live embedded dispatch
  therefore (a) reverses that deliberate host-only guard, (b) introduces a SECOND
  embedded Q15 oscillator backend alongside arm32 (which one owns embedded Q15?),
  and (c) changes shipped embedded behavior. Per the north star (don't gratuitously
  move embedded output) and the ultraplan-before-execution rule (maintainer sets
  the order), that promotion is surfaced, not taken autonomously. The kernel is
  written, contract-shared, version-pinned, and declared backend-uniform so the
  eventual wiring is a small, well-scoped change.
```

## Verification

```text
- VERIFICATION IS HARDWARE-GATED, and this pass pinned exactly how far local
  verification reaches. OSC_SIMD_CMSIS is set ONLY by a real Cortex-M cross-build
  (ARM_MATH_CM7, daisy-config.cmake:87); source-manifest.cmake:64-65 confirms
  "every current target is a host (SIMDe) build ... the CMSIS-DSP body is selected
  only by a real Cortex-M cross-build." None of the 5 standard builds compile
  core/port/embedded/oscillator_simd.c at all. Worse, the dev environment has NO
  arm_math.h / libDaisy/DaisySP and DAISY_PATH is unset, so `make daisy`
  FATAL_ERRORs and even a bare arm-none-eabi-gcc -c cannot resolve
  oscillator_dispatch.h's #include "arm_math.h". The existing CMSIS *float*
  oscillator is equally un-buildable locally -- this is not new to Q15; the whole
  CMSIS path is genuinely hardware-gated.

- Strongest local check performed: the OSC_SIMD_CMSIS branch was type-checked with
  `clang -fsyntax-only -std=c11 -Wall -Wextra -DARM_MATH_CM7` against a minimal
  arm_math.h SHIM (float32_t/q15_t + the 7 arm_* signatures the file uses). Exit 0,
  zero diagnostics: the new code is valid C11, all engine types/headers resolve,
  the canonical evaluators + boundary helper resolve, the arm_* calls match their
  signatures, and the _Static_assert version pin holds. The shim models SIGNATURES
  only, NOT CMSIS-DSP semantics -- cycle/throughput (does the FPU/integer dual-issue
  actually pay on M7?) and on-hardware numeric run are deferred to hardware/QEMU,
  exactly like the existing A2/A3 ARM deferral and the float CMSIS path it mirrors.

- five standard production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mavx2/-mno-avx512f
  unused-arg notes on host). The new code is inert in all five (host port selected;
  dispatch-header additions are OSC_SIMD_CMSIS-guarded -> false on host).
- ctest: 12/12 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity, osc_width_parity, q_domain_contract,
  q15_compute_precision, q15_production_parity, q15_simd_parity, phase_nco_precision,
  full_fused_parity).
- default desktop render byte-identical BY CONSTRUCTION: every host-compiled change
  this pass is a comment or an OSC_SIMD_CMSIS-guarded (host-false) declaration; the
  kernel body compiles only on real Cortex-M. Zero host runtime bytes change.
```

## Scope (Oscillator-Backend-Contract Phase 2a)

CMSIS-Q15 oscillator kernel on the canonical contract + backend-uniform dispatch
declaration + matrix doc. No behavior change on any locally-buildable path; no
embedded output moved (the kernel has no live caller). The byte-changing /
behavior-changing part of Phase 2 — promoting CMSIS-Q15 into the live embedded
dispatch (reverses the PASS216 !SPECTRAL_EMBEDDED Q15 guard; decides whether CMSIS
or arm32 owns embedded Q15) — is surfaced as a maintainer decision. Remaining:
Phase 2b on-hardware cycle/throughput (hardware/QEMU), Phase 3 (backend contract
hardening + unified parity matrix), Phase 4 (vDSP math-accel audit, measure-first),
Phase 5 (GPU Q15 double-pack, measure-first).
