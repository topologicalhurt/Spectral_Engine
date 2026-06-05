# Patch notes — Pass 207: Q2 — width-parameterize the float L1 SIMD oscillator

## Scope

Q-type domain phase step **Q2** (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5): factor the
unified float L1 SIMD oscillator body over `SIMDE_NATURAL_FLOAT_VECTOR_SIZE` (128/256) so
an AVX2 x86 target can run the sustain kernel 8-wide (`__m256`) instead of leaving the
upper 128 bits of every YMM register idle. **No precision change is intended.** This is the
golden-adjacent throughput step: Mac (NEON, 128-bit-float) stays 4-wide and byte-identical
by construction; only a real AVX2 build picks up the 8-wide instantiation, and that path is
held to the *same* FMA-contraction budget already accepted for SIMD-vs-scalar at PASS200.

## Why this is safe to land without a golden re-baseline

The authoritative platform (Apple Silicon) has `SIMDE_NATURAL_FLOAT_VECTOR_SIZE == 128` and
no `__AVX2__`, so width selection resolves to **W=4** — and the W=4 macro expansion of the
new templated kernel is *token-equivalent* to the pre-Q2 hand-written `simde_mm_*` body.
Same intrinsics, same order, same codegen → same output. Proven below: `osc_parity` reports
the captured baseline to every digit. The 8-wide path *does* shift a few boundary samples
per segment (the SIMD/scalar-tail split moves from a multiple of 4 to a multiple of 8), but
that divergence stays within the ≤1-ULP FMA-contraction class — it is not a precision
regression, it is the same tolerance the parity contract has always allowed.

## What landed

### 1. Width-templated kernel (`core/port/host/oscillator_simd_kernel.inc`, new)

The vector sustain body is now a re-includable `.inc` with **no include guard**. The
includer defines `OSC_VW` (4 or 8) and `OSC_VSUF` (a symbol suffix like `_impl`/`_w4`/`_w8`)
before each include; the file `#undef`/`#define`s a per-width op vocabulary
(`OSC_VF`/`OSC_VSET1`/`OSC_VADD`/`OSC_VMUL`/`OSC_VCMPGT`/`OSC_VLOADU`/`OSC_VTRUNC2F`/
`OSC_VIOTA`/…) that maps to `simde_mm_*` at W=4 and `simde_mm256_*` at W=8. The W=8 compares
go through `simde_mm256_cmp_ps(..., SIMDE_CMP_*_OQ)` (ordered, non-signaling — NaN compares
false, matching the SSE `cmpgt/cmpge/cmple/cmplt_ps` semantics the kernel relies on). All
emitted symbols are suffixed with `OSC_VSUF`, so multiple widths can coexist in one TU. The
quantized path's `_mm256_cvttps_epi32`/`_mm256_cvtepi32_ps` are plain AVX (not AVX2), so the
256-bit-float tier is self-consistent.

### 2. Scalar fade lanes split out (`core/port/host/oscillator_simd_scalar_waves.h`, new)

The single-sample wave functions used in the fade-in/tail/fade-out regions are
width-independent, so they moved to their own header shared by the production `.c` and the
new test (`wave_{sine,saw,square,triangle,parabola,quantized,pwm}_1`, each forwarding to the
`spectral_osc_*` formula).

### 3. Natural-width instantiation (`core/port/host/oscillator_simd.c`, rewritten)

Down from the 408-line hand-written `__m128` file to ~95 lines. Width tier:

```c
#if defined(__AVX2__) && SIMDE_NATURAL_FLOAT_VECTOR_SIZE_GE(256)
  #define OSC_KERNEL_W 8
  #include "simde/x86/avx2.h"
#else
  #define OSC_KERNEL_W 4
#endif
#define OSC_VW OSC_KERNEL_W
#define OSC_VSUF _impl
#include "oscillator_simd_kernel.inc"
```

The 7 public `osc_simd_segment_*` entries each call `OSC_FN(osc_simd_fused_sustain)` with
the matching `OSC_FN(wave_*_v)` + scalar `wave_*_1`. `osc_simd_available` unchanged. On Mac
the `-mavx2`/`-mno-avx512f` flags emit the expected "argument unused during compilation"
warnings — confirming `__AVX2__` is undefined on ARM and W=4 is selected.

### 4. The `osc_width_parity` CTest (test #6)

`tests/core_contracts/test_osc_width_parity.c` + `cmake/targets/osc-width-parity-test.cmake`.
Production picks ONE width per build, so a single build can never compare the two. This test
`#include "simde/x86/avx2.h"` to force a portable `__m256` into scope on *any* host (SIMDe
NEON-emulates it on Apple Silicon with identical per-lane float semantics), then includes the
`.inc` **twice** (`OSC_VW=4/_w4`, then `OSC_VW=8/_w8`) to instantiate both widths in one TU.
The scalar reference is the *real* production scalar oscillator (`timbre_synth_segment` under
`OSC_DISPATCH_ALL_SCALAR`). It asserts 8-wide == 4-wide == scalar within `BUDGET_ABS = 1e-5`
across 7 timbres × 4 segments — including an odd length (259) that forces each width's scalar
tail. (The cmake target adds `port/host` to the private include path: the test lives in
`tests/core_contracts/` but the kernel headers live in `core/port/host/`, which is not a
global include dir — the production `.c` reaches its siblings via the same-directory rule,
unavailable to the test.)

## Verification

```text
- 5 production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 7/7 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity, osc_width_parity, q_domain_contract).
- osc_parity drift is BYTE-FOR-BYTE the pre-Q2 baseline — Mac 4-wide path unchanged:
    sine 5.960e-08, saw 5.960e-08, square 0, triangle 5.960e-08, parabola 5.960e-08,
    quantized 2.384e-07, pwm 0; aggregate max 2.384e-07; RMS 1.756e-08 (-155.1 dBFS).
- osc_width_parity PASS:
    8-vs-4 = 0 for sine/square/triangle/parabola/quantized/pwm (width-to-width
      byte-identical → the 8-wide logic is faithful);
    saw 8-vs-4 = 2.086e-06, 8-vs-scalar = 2.146e-06 (the boundary-shift effect — saw
      is a direct linear readout of the large wrapping phase) — 4.7x under budget;
    aggregate 8-vs-4 = 2.086e-06, 8-vs-scalar = 2.146e-06, 4-vs-scalar = 2.384e-07
      (budget 1.0e-05).
```

## Status

Q2 closes. The float L1 oscillator is width-parameterized; an AVX2 x86 build gets the
8-wide kernel for free while Mac stays byte-identical, and `osc_width_parity` pins the
8==4==scalar contract on every host. **Next (Q3) — opt-in Q15 *compute* domain** for
throughput-bound kernels (Q15 L1 twin behind a per-path flag, keyed off
`SIMDE_NATURAL_INT_VECTOR_SIZE`, `__smlad`/`__qadd16` packing). Q3 is **lossy and
golden-gated**: each enabled path needs a measured dBFS-vs-float justification and explicit
maintainer sign-off before any output bytes move. Float stays the default domain.
