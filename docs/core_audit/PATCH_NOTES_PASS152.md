# Patch notes — Pass 152: oscillator_simd.c port-layer split (Phase E)

## Problem

`core/oscillator_simd.c` was a single translation unit that interleaved two
unrelated SIMD backends behind `#ifdef`, plus a profile-agnostic tail:

```text
#if defined(OSC_SIMD_GENERIC)   -> host SIMDe (SSE2 -> NEON/SSE/scalar) impl
#elif defined(OSC_SIMD_CMSIS)   -> embedded CMSIS-DSP (arm_*_f32) impl
... shared tail: osc_set_native_available / osc_native_available
```

This is exactly the embedded/host interleaving Phase E targets. The two bodies
share no code — they are different instruction sets for different execution
profiles — yet lived in one file gated by capability `#ifdef`s. The maintainer's
Phase E goal calls for build-selected per-profile implementation files behind a
shared interface, not `#ifdef`-multiplexed bodies (the FreeRTOS `portable/`,
musl per-arch, CMSIS device-agnostic-core model).

## Change

Split the one file into three, behind the existing `oscillator_dispatch.h`
interface (unchanged):

```text
core/oscillator_dispatch.c          NEW  profile-agnostic shared tail only:
                                         osc_set_native_available / osc_native_available.
                                         Compiled into CORE for every profile (preserves
                                         the prior always-present behavior).

core/port/host/oscillator_simd.c    NEW  the SIMDe GENERIC implementation, verbatim.
                                         Build-selected onto host/desktop targets.
                                         Unguarded: host is its only build context.

core/port/embedded/oscillator_simd.c NEW the CMSIS-DSP implementation, verbatim.
                                         Build-selected onto Cortex-M cross-builds.
                                         Body kept under `#if defined(OSC_SIMD_CMSIS)`
                                         (a capability gate, Phase-E-allowed) so the TU
                                         is inert if ever seen without CMSIS-DSP present.

core/oscillator_simd.c              DELETED (git rm).
```

Profile selection now lives in the build system, not the preprocessor:

```text
source-manifest.cmake:
  + SPECTRAL_CORE_PORT_HOST_DIR / SPECTRAL_CORE_PORT_EMBEDDED_DIR
  + SPECTRAL_SOURCES_CORE_OSC_SIMD_HOST       = port/host/oscillator_simd.c
  + SPECTRAL_SOURCES_CORE_OSC_SIMD_EMBEDDED   = port/embedded/oscillator_simd.c
  CORE list: oscillator_simd.c -> oscillator_dispatch.c
  host/desktop source sets gain SPECTRAL_SOURCES_CORE_OSC_SIMD_HOST.
utilities.cmake:
  + both relocated files added to SPECTRAL_LOG_CHECK_FILES (keep lint coverage).
```

The `SPECTRAL_FADE_SAMPLES_ACTIVE` selection (profile-based, independent of the
SIMD backend) was hoisted to `spectral_config.h` in this pass so both new files
reference one profile-selected constant instead of re-deriving it.

## Verification

Behavior-preserving: each new file holds the original body verbatim; the only
new code is the CMSIS capability gate (which is true exactly when the old
`#elif OSC_SIMD_CMSIS` arm was active) and the build-system selection that
mirrors which body each profile previously compiled.

```text
- six green targets build: desktop, simulate, embedded_arm, embedded_arm_float,
  simulate_daisy, arm_core_test.
- ctest arm32_process_correctness: passed (100%, 0 failed).
- sim oracle check: all 6 cases match goldens.
- desktop CPU render (backend=CPU, timbre=sine) on sine_bin.wav: non-silent,
  peak 0.95 (normalization target), rms 0.50, ~80% nonzero — confirms the
  relocated osc_simd CPU-float path (which the Q15 oracle does not exercise)
  still renders correct audio.
```

The embedded CMSIS file is real, working CMSIS-DSP code that is build-selectable
but not compiled by any current green target (the daisy build excludes the SIMD
file and `ARM_MATH_CM7` is only set on the daisy cross-build). It is retained,
not deleted, because Phase E makes it the per-profile embedded implementation —
this is the "separate arm files" deliverable, not dead code.

## Scope (Phase E increment)

One module split into the `core/port/{host,embedded}/` layout. Remaining Phase E
core files with in-body profile branches: `spectral_vector_ops.c/.h`,
`spectral_out.c`, `spectral_synth_internal.c`, and a few header notes —
addressed in following passes.
