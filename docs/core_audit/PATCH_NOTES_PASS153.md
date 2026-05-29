# Patch notes — Pass 153: spectral_vector_ops.c port-layer relocation (Phase E)

## Problem

`core/spectral_vector_ops.c` carried a whole-file profile guard:

```c
#if (!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && \
    !defined(ARM_MATH_CM4) && !defined(ARM_MATH_CM7)
... entire SIMDe implementation ...
#endif
```

The body is the host SIMDe vector-ops implementation; the guard disabled it on a
real bare-metal Cortex-M (CMSIS) build, where callers fall back to scalar paths.
This is the embedded/host `#ifdef` split Phase E targets: the file is one
profile's implementation, gated in-place rather than build-selected. (The
internal `#ifdef __AVX2__` branches are SIMD-width *capability* gates, which
Phase E explicitly keeps.)

## Change

Relocated the implementation to the port-layer host directory and made profile
selection a build-system concern, mirroring the pass-152 oscillator_simd split:

```text
core/spectral_vector_ops.c  ->  core/port/host/spectral_vector_ops.c
  - whole-file profile guard removed (the file IS the host impl; build-selected).
  - includes (simde sse2 / conditional avx2 / math / stdint) hoisted to the top.
  - __AVX2__ capability gates retained verbatim.
  - file header comment updated to describe the host profile + where a future
    embedded (CMSIS) counterpart would live.
```

`core/spectral_vector_ops.h` is unchanged. Its guard
`(!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && !CMSIS` is the shared
*interface-availability* gate — it makes the declarations absent on bare-metal
CMSIS so callers compile their scalar fallbacks (e.g. `spectral_out.c`
`spectral_normalize_float` guards its `spectral_vmaxmgv`/`spectral_vsmul` calls
with the same condition). That gate stays; it is the interface contract, not an
implementation-body branch.

Build wiring (`source-manifest.cmake`):

```text
- removed core/spectral_vector_ops.c from SPECTRAL_SOURCES_CORE.
- added SPECTRAL_SOURCES_CORE_VECTOR_OPS_HOST = port/host/spectral_vector_ops.c.
- added it to SPECTRAL_SOURCES_HOST_CLI_STACK and SPECTRAL_SOURCES_TARGET_DESKTOP
  (the same two places SPECTRAL_SOURCES_CORE_OSC_SIMD_HOST rides).
utilities.cmake: added it to SPECTRAL_LOG_CHECK_FILES (kept lint coverage; it
left SPECTRAL_SOURCES_CORE).
```

### Why the file rides with every current target (not "host-only")

The former guard was TRUE on *all six green targets*, not just desktop. The
`embedded_arm*` targets are **simulation** builds: they compile with
`SPECTRAL_SIMULATION_DEFINES` (`SPECTRAL_EMBEDDED=1` **and**
`SPECTRAL_EMBEDDED_SIMULATION=1`), so `SPECTRAL_IS_EMBEDDED_SIM` is set and
`(!EMBEDDED || EMBEDDED_SIM)` is true. The guard was false only on a real
bare-metal CMSIS cross-build, which never compiles `SPECTRAL_SOURCES_CORE` at
all. So placing the host file with the host CLI stack + desktop reproduces
exactly the prior set of targets that compiled a non-empty `vector_ops.c`.

(First attempt scoped this to desktop/simulate/simulate_daisy on the mistaken
assumption that `embedded_arm*` were non-sim; the embedded_arm link then failed
on undefined `spectral_vmaxmgv`/`spectral_vsmul` from `spectral_out.c`, which
pinned down that those targets are simulation builds and need the host impl.)

## Verification

Behavior-preserving: the implementation body is byte-identical (only the guard
wrapper and a comment changed); the build now selects it for the same targets
the guard previously enabled.

```text
- six green targets build: desktop, simulate, embedded_arm, embedded_arm_float,
  simulate_daisy, arm_core_test.
- ctest arm32_process_correctness: passed (100%, 0 failed).
- sim oracle check: all 6 cases match goldens.
- desktop CPU render (sine) on sine_bin.wav: peak 0.9500, rms 0.4993,
  nonzero 17640/22050 — identical to the pre-refactor render, confirming the
  relocated vmaxmgv/vsmul normalization path is unchanged.
```

## Scope (Phase E increment)

One module relocated into the `core/port/host/` layout. No embedded counterpart
created (none exists; CMSIS callers scalar-fallback). Remaining Phase E core
files with in-body profile branches: `spectral_out.c`, `spectral_synth_internal.c`,
and a few header notes — addressed in following passes.
