# Patch notes — Pass 155: spectral_synth_internal.c GPU-tile port-layer split (Phase E)

## Problem

`core/spectral_synth_internal.c` carried the GPU tile-preprocessing kernel
(`gpu_tile_preprocess` + its two static helpers and the `SpectralGpuTileSpan`
typedef) inside a `#if !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE` block,
and the always-compiled wrapper `gpu_tile_preprocess_cached` mirrored that with
its own in-body profile `#if` (real call vs `#else return
SPECTRAL_ERR_BACKEND_UNAVAIL`):

```text
- gpu_tile_preprocess (+ spectral_gpu_segment_tile_span,
  gpu_tile_preprocess_scratch_free, SpectralGpuTileSpan)
      : #if !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE  full impl
        #else                                                 (absent)
- gpu_tile_preprocess_cached
      : #if !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE  call real impl
        #else                                                 BACKEND_UNAVAIL
```

`SPECTRAL_EMBEDDED` / `SPECTRAL_RESTRICTED_MODE` are exactly the profile/device
branches the Phase E closure criteria forbids inside core algorithm bodies. The
rest of the file (param scalars, preflight, segment loop params, the GPU tile
cache and segment cache, dispatch-plan init/free) is device-agnostic.

## Change

Extracted the profile-divergent GPU tile kernel into build-selected port files
behind the existing `spectral_synth_internal.h` interface (the
`gpu_tile_preprocess` declaration there was already unconditional — the
interface was device-agnostic; only the body diverged):

```text
core/port/host/spectral_gpu_tile.c       NEW  real segment->tile mapping for
                                              GPU (Metal/CUDA) dispatch — the
                                              former #if-true body verbatim
                                              (OpenMP two-pass count+fill).
                                              Unguarded; host/desktop context.
core/port/embedded/spectral_gpu_tile.c   NEW  stub returning
                                              SPECTRAL_ERR_BACKEND_UNAVAIL
                                              (no GPU backend on this profile).
core/spectral_synth_internal.c           keeps only device-agnostic code: the
                                              wrapper now calls
                                              gpu_tile_preprocess unconditionally
                                              (the #if/#else is gone), the tile
                                              cache, seg cache and dispatch plan.
                                              No SPECTRAL_EMBEDDED / ARM_M7 /
                                              RESTRICTED branches remain; dropped
                                              the now-unused <math.h> include
                                              (floor/ceil left with the kernel;
                                              <float.h> kept for FLT_MAX).
```

### Build selection differs from the prior host/sim passes

Unlike passes 152–154 (host-vs-sim kernels that rode with *every* host/sim
target), this split is a host-only-*capability* boundary. The former guard
`!SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE` has **no** `|| EMBEDDED_SIM`
escape, so it evaluated true on exactly one target — desktop. Every
simulation/embedded target (`simulate`, `simulate_daisy`, `embedded_arm`,
`embedded_arm_float`, `embedded_arm_restricted`) defines `SPECTRAL_EMBEDDED`
(via `SPECTRAL_SIMULATION_DEFINES`) and took the `#else` BACKEND_UNAVAIL arm.

So the build wires:

```text
SPECTRAL_SOURCES_CORE_GPU_TILE_HOST     -> SPECTRAL_SOURCES_TARGET_DESKTOP only
                                           (like SPECTRAL_SOURCES_CORE_DESKTOP).
SPECTRAL_SOURCES_CORE_GPU_TILE_EMBEDDED -> SPECTRAL_SOURCES_HOST_CLI_STACK only
                                           (feeds simulate / simulate_daisy /
                                           embedded_arm* / restricted — every
                                           non-desktop target that compiles
                                           spectral_synth_internal.c).
```

No target compiles both files; `arm_core_test` compiles neither
`spectral_synth_internal.c` nor a gpu_tile file, so it is unaffected. Both new
files added to `SPECTRAL_LOG_CHECK_FILES` (lint coverage).

## Verification

Behavior-preserving: the real kernel body is byte-identical to the
former `#if`-true body; on desktop the wrapper already called the real
`gpu_tile_preprocess` (the `#if` was true), and on every other target the stub
returns the same `SPECTRAL_ERR_BACKEND_UNAVAIL` the old `#else` did.

```text
- six green targets build: desktop, simulate, embedded_arm, embedded_arm_float,
  simulate_daisy, arm_core_test (only pre-existing -Wunused-function on
  synth_zero_output_if_valid, present at HEAD).
- ctest arm32_process_correctness: passed (100%, 0 failed).
- sim oracle check: all 6 cases match goldens.
- desktop CPU render (sine_bin.wav t0 s1 p0 n1024 h256 d-70, backend=cpu):
  byte-for-byte identical to the HEAD-built render (cmp clean) — the CPU synth
  path never calls gpu_tile_preprocess, and the relocation changed nothing on
  the GPU path either.
```

## Scope (Phase E increment)

One module split. `core/spectral_synth_internal.c` now carries no
profile/device branching. Remaining Phase E items: a few header notes
(`spectral_wavetable.h`, `spectral_macros.h`, `spectral_vector_ops.h`,
`spectral_resource_fs.h`) and the device-specific memory-section macros in
`spectral_config.h` — addressed in the closure pass.
