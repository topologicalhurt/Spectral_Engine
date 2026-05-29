# Patch notes — Pass 154: spectral_out.c kernel port-layer split (Phase E)

## Problem

`core/spectral_out.c` interleaved host and embedded implementations of three
kernels via in-body profile `#if`s — the embedded/host split Phase E targets:

```text
- spectral_normalize_float   : #if (!EMBEDDED || EMBEDDED_SIM) && !USE_CMSIS
                                  SIMDe (spectral_vmaxmgv / spectral_vsmul)
                                #else scalar (fabsf loop)
- spectral_normalize_q15     : #if SPECTRAL_USE_CMSIS  arm_absmax_q15 / arm_shift_q15
                                #else scalar
- spectral_mono_to_stereo_q15: #if SPECTRAL_ARM_M7 && __ARM_FEATURE_DSP  2x unroll
                                #else scalar
```

`SPECTRAL_ARM_M7` is exactly the kind of device-class branch the Phase E closure
criteria forbids inside core algorithm bodies. The rest of the file
(`spectral_mono_to_stereo_float`, the libsndfile writers, the PEAK-timestamp
scrubber) is device-agnostic, gated only by the `SPECTRAL_HAS_FILE_IO`
*capability* macro (filesystem presence) — which Phase E keeps.

## Change

Extracted the three profile-divergent kernels into build-selected port files
behind the existing `spectral_io.h` interface (their declarations were already
unconditional there — the interface was device-agnostic; only the bodies
diverged):

```text
core/port/host/spectral_out_kernels.c      NEW  normalize_float = SIMDe,
                                                normalize_q15   = scalar,
                                                mono_to_stereo_q15 = scalar.
                                                Unguarded; host/sim build context.
core/port/embedded/spectral_out_kernels.c  NEW  normalize_float = scalar,
                                                normalize_q15   = CMSIS|scalar,
                                                mono_to_stereo_q15 = M7-DSP|scalar.
                                                Retains SPECTRAL_USE_CMSIS /
                                                SPECTRAL_ARM_M7 capability gates
                                                (allowed inside the embedded port).
core/spectral_out.c                         keeps only device-agnostic code:
                                                mono_to_stereo_float + file I/O
                                                (SPECTRAL_HAS_FILE_IO capability).
                                                No SPECTRAL_EMBEDDED / ARM_M7 / CMSIS
                                                branches remain; dropped the now-unused
                                                spectral_q15.h / spectral_vector_ops.h
                                                / <math.h> / arm_math.h includes.
```

### Each profile file holds exactly the kernel variant that profile uses

The green targets are all host or *simulation* builds (the `embedded_arm*`
targets compile with `SPECTRAL_EMBEDDED_SIMULATION`, i.e.
`SPECTRAL_IS_EMBEDDED_SIM`). For all of them the old `#if`s selected:
SIMDe `normalize_float` (guard true via host or EMBEDDED_SIM) + scalar
`normalize_q15` (`USE_CMSIS=0`) + scalar `mono_to_stereo_q15` (`ARM_M7=0`) — which
is precisely the host kernels file. So the host file is wired into the host CLI
stack + desktop (same placement as the pass-152/153 host port files), and the
build now selects it for the same targets the guards previously enabled.

The embedded kernels file is the bare-metal Cortex-M variant; no current target
compiles it (no green target builds `spectral_out.c` on real hardware — the daisy
engine source set excludes it), exactly like the pass-152 embedded oscillator
file. It is retained as the build-selectable embedded port, not dead code.

Build wiring: `SPECTRAL_SOURCES_CORE_OUT_KERNELS_HOST` added to
`SPECTRAL_SOURCES_HOST_CLI_STACK` + `SPECTRAL_SOURCES_TARGET_DESKTOP`; both new
files added to `SPECTRAL_LOG_CHECK_FILES` (lint coverage).

## Verification

Behavior-preserving: each kernel body is byte-identical to the branch it came
from; the build selects the same variant per target that the `#if` did.

```text
- six green targets build: desktop, simulate, embedded_arm, embedded_arm_float,
  simulate_daisy, arm_core_test.
- ctest arm32_process_correctness: passed (100%, 0 failed).
- sim oracle check: all 6 cases match goldens.
- desktop CPU render (sine) on sine_bin.wav: peak 0.9500, rms 0.4993 — identical
  to the pre-refactor render, confirming the relocated SIMDe vmaxmgv/vsmul
  normalization path is unchanged.
```

## Scope (Phase E increment)

One module split. `core/spectral_out.c` now carries no profile/device branching
(only the `SPECTRAL_HAS_FILE_IO` capability gate). Remaining Phase E core file
with in-body profile branches: `spectral_synth_internal.c`, plus a few header
notes — addressed in the next passes.
