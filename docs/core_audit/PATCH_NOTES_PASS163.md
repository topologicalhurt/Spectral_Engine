# Patch notes — Pass 163: CTF sweep increment 3 — port/SIMD/out cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass clears the
**port/SIMD/out cluster** — the build-selected port layer
(`core/port/{host,embedded}/*.c`: oscillator/SIMD waveforms, vector-ops, out
kernels) plus the unconditional core oscillator dispatcher (`core/oscillator.c`).
These files are the Phase-E split that made `core/` device-agnostic: the
algorithm body lives in shared `core/` headers and the profile-divergent kernel
in `core/port/{host,embedded}/`. A defect here is latent because the host CTest
build collapses both profiles onto the desktop tuning (and the embedded M7/CMSIS
branches never compile on host), so a profile-selection slip or an
embedded-only width bug passes every green build and every CTest unnoticed.

## Change

Three defects found and fixed (each behaviour-identical on the host targets, so all
six builds stay byte-identical and ctest stays 4/4 green; what is removed is the
embedded/latent divergence):

```text
1. Scalar oscillator fallback hardcodes the DESKTOP fade length  (profile slip)
   core/oscillator.c  (osc_segment, scalar fallback path)
   The SIMD dispatch path builds its crossfade from the profile-selected constant
     FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);   (oscillator_simd.c)
   but the scalar fallback in the SAME function hardcoded
     FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
   SPECTRAL_FADE_SAMPLES_ACTIVE is 64 (DESKTOP) on host and 32 (EMBEDDED) on a
   Cortex-M build (core/spectral_config.h:498-517), and that header's own comment
   directs synthesis files to use _ACTIVE precisely so they carry NO
   SPECTRAL_EMBEDDED branch (Phase E). So on an embedded build the scalar path
   would fade over 64 samples while every SIMD waveform faded over 32 — a
   per-segment amplitude-envelope divergence between the two code paths that are
   supposed to be bit-equivalent fallbacks of each other. Fix: DESKTOP -> ACTIVE,
   matching the SIMD sibling and the config contract. Byte-identical on host
   (ACTIVE == DESKTOP == 64); removes the embedded scalar/SIMD mismatch.

2. Missing NULL/zero-length guard on an exported SIMD primitive  (latent trap)
   core/port/host/spectral_vector_ops.c  (spectral_magsq_split)
   All eight sibling kernels in this file open with the standard boundary guard
     if (!a || !b || !dst || len == 0) return;
   spectral_magsq_split (declared in spectral_vector_ops.h, external linkage) was
   the lone exception and dereferenced re/im/dst with no guard. It currently has
   no in-tree caller (the linker dead-strips it from every shipped binary, which
   is why this pass is byte-identical), but it is a published primitive — a future
   caller passing NULL or len==0 would fault. Fix: add the same guard the eight
   siblings already carry. Consistent with the pass-161 smlad precedent (harden the
   shipped primitive even with no current caller).

3. 32-bit mask on a size_t pair count in the M7 stereo-widen kernel  (portability)
   core/port/embedded/spectral_out_kernels.c  (spectral_mono_to_stereo_q15)
   Inside #if SPECTRAL_ARM_M7 && defined(__ARM_FEATURE_DSP):
     size_t pairs = num_frames & ~1U;
   ~1U is a 32-bit unsigned (0xFFFFFFFE). On any target where size_t is 64-bit the
   high 32 bits of num_frames are cleared, truncating the pair count. Inert on the
   real Cortex-M (32-bit size_t) and never compiled on host, so it can affect no
   current binary — but it is a correctness landmine for any 64-bit DSP port. Fix:
   ~(size_t)1, so the mask is full-width on every target.
```

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is Phase-E /
Campaign-1 hardened:
- `core/port/host/oscillator_simd.c` — all seven `wave_*_4` SIMDe waveforms
  (saw/square/triangle/parabola/sine/quantized/pwm) reproduce the scalar
  `spectral_osc_*` formulas exactly (verified term-by-term against
  `core/spectral_osc_formulas.h`); no phase-boundary discontinuity, and the fade
  already uses `SPECTRAL_FADE_SAMPLES_ACTIVE`.
- `core/port/{host,embedded}/spectral_out_kernels.c` and `core/spectral_out.c` —
  interleave/deinterleave, dither, q15<->float scaling: every entry guarded for
  NULL/zero-length, no overflow in the index arithmetic.
- `core/port/{host,embedded}/spectral_gpu_tile.c` — host real impl + the
  BACKEND_UNAVAIL stub on every non-desktop target; bounds correct.
- `core/spectral_windows.c`, `core/spectral_lut.c/.h`, `core/spectral_envelope.c/.h`
  — window generation, the `SPECTRAL_OSC_LUT_SIZE + 1` guard-indexed sine LUT
  (already cleared in pass 161), and the envelope ramps are all in-range.
- `core/oscillator_dispatch.h` / `core/spectral_osc_formulas.h` — the 2-bit
  per-timbre mode select and the shared formulas are branch-clean.
- `synth/backends/cpu/spectral_synth_cpu.c` — `SPECTRAL_SYNTH_CPU_FADE_SAMPLES`
  (lines 18-20) IS correctly profile-selected (EMBEDDED/DESKTOP) and used at the
  three fade sites; this is the positive control that pinpointed oscillator.c as
  the lone hardcode (defect 1).

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mno-avx512f
  unused-arg note on host).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- desktop AND simulate binaries are BYTE-IDENTICAL to the pre-change build (cmp
  clean): defect 1 is a no-op on host (ACTIVE == DESKTOP == 64); defect 2's
  function is dead-stripped from every binary (no caller — confirmed via nm:
  magsq_split absent from the desktop link); defect 3 lives in an
  __ARM_FEATURE_DSP branch that host never compiles. The fixes change ONLY the
  embedded/latent behaviour, exactly as intended.
- coverage note: like the prior Phase-C increments, the embedded-divergent effect
  of defects 1 and 3 is not exercised by any host CTest (no Cortex-M target / no
  QEMU here). They are verified by build-clean + byte-identical host output, with
  the behaviour change documented here.
```

## Scope (Phase C increment)

Port/SIMD/out cluster only. Defect 1 restores scalar/SIMD fade-length parity on
embedded builds and honours the Phase-E `_ACTIVE` profile contract; defects 2-3
are latent-trap hardenings (NULL guard on an exported primitive; full-width size_t
mask). No change to any host binary (byte-identical) and no change to the exercised
synthesis path. With this increment the core/analysis/synth CTF sweep has cleared
the fixed-point (161), analysis/peak-track (162) and port/SIMD/out (163) clusters.
Remaining Phase-C surface per ULTRAPLAN: hashing/parsing/path and
allocation/pool/cache. Phase D (compiled CTest harness + LUT golden-vector loop)
follows the sweep.
