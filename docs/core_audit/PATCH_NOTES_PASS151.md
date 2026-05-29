# Patch notes — Pass 151: wavetable.c device-agnostic sample abstraction (Phase E)

## Problem

`spectral_wavetable.c` carried four `#if SPECTRAL_EMBEDDED` / `#else` branches
inside algorithm bodies — the embedded (Q15 fixed-point) vs host (float) split for
the *sample type*, interleaved by `#ifdef`:

```text
- wavetable_runtime_samples_valid()  : embedded returns 1 (skip finite check);
                                        host runs spectral_f32_span_finite().
- spectral_wavetable_load()          : runtime_format = Q15 vs FLOAT.
- spectral_wavetable_save()          : hdr.format    = Q15 vs FLOAT.
- spectral_wavetable_lookup_f/_q()   : Q8 fixed-point lerp vs float lerp.
```

This is the embedded/host interleaving Phase E targets: the difference is a
property of the *sample type* (fixed vs float), not something the wavetable
algorithm should branch on per call site.

## Change

Hoisted the sample-type distinction into the profile-selected sample-type block in
`spectral_config.h` (the single place that already defines `spectral_sample_t` and
its `SAMPLE_*` ops), as device-agnostic abstractions:

```text
SPECTRAL_SAMPLE_IS_FIXED            1 (fixed) / 0 (float)
SPECTRAL_SAMPLE_SPAN_FINITE(s, n)   fixed: trivially-true; float: spectral_f32_span_finite()
spectral_sample_lerp_f (s0,s1,frac) Q8 fixed lerp / float lerp   (static inline)
spectral_sample_lerp_q8(s0,s1,q8)   Q8 fixed lerp / float lerp   (static inline)
```

`spectral_wavetable.c` now calls these by intent — no `#if SPECTRAL_EMBEDDED` in
any of its bodies. The macros/inlines resolve at compile time to the exact code
each branch previously contained, so this is a mechanical hoist, not a behavior
change.

## Verification

Behavior-preserving: each abstraction expands to the original per-profile code.

```text
- six green targets build: desktop, simulate, embedded_arm, embedded_arm_float,
  simulate_daisy, arm_core_test.
- ctest arm32_process_correctness: passed.
- sim oracle check: all 6 cases match goldens.
```

`spectral_wavetable.h:16` still documents "sample type determined at compile time
by SPECTRAL_EMBEDDED" — that is an accurate profile note (embedded vs host is a
required Phase E distinction), not a device name or an algorithm-body branch, so
it stays.

## Scope (Phase E increment)

One module. Remaining Phase E core files with in-body profile branches:
`oscillator_simd.c`, `spectral_vector_ops.c/.h`, `spectral_out.c`,
`spectral_synth_internal.c` — addressed in following passes.
