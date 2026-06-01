# Patch notes — Pass 161: CTF sweep increment 1 — fixed-point UB cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass clears the
**fixed-point cluster** — the live Q15 synthesis path
(`synth/backends/arm/spectral_synth_arm32.c`, `synth/math/spectral_q15.{c,h}`,
`core/spectral_lut.h`). The build configs add no `-fwrapv` / `-fno-strict-overflow`,
and the default dev build is `-ffast-math -O2`, so signed-integer overflow and
out-of-range float→int conversions here are genuine UB the optimizer may exploit —
exactly the class that can silently miscompile the modular phase wrap the synth
depends on.

## Change

Four defects found and fixed (each behaviour-identical for in-range inputs, so the
`arm32_process_correctness` CTest stays bit-for-bit green; the UB is what's removed):

```text
1. Phase-accumulator signed-overflow + signed-left-shift UB
   synth/backends/arm/spectral_synth_arm32.c
   The working phase accumulator and frequency increment were carried as q31_t
   (signed int32) through the whole hot path, although the STORED state
   (SpectralActiveSoA::phase_acc, SpectralActiveSegQ15::phase_acc) is already
   uq32_t. Modular synthesis relies on wraparound, but every step was signed:
     - activation init  ((q31_t)seg->phase_q15 + 32768) << 16
       operand is 32768..65535 for phase_q15 >= 0, so <<16 exceeds INT32_MAX
       (signed left-shift overflow, UB)
     - phase += freq_inc / phase + freq_inc / freq_inc << 1 / p3 + freq_inc
       (signed-overflow UB on the modular accumulate)
   Fix: carry working `phase`/`freq_inc` as uint32_t end-to-end (matching the
   unsigned storage) across spectral_phase_batch4, synth_core_m7, synth_fade_m7,
   synth_segment_m7 and spectral_arm32_process; read freq_inc back from its q31_t
   slot with an explicit (uint32_t) cast; init
     uint32_t phase_acc = ((uint32_t)((int32_t)seg->phase_q15 + 32768)) << 16;
   Two's-complement-identical: the LUT index (uq16_t)(phase >> 16) is unchanged.

2. Amp-ramp activation multiply overflow
   synth/backends/arm/spectral_synth_arm32.c
   q31_t amp_advance = (q31_t)seg->da_q15 * (q31_t)sample_offset; multiplies a Q15
   delta by the (unbounded) sample offset in int32 -> overflows for long segments /
   large deltas (UB) BEFORE the spectral_ssat16 meant to bound it.
   Fix: compute amp_target in int64 (amp_q15 + da_q15*sample_offset) and clamp to
   [Q15_MIN, Q15_MAX]. Saturating intent preserved, overflow removed.

3. Phase-radians -> Q15 out-of-range float->int UB
   synth/math/spectral_q15.h  (spectral_phase_rad_to_q15)
   After `if (n < 0) n += 1.0f;`, a tiny-negative n rounds to exactly 1.0f, so
   (n-0.5f)*65536.0f == 32768.0f and (q15_t)32768.0f is an out-of-range float->int
   conversion (UB, C11 6.3.1.4). Live on the sim/converter segment_to_q15 path
   (spectral_synth_simulation.c:142, convert_segments.c:328).
   Fix: add `if (n >= 1.0f) n -= 1.0f;` to keep n in [0,1) (cyclically correct), so
   the product stays < 32768.

4. Portable spectral_smlad signed overflow
   synth/math/spectral_q15.h  (non-__ARM_FEATURE_DSP fallback)
   acc + a0*b0 + a1*b1 can sum two Q15 products past INT32_MAX (signed-overflow UB)
   and also diverges from the ARM __smlad (non-saturating, wraps) it mirrors.
   Fix: accumulate in uint32_t (defined wrap), matching the hardware accumulator.
   (No current caller, but it is a shipped primitive; removes a latent trap and
   aligns host/target semantics.)
```

## Finding

Audited and left unchanged (no defect):
- `synth/math/spectral_q15.c` — `spectral_q30_to_q15_scaled`: `accum[i] >> 15` is a
  well-defined arithmetic shift on signed q31, result saturated by `spectral_ssat16`.
- `core/spectral_lut.h` — `spectral_lut_sin`: the `lut[idx + 1]` guard read is safe
  because every LUT allocation is `SPECTRAL_OSC_LUT_SIZE + 1`
  (`spectral_lut_flash`, sim `lut[...+1]`, test `lut[...+1]`, daisy `s_osc_lut[...+1]`)
  and `spectral_lut_init_sine` fills index `<= SIZE`. The interpolation
  `s0 + (((s1-s0)*frac) >> 8)` stays in q15 range (frac in [0,240], result between
  s0 and s1), so the `(q15_t)` cast cannot be out of range.

The embedded DSP path is untouched: only the portable (`#else`) `spectral_smlad`
fallback was edited; the `__smlad`/`__qadd16`/`__ssat` intrinsic branch is unchanged.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mno-avx512f /
  -mavx2 unused-arg notes + the pre-existing spectral_accum_batch4 unused-function
  note on host).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift. The correctness test exercises the real
  spectral_arm32_process with the exact phase/amp math edited here and is
  bit-for-bit green, confirming the fixes are behaviour-preserving.
- desktop float render is byte-identical by construction: the arm32 synth body is
  #if SPECTRAL_EMBEDDED (compiles empty for desktop) and the desktop float backend
  calls none of the edited Q15 inlines (phase_rad_to_q15 / smlad / arm32 synth).
```

## Scope (Phase C increment)

Fixed-point UB cluster only. No algorithm/behaviour change for in-range inputs;
this removes signed-overflow and out-of-range-cast UB from the live Q15 path.
Regression coverage is the existing `arm32_process_correctness` CTest (the changed
math is on its exercised path). Next CTF clusters per ULTRAPLAN Phase C:
hashing/parsing/path, allocation/pool/cache, analysis/FFT/peak, port/SIMD/out.
