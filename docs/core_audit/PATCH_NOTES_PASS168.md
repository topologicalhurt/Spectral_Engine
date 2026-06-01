# Patch notes — Pass 168: CTF sweep increment 8 — embedded fade envelope (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass sweeps the **embedded
Q15 synthesis fade envelope** in the real Cortex-M backend
(`synth/backends/arm/spectral_synth_arm32.c`) — the DSP correctness heart of the
`simulate`/`embedded_*` targets and the path guarded by the
`arm32_process_correctness` CTest.

The defect is a **fade-step / fade-length decoupling that produces an amplitude
discontinuity (click) on short segments**.

At segment activation the per-segment fade length is clamped to the segment size:

```c
uint32_t fade_len = SPECTRAL_FADE_SAMPLES_EMBEDDED;   /* 32 */
if (fade_len > seg_length / 2) fade_len = seg_length / 2;
if (fade_len == 0) fade_len = 1;
```

so `fade_len` ranges over `[1, 32]` and equals `seg_length/2` for every segment
shorter than `2*SPECTRAL_FADE_SAMPLES_EMBEDDED` (= 64) samples. But both synthesis
paths built the linear Q15 ramp with the **fixed** constant
`SPECTRAL_FADE_STEP_Q15` (= `Q15_MAX / SPECTRAL_FADE_SAMPLES_EMBEDDED` = `Q15_MAX/32`
= 1023), independent of the actual `fade_len`:

- M7 path `synth_segment_m7`: fade-in seed `seg_offset * STEP` + step `STEP`
  passed to `synth_fade_m7`; fade-out seed `Q15_MAX - into_fade*STEP` + step `-STEP`.
- generic/scalar path: the identical four uses inline.

For a short segment the fade-in ramp therefore stops at
`(fade_len-1) * (Q15_MAX/32)` instead of reaching full scale at `fade_len`. The
fade-OUT, however, always *starts* at `Q15_MAX` (its seed is `Q15_MAX - 0*STEP`),
so the rendered envelope jumps from a fraction of full scale to full scale at the
fade-in→fade-out boundary, and ends the segment at a large non-zero value — two
amplitude discontinuities, i.e. clicks. Worked example, `seg_length = 32`
(`fade_len = 16`):

```text
                last fade-in sample      peak / fade-out start     last sample
  pre-fix:      15*1023/32767 = 0.47  ->  1.00  (jump 0.53)         0.53  (click to 0)
  fixed:        15*2047/32767 = 0.94  ->  1.00  (jump 0.06)         0.06  (clean)
```

The desktop/GPU reference backend does not have this defect: `spectral_envelope.c`
uses `inv_fade = 1.0f / fade_len`, i.e. a step tied to the actual fade length. The
embedded backend was the sole divergent implementation.

## Change

```text
1. Fade step decoupled from fade_len -> short-segment amplitude discontinuity
   synth/backends/arm/spectral_synth_arm32.c
   Both synthesis paths now derive a per-segment step from the ACTUAL fade_len:
       q15_t fade_step = (q15_t)(Q15_MAX / fade_len);
   fade_len is guaranteed >= 1 at activation (clamp above), so the divide is safe,
   and for fade_len == 32 (every segment >= 64 samples) it equals the old
   Q15_MAX/32 = 1023 EXACTLY -> long-segment output is byte-identical. Substituted
   for the fixed constant at all eight use sites:
     - synth_segment_m7 (M7): fade-in seed + step, fade-out seed + (-step).
     - generic/scalar path:   fade-in seed + step, fade-out seed + (-step).
   The fade-region boundary math (seg_fade_out_start, fi_end, fo_start, into_fade)
   was already correct and is unchanged; only the ramp SLOPE is corrected.

2. Removed the now-dead fixed-step macro  (divergent-duplicate hazard)
   synth/math/spectral_q15.h
   SPECTRAL_FADE_STEP_Q15 had exactly one consumer (this backend) and now has
   none. Leaving a constant that encodes the wrong (fixed) fade semantics is a
   drift hazard for any future caller (cf. pass 164's dead divergent validator),
   so the definition is deleted. No other TU referenced it.

3. Added a short-segment fade-continuity regression test
   tests/arm_core/test_arm32_process.c  (test_short_segment_fade)
   The existing arm32_process_correctness golden only exercises a long segment
   (length = sr/2 -> fade_len = 32), so it proves the long-segment path is
   byte-identical but never touched the short-segment case the fix changes. The
   new case pins the oscillator at its positive peak (freq=0, phase at the quarter
   turn) with amp=1.0, so each rendered sample reads back the fade envelope value
   directly, and asserts on a len=32 (fade_len=16) segment: the ramp reaches full
   scale at the midpoint, the fade-in/fade-out boundary jump is small, and the
   fade-out ends near zero. This FAILS on the pre-fix fixed step and passes on the
   fix.
```

## Why this is correct and behaviourally inert for the long-segment path

`fade_len == 32` for every segment of >= 64 samples (the activator clamp only
bites below 64). For those segments `Q15_MAX / 32 == 1023 == SPECTRAL_FADE_STEP_Q15`,
so the ramp seed and step are bit-for-bit what they were — analysis, synthesis and
the rendered audio are unchanged. The behaviour change is confined to segments
shorter than 64 samples, where the ramp now correctly spans `[0, full-scale]` over
`fade_len` rather than stopping short. This is confirmed by the
`arm32_process_correctness` golden (a long segment) passing unchanged.

## Finding

Audited and left unchanged (no defect) in this backend:
- `spectral_arm32_validate_segment_data` — accepts any `length >= 1` (overflow-checked
  `start+length`, `start < output_len`, monotone `start` AND `end`, active-count
  `<= SPECTRAL_ARM32_MAX_ACTIVE` via the `first_live` sweep) and rejects unsupported
  chirp; confirms short segments are VALID, reachable input (so the fade defect is
  live, not theoretical).
- the batch helpers `spectral_phase_batch4` / `spectral_amp_batch4` /
  `spectral_accum_batch4`, the saturating `spectral_qadd16` ramp accumulation, and
  the three-region boundary partition (fade-in / sustain / fade-out) are all correct.

Flagged for a later pass (out of scope for this fade fix, not addressed here):
- `core/spectral_synth_internal.c:84` `synth_zero_output_if_valid` is a defined-but-
  unused static function on the `simulate_daisy`/`embedded_arm`/`embedded_arm_float`
  targets (`-Wunused-function`) — either dead code or a missing call site.
- `spectral_accum_batch4` warns `-Wunused-function` in the SPECTRAL_ARM_M7-forced
  `arm_core_test` build (its only caller is the generic `#else` path); benign, but a
  candidate for a `static inline`/guard cleanup.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mavx2 /
  -mno-avx512f unused-command-line-arg notes on host, and the pre-existing
  synth_zero_output_if_valid unused-function note on the embedded/daisy targets).
- ctest: 4/4 PASSED — arm32_process_correctness (now incl. the new
  test_short_segment_fade case), core_contracts, core_guarantees,
  core_guarantees_drift.
- new regression case output (fix in place):
    test_short_segment_fade (len=32 fade_len=16):
      fade_in_end=0.935 peak=0.998 fade_out_end=0.063 jump=0.063
  (the pre-fix fixed step yields fade_in_end ~0.47 / jump ~0.53 / fade_out_end
  ~0.53 — the click — and would fail the new asserts.)
- functional parity on resources/testing/sin_440hz.wav: with matched analysis
  params (n_fft=1024 hop=256 thresh=-70) desktop and simulate both detect 340
  segments (exact analysis parity; the previously-noted 657 was simply the desktop
  binary's different DEFAULT n_fft/hop, not a divergence). Every sin_440hz segment
  is long (~259 samples avg -> fade_len=32), so the rendered output is byte-identical
  to pass 167; the long-segment golden passing is the byte-level proof.
- the arm32 backend changed, so sim/embedded binaries are not byte-identical to
  pass 167 for short-segment inputs (by design); correctness for the long-segment
  path is established by the unchanged golden, and the short-segment fix by the new
  test.
```

## Scope (Phase C increment)

Embedded fade-envelope cluster, one defect fixed: the Q15 synthesis fade ramp now
spans full scale over the ACTUAL clamped `fade_len` instead of a fixed `Q15_MAX/32`
step, removing the short-segment amplitude discontinuity (click) in both the M7 and
generic paths and matching the desktop reference's `1/fade_len` semantics. The now-
dead `SPECTRAL_FADE_STEP_Q15` macro is removed, and a short-segment fade-continuity
regression test locks in the corrected behaviour. Behaviourally inert (byte-identical)
for every segment >= 64 samples. With this increment the Phase C sweep has cleared
fixed-point (161), analysis/peak-track (162), port/SIMD/out (163), hashing/parsing/
path (164), DSP-math/FFT-scaling + alloc/cache (165), synth-backends + analysis-
orchestration (166), CLI/orchestration (167), and the embedded fade envelope (168).
