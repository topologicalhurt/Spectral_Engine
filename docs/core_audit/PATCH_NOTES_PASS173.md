# Patch notes — Pass 173: CTF sweep increment 13 — host SIMD quantized domain-guard parity (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep across `core/`, `analysis/`,
`synth/` (and the CLI/converter layer). This pass sweeps the **host SIMD
oscillator port** (`core/port/host/oscillator_simd.c`) — the SSE/SIMDe vectorized
synthesis path that the desktop float CPU backend dispatches to for the
SIMD-eligible timbres (sine, saw, square, triangle, parabola, **quantized**, pwm).

The one defect is a **dropped domain guard in the SIMD `quantized` lane**
(`wave_quantized_4`). The canonical scalar contract `spectral_osc_quantized`
(`core/spectral_osc_formulas.h`) returns `0` whenever the intermediate
`scaled = rads*width` is non-finite or falls outside the representable `int`
range; the SIMD lane skipped that guard:

```c
/* core/port/host/oscillator_simd.c (pre-fix) */
static inline simde__m128 wave_quantized_4(simde__m128 rads, const void* ctx) {
    float width = *(const float*)ctx;
    if (width <= 0.0f) return simde_mm_setzero_ps();
    simde__m128 v_width = simde_mm_set1_ps(width);
    simde__m128 v_inv_w = simde_mm_set1_ps(1.0f / width);
    simde__m128 scaled = simde_mm_mul_ps(rads, v_width);
    simde__m128 truncated = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(scaled));
    return simde_mm_mul_ps(truncated, v_inv_w);   /* no out-of-int-range guard */
}
```

vs. the canonical scalar it is supposed to mirror:

```c
/* core/spectral_osc_formulas.h */
OSC_FORMULA_FUNC float spectral_osc_quantized(float rads, float width) {
    if (!isfinite(rads) || !isfinite(width) || width <= 0.0f) return 0.0f;
    float scaled = rads * width;
    float inv_w = 1.0f / width;
    if (!isfinite(scaled) || !isfinite(inv_w) ||
        scaled < (float)INT_MIN || scaled > (float)INT_MAX) {
        return 0.0f;                              /* <-- the dropped guard */
    }
    return (float)(int)scaled * inv_w;
}
```

`simde_mm_cvttps_epi32` (CVTTPS2DQ on x86, `vcvtq_s32_f32` on NEON) is *defined*
behavior for out-of-range / non-finite inputs — it returns `INT_MIN`
(`0x80000000`), **not** 0. So when `scaled` exceeds `INT_MAX` the SIMD lane emits
`INT_MIN * inv_w` (e.g. `-2.147` for `width ~ 1e9`) — an **out-of-`[-1,1]`
oscillator value** — exactly where the canonical scalar yields `0`.

`rads` is the normalized phase in `[-pi, pi]`, so `scaled` only overflows `int`
when `|width| > INT_MAX / pi ~ 6.84e8`. The analysis pipeline always assigns
`width = SPECTRAL_TRACK_DEFAULT_WIDTH = 0.5` (`analysis/spectral_peak_interp.c`),
for which `scaled <= ~1.57` and the lane is already correct. The large-width band
is reached only through **segment deserialization**: `spectral_segment_parser.c`
validates loaded segments with `spectral_segment_array_payload_valid`, which
checks `isfinite(s->width)` (`core/spectral_contracts.h:42`) **but not its
magnitude**. A crafted/corrupt `.seg` (the v2 format that "adds width field") can
therefore carry a finite `width = 1e9` that passes validation and reaches the
quantized SIMD path.

The divergence is also **internal to a single segment**: `osc_simd_fused_sustain`
runs the fade-in, fade-out, and sustain-tail samples through the *scalar* lane
`wave_quantized_1` → `spectral_osc_quantized` (guarded → 0), while the sustain
body runs through `wave_quantized_4` (unguarded → `INT_MIN*inv_w`). So one
quantized segment would emit `0` for its fade samples and out-of-range garbage
for its sustain samples. This is the campaign's recurring "missing domain/range
guard" defect class (cf. Pass 172's asin NaN clamp).

## Change

```text
1. Domain clamp on the SIMD quantized lane   (cross-path / contract parity)
   core/port/host/oscillator_simd.c
   - wave_quantized_4: build an in_range mask
       (scaled >= (float)INT_MIN) & (scaled <= (float)INT_MAX)
     and AND it onto the result, zeroing lanes the canonical scalar rejects.
     The >=/<= comparisons also reject NaN/Inf (every NaN comparison is false),
     so the mask reproduces spectral_osc_quantized()'s combined
     !isfinite(scaled) || scaled<INT_MIN || scaled>INT_MAX -> 0 behavior exactly.
```

`INT_MIN` / `INT_MAX` come from `<limits.h>`, already included transitively via
`spectral_osc_formulas.h`. `simde_mm_cmpge_ps` / `simde_mm_cmple_ps` /
`simde_mm_and_ps` are standard SIMDe SSE intrinsics (NaN-correct).

The SIMD **pwm** lane (`wave_pwm_4`) was audited and deliberately left unchanged:
it performs only a `(rads+pi)*inv_2pi < width` comparison producing `+/-1` (always
bounded, no `int` cast), and its only guards absent vs. the canonical
(`!isfinite(rads)`, `!isfinite(width)`) are already enforced upstream by
`spectral_segment_payload_valid` before the segment ever reaches synthesis — so
there is no reachable pwm divergence and KISS forbids a speculative no-op guard.
The embedded SIMD port (`core/port/embedded/oscillator_simd.c`) routes quantized
to a scalar stub (`osc_simd_available` excludes it there), so the canonical guard
already applies and no embedded change is needed.

## Why this is correct and behaviourally inert for in-range inputs

The mask only zeros lanes where `scaled` is out of `[INT_MIN, INT_MAX]` or
non-finite — exactly the set the canonical scalar already maps to `0`. For every
in-range lane the mask is all-ones and the result is bit-for-bit the previous
`(float)(int)scaled * inv_w`. Non-quantized timbres never call this function. So
the change only *replaces out-of-range SIMD garbage with the canonical `0`* and is
otherwise a no-op.

## Finding / reachability

The fix changes host codegen, so byte-identical-binary does not apply to the
quantized path; verified by output/value parity instead.

```text
- Function level (definitive). A direct harness compiled both the pre-fix and
  post-fix wave_quantized_4 against the canonical scalar spectral_osc_quantized
  over a 4,000,001-point rads sweep across [-pi, pi] at several widths:
    width=0.5  (analysis default) : pre-fix 0 divergences,  post-fix 0
    width=1.0                     : pre-fix 0 divergences,  post-fix 0
    width=1e9                     : pre-fix 1,265,743 divergences (worst |d|=2.147),
                                    post-fix 0
    width=1e30                    : pre-fix 4,000,003 divergences, post-fix 0
    width=FLT_MAX (scaled->Inf)   : pre-fix 4,000,003 divergences, post-fix 0
  So the defect is real (large finite width => SIMD emits out-of-[-1,1] values
  where the scalar/fade-lane returns 0) and the fix achieves exact scalar parity
  at every width, while being a no-op at the realistic width=0.5.
- Reachability of the width band. The analysis path hard-codes width=0.5; the
  large-width band is reachable only via deserialized .seg segments, whose
  validation (spectral_segment_array_payload_valid) checks isfinite(width) but
  not magnitude — consistent with the Phase C adversarial/untrusted-input model.
```

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float. Only the pre-existing benign
  -mavx2 / -mno-avx512f unused-command-line-arg notes on host; no new warnings.
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- integration inertness (output is float32 WAV; quantized takes the SIMD path on
  backend=cpu): desktop, sin_440hz.wav, timbre=6 (quantized), n_fft=1024 hop=256
  thresh=-70 threads=1 backend=cpu. HEAD (git-stashed) vs FIXED:
    * out_c.wav BYTE-IDENTICAL (sha256 6884b3dc855a975b...) — the realistic
      analysis->synth pipeline only ever uses width=0.5, so the guard is inert
      and the full-pipeline output is unchanged.
  The function-level sweep above is the conclusive proof that the fix closes the
  large-width divergence; the integration run confirms byte-inertness on the
  common path.
```

## Scope (Phase C increment)

Host SIMD oscillator port, one defect fixed: `wave_quantized_4` now mirrors the
canonical `spectral_osc_quantized` out-of-`int`-range / non-finite `-> 0` domain
guard, closing a cross-path divergence (SIMD sustain body vs. canonical scalar and
vs. this segment's own scalar fade lane) reachable via a finite-but-large
deserialized `width`. The SIMD pwm lane, the embedded SIMD port, and the rest of
the file were audited clean. With this increment the Phase C sweep has cleared
fixed-point (161), analysis/peak-track (162), port/SIMD/out (163),
hashing/parsing/path (164), DSP-math/FFT-scaling + alloc/cache (165),
synth-backends + analysis-orchestration (166), CLI/orchestration (167), embedded
fade envelope (168), core synth dispatch/internal helpers (169),
binary-deserialization/converter (170), host GPU-tile concurrency (171), the
oscillator asin domain guard (172), and the host SIMD quantized domain guard
(173). Phase D (compiled harness + LUT golden-vector loop) follows.
