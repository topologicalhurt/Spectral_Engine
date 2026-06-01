# Patch notes — Pass 178: CTF sweep increment 18 — CPU additive-synthesis + wavetable + oscillator-math cluster (defect fixed: host SIMD PWM non-finite divergence) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **CPU additive
resynthesis path and the oscillator math it depends on** — the layer that turns a
validated `SegmentArray` into audio samples:

```text
- synth/backends/cpu/spectral_synth_cpu.c   thread-buffer arena, parallel-for over
                                            partitions, float/native reduce, the four
                                            per-segment callbacks
- core/spectral_synth_internal.c            synth_preflight_common (double** t_synth
                                            null-redirect to g_synth_timing_dummy)
- core/spectral_envelope.c                  raised-cosine fade params + in/out/combined
- core/spectral_wavetable.c                 builtins, .spwt load/save (w/ format
                                            conversion), load_raw, Intel-HEX load,
                                            load_buffer, lookup_f / lookup_q
- core/spectral_osc_formulas.h              8 canonical waveforms + normalize_phase +
                                            fast_sin + fade envelope (the cross-backend
                                            math contract, also mirrored in the Metal MSL
                                            string in oscillator.c)
- core/oscillator.c / oscillator_dispatch.c timbre_table dispatch, scalar segment, SIMD
                                            dispatch switch
- core/port/host/oscillator_simd.c          SIMDe SSE sustain kernels (DEFECT HERE)
- core/port/embedded/oscillator_simd.c       CMSIS counterpart (quantized/pwm = scalar
                                            fallback stubs — unaffected)
Consumer/producer contract traced into:
- core/spectral_contracts.h                 spectral_segment_payload_valid field guards
```

**Outcome: one real defect found and fixed** — the host SIMD PWM sustain kernel did
not honour the canonical non-finite domain guard, diverging from both the scalar
contract and its own fade-region lane. Everything else in the cluster is clean.

## The defect — `wave_pwm_4` does not mirror the canonical `spectral_osc_pwm` guard

### Canonical contract (`spectral_osc_formulas.h:129-132`)

```c
OSC_FORMULA_FUNC float spectral_osc_pwm(float rads, float width) {
    if (!isfinite(rads) || !isfinite(width)) return 0.0f;
    return (width > 0.0f) ? (((rads + SPECTRAL_PI) * SPECTRAL_INV_TWO_PI < width) ? 1.0f : -1.0f) : 1.0f;
}
```

So the contract is: **non-finite `rads` or `width` → 0**; finite `width <= 0` → 1;
otherwise the duty-cycle threshold gives ±1.

### What the SIMD sustain kernel did (before)

```c
static inline simde__m128 wave_pwm_4(simde__m128 rads, const void* ctx) {
    float width = *(const float*)ctx;
    if (width <= 0.0f) return simde_mm_set1_ps(1.0f);   /* NaN/Inf width slips through */
    ... norm = (rads+PI)*INV_TWO_PI; cmp = norm < width;
    return or(and(cmp,1), andnot(cmp,-1));              /* always +/-1, never 0 */
}
```

There is **no finite guard**. For a non-finite lane the comparison `norm < width`
is unordered (false), so the kernel emits `-1` (or `+1` for a `+Inf` width) where
the scalar contract emits `0`. Its sibling `wave_quantized_4` was given exactly
this guard in Pass 173 (the in-range mask), but the PWM kernel was missed.

### Why it is reachable (not an impossible scenario)

- The live desktop float path for a PWM timbre is, per `spectral_synth_cpu.c:287-289`
  and `oscillator.c:113`:
  `segment_fn_timbre → timbre_synth_segment → osc_simd_segment_pwm →
   osc_simd_fused_sustain(wave_pwm_4 [sustain lanes], wave_pwm_1 [fade lanes])`.
  `wave_pwm_1` is the canonical (returns 0 for non-finite); `wave_pwm_4` is the
  divergent sustain kernel — so the seam is **inside a single PWM segment**.
- Non-finite `rads` is reachable from untrusted input. `spectral_segment_payload_valid`
  (`spectral_contracts.h:32-43`) bounds `width` finite but checks only
  `isfinite(omega) && omega >= 0` with **no upper bound**. A deserialized `.seg`
  file may carry a finite-but-huge `omega` (e.g. ~1e38); the per-sample phase
  `p = phase0 + j*(alpha + beta*j)` then overflows to `±Inf` within a few samples,
  and `spectral_normalize_phase(±Inf) = ±Inf − 2π·floor(±Inf) = NaN`. This is the
  same "finite-but-large field from a deserialized segment" threat model the Pass 173
  quantized fix is written against.
- Consequence: the segment body plays a `±1` square wave where the contract (and its
  own fade ramps) intend silence — an audible seam / DC artifact, and a contract
  divergence. (Output stays bounded, so it is not a NaN-poisoning bug; it is a
  correctness/consistency bug.)

### The fix

`wave_pwm_4` now mirrors the canonical guard in all four corners:

```c
simde__m128 wave;
if (!isfinite(width)) return simde_mm_setzero_ps();   /* non-finite width -> 0   */
if (width <= 0.0f) wave = simde_mm_set1_ps(1.0f);     /* finite width<=0 -> 1     */
else { ... wave = +/-1 via the threshold compare ... }
/* per-lane finite-rads mask: |rads| <= FLT_MAX rejects NaN (unordered) and +/-Inf */
simde__m128 abs_rads = simde_mm_andnot_ps(simde_mm_set1_ps(-0.0f), rads);
simde__m128 finite   = simde_mm_cmple_ps(abs_rads, simde_mm_set1_ps(FLT_MAX));
return simde_mm_and_ps(finite, wave);
```

Corner-by-corner equivalence to `spectral_osc_pwm`:

```text
  width non-finite             -> 0            (canonical 0)            ✓
  width finite<=0, rads finite -> 1            (canonical 1)            ✓
  width finite<=0, rads !finite-> 0  (mask)    (canonical 0)            ✓
  width>0,        rads finite  -> +/-1 compare (canonical +/-1)         ✓
  width>0,        rads !finite -> 0  (mask)    (canonical 0)            ✓
```

`<float.h>` was added for `FLT_MAX`. The embedded CMSIS port is unaffected:
`osc_simd_segment_pwm`/`_quantized` there are scalar-fallback stubs and
`osc_simd_available` excludes PWM, so embedded PWM always runs the canonical scalar.

## What else was checked and is correct (no change)

```text
- spectral_synth_cpu.c: thread-buffer arena is cache-aligned + overflow-checked;
  parallel-for writes disjoint tb.bufs[p]; the float reduce runs spectral_f32_span_finite.
  The native reduce omits the finite check but synth_cpu_native/_wavetable_native have
  ZERO callers in the tree (desktop uses float synth_cpu; embedded uses saturating Q15
  that is finite by construction) — an unreachable defensive-parity gap, not a defect.
- synth_preflight_common takes double** t_synth and null-redirects to g_synth_timing_dummy
  before the driver dereferences *t_synth — no null deref.
- spectral_envelope.c raised-cosine fade = 0.5(1-cos(πj/L)); identical to the
  spectral_osc_formulas.h sin-form (sin((x-0.5)π) = -cos(πx)); non-overlapping, bounds-safe.
- spectral_wavetable.c: builtins write [0,SIZE) then samples[SIZE]=samples[0] (the +1 wrap
  guard); .spwt load validates magic/version/size/format/timbre_id, recomputes expected file
  size per the FILE's format, mallocs a typed temp of hdr.size, reads exactly payload_bytes,
  finite-validates, then converts; Intel-HEX load bounds every record
  (offset>expected || data_len>expected-offset -> ERR), requires covered_bytes==expected_bytes
  and an EOF record, and the data buffer is data[32] with byte_count<=capacity; load_buffer/
  load_raw size-check + finite-check before a copy_bytes=SIZE*sizeof memcpy into a SIZE+1 array.
  lookup_f reduces phase to [0,1), idx>=SIZE -> 0 guard makes samples[idx+1] in-bounds;
  lookup_q top-bits index <= SIZE-1. All paths bounds-safe.
- spectral_osc_formulas.h: normalize_phase maps to [-π,π) (round-to-nearest); sine=sinf;
  saw/square/triangle/parabola ranges verified; asin clamps to [-1,1] (Pass 172);
  quantized guards finite+[INT_MIN,INT_MAX] (Pass 173); fade = raised cosine. phase_to_rads
  delegates to normalize_phase, so the single-sample (timbre_oscillator) and segment-loop
  paths reduce phase identically.
```

## Verification

```text
- Host-compiled code changed (oscillator_simd.c, host profile), so byte-identity does
  NOT apply; verified by VALUE PARITY instead: for every finite-rads / finite-width input
  the finite mask is all-ones and the width>0 branch is unchanged, so the produced sample
  bits are identical to Pass 177; only the degenerate non-finite PWM lane changes (now 0,
  matching the canonical). The existing tests synthesize only finite phases, so they are
  unaffected.
- Full triad re-run:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared fixed-point (161), analysis/peak-track scan
(162), port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling + alloc/cache
(165), synth-backends + analysis-orchestration (166), CLI/orchestration (167), embedded
fade envelope (168), core synth dispatch/internal helpers (169), binary-deserialization/
converter (170), host GPU-tile concurrency (171), the oscillator asin domain guard (172),
the host SIMD quantized domain guard (173), the file-I/O + CLI untrusted-input boundary
cluster (174, clean), the peak frequency-estimation cluster (175, clean), the SpectralTracker
lifecycle/per-thread-storage/OpenMP-reduction cluster (176, clean), the STFT analysis FFT
driver + orchestration cluster (177, clean), and the CPU additive-synthesis/wavetable/
oscillator-math cluster (178 — host SIMD `wave_pwm_4` non-finite divergence FIXED to mirror
the canonical `spectral_osc_pwm` guard; the rest of the cluster clean). Phase D (compiled
harness + LUT golden-vector loop) follows.
