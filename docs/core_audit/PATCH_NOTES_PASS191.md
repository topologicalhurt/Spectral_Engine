# Patch notes — Pass 191: CTF sweep increment 31 — tree-wide defect-class cross-cut round 3 (strict-aliasing type-pun / signed-integer-overflow in arithmetic) (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. Continuing the orthogonal defect-*class*
cross-cut (189: underflow-shift/div-zero/memcpy; 190: float→int/left-shift/transcendental),
this pass cross-cuts the final two UB classes a mixed float/fixed-point DSP kernel is
exposed to:

```text
- CLASS G: strict-aliasing — reading an object through a pointer of an incompatible
           type (C11 6.5/7: UB; the classic float<->int "bit reinterpret" trap)
- CLASS H: signed-integer-overflow in arithmetic — a + b / a * b / -a on a signed type
           where the result is not representable (C11 6.5/5: UB, unlike unsigned wrap)
```

**Outcome: clean audit. No defect found; no code changed.** Every bit-reinterpret goes
through a `union` (the well-defined idiom); the only raw pointer-casts are to genuinely
compatible types; and every signed multiply/add in the hot paths is bounded by
construction (the fixed-point ramps span a known Q15 range; the wide products are
promoted to `int64_t`; the EWMA deltas fixed in Pass 187 are `<< INT32_MAX`).

## Class sweep G — strict-aliasing (type-pun through an incompatible pointer)

```text
- float<->bits reinterpretation is ALWAYS done via `union { float f; uint32_t u; }`:
  spectral_endian.h:44, spectral_fast_math.c:36/93/94 (fast_inv_sqrt seed, fast_peak_log
  mantissa/exponent split). Union member-access type-pun is well-defined in C (6.5.2.3).
- The ONLY raw pointer-casts that reinterpret are to COMPATIBLE types:
    * analysis_fft.c:363/365  (float*)out_buf  where out_buf is fftwf_complex* — and
      fftwf_complex is `typedef float fftwf_complex[2]`, so the storage IS an array of
      float; reading it as float* is FFTW's documented, standards-compliant access. No
      aliasing violation (same effective type).
    * peak_track.c:1194  (void**)&segs for posix_memalign — void** is the API contract;
      no aliasing of a typed object.
- peak_track.c:1229 explicitly uses `memcpy` instead of a pointer cast to read a value
  out of a packed buffer "to avoid strict aliasing violation" — i.e. the codebase already
  applies the correct discipline where a genuine pun would otherwise occur.
```

No incompatible-type pointer pun exists.

## Class sweep H — signed-integer-overflow in arithmetic

Every signed multiply/add in the DSP hot paths is bounded so the result fits int32/int64,
or is deliberately promoted before the product:

```text
- spectral_synth_arm32.c fade ramps (697, 713, 988, 1034): `(int32_t)pos * fade_step`
  with fade_step = Q15_MAX/fade_len and pos < fade_len, so the product is < Q15_MAX
  (32767) by construction — three orders of magnitude below INT32_MAX. The fade-out form
  `Q15_MAX - (int32_t)into_fade * fade_step` has into_fade in [0,fade_len) likewise.
- spectral_synth_arm32.c:871  `(int64_t)seg->da_q15 * (int64_t)sample_offset` — the one
  product that CAN be large is explicitly widened to int64 before multiplying (da_q15 is
  int16, sample_offset up to a block span; int64 holds it with vast headroom).
- spectral_synth_arm32.c phase/freq accumulators (319, 865) are uint32_t — unsigned,
  modular-by-design (defined wrap, not UB), already noted in Pass 190 class E.
- spectral_debug_embedded_arm.c EWMA deltas (Pass 187 fix): `(int32_t)new - (int32_t)avg`
  on per-block cycle counts (~10^5-10^6) and us latencies (~10^3-10^5), both << INT32_MAX
  (~2.1x10^9), so the signed delta and its arithmetic shift cannot overflow (domain proof
  recorded in PATCH_NOTES_PASS187).
```

### Adjacent confirmation — the fade divisor guard (div-by-zero class, re-confirmed)

The fade ramp's `fade_step = (q15_t)(Q15_MAX / fade_len)` (arm32:670, 972) divides by a
runtime `fade_len`. This is safe: the activator clamps `fade_len` to `>= 1` at the
activation site before storing it —

```c
uint32_t fade_len = SPECTRAL_FADE_SAMPLES_EMBEDDED;   /* :857 */
if (fade_len > seg_length / 2) fade_len = seg_length / 2;
if (fade_len == 0) fade_len = 1;                       /* :859 — divisor never 0 */
```

— and the comment at lines 665-668 documents the invariant. This complements Pass 189's
integer-division class sweep (which traced the perf/pool/gpu divisors); the arm32 fade
divisor is the embedded-only analogue and is likewise guarded.

## Verification

```text
- No source changed this pass (read-only cross-cut). The triad was just re-run green for
  Pass 190 on this same tree and nothing has changed since:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-188 file-by-file and cross-cut EIGHT
independent defect classes tree-wide — (189) unsigned-underflow-shift / integer
div-by-zero / computed-size memcpy; (190) float→int out-of-range conversion /
signed-left-shift / transcendental-domain NaN; (191) strict-aliasing type-pun /
signed-integer-overflow — every one clean. The host-verifiable kernel has **no open
defect leads**: every compute, support, dispatch, I/O, instrumentation,
optional-processing, and firmware surface is audited file-by-file, and the eight
highest-risk UB/defect classes have been confirmed absent across the whole tree. The two
recorded observations (GPU fade-tail-under-time-stretch; Daisy SD `.spq` load
re-validation) remain bounded, memory-safe, and deferred maintainer-directed because they
are unverifiable on this host. Phase C is at convergence; Phase D (compiled harness + LUT
golden-vector loop) is the natural home for the two deferred items.
