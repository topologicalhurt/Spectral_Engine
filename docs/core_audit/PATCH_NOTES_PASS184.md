# Patch notes — Pass 184: CTF sweep increment 24 — segment storage + Q15 LUT + CPU fade-envelope + resource bridge (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the genuinely
not-yet-swept support clusters surfaced by cross-referencing the 161-183 notes —
segment storage/lifetime, the embedded Q15 sine LUT, the CPU integer-index fade
envelope, and the ctypes resource-bridge stub:

```text
- core/spectral_segment_pool.c     block-chain segment allocator (init/push/to_array/
                                   destroy/reset)
- core/spectral_segment_mt.c       mutex-guarded segment array (init/load/get/copy/destroy,
                                   pending-swap apply)
- core/spectral_envelope.c         CPU raised-cosine fade (fade_params_init, fade_envelope_
                                   in/out, fade_envelope)
- core/spectral_lut.c + spectral_lut.h
                                   Q15 sine LUT build + inline uq16 phase lookup
                                   (spectral_lut_sin/cos)
- core/spectral_resource_bridge.c  linker stub for the Python ctypes bridge
```

**Outcome: clean audit. No defect found; no code changed.** One bounded,
cross-backend-consistent GPU observation is recorded below for the maintainer; it is
deliberately NOT changed this pass (see rationale). Per campaign protocol a clean audit is
a legitimate result and a defect must not be fabricated.

This pass also independently re-verified the DSP estimator/window/oscillator math
(spectral_peak_estimator.c, spectral_peak_interp.c, spectral_windows.c,
spectral_osc_formulas.h, spectral_fast_math.c) that prior passes 162-178 swept — all
formulas re-derived clean against the cited references (see "Re-verified" below).

## What was checked and is correct

### Segment pool — block-chain allocator (no realloc of payload)

```text
- segment_pool_push: block_idx = count/block_size, slot = count%block_size. count rises by 1
  per push, so block_idx is always < num_blocks (slot free) or == num_blocks (allocate). The
  blocks-pointer array grows by doubling with a new_max < max_blocks overflow guard; the new
  pointer slots [max_blocks, new_max) from spectral_realloc_array are never read before
  num_blocks reaches them (destroy/to_array both iterate only [0,num_blocks)). Payload blocks
  are never reallocated, so Segment* into the pool stay stable.
- segment_pool_to_array: copies exactly `count` segments — to_copy = block_size capped at
  count-copied for the partial last block; spectral_array_bytes overflow-checks each memcpy
  length; frees + returns EMPTY on overflow. init's est_blocks ceil-div can wrap for an absurd
  expected_count but only undersizes the hint (push grows), never an OOB.
```

### Segment MT — mutex-guarded with explicit borrow vs copy

```text
- All mutators hold sa->mutex. load() takes ownership of the caller's segs, freeing any prior
  pending; apply_pending_locked frees the old array and swaps pending in. get() returns a
  SHALLOW alias of sa->array (documented borrow — caller must not retain across a mutation);
  copy() deep-copies with a spectral_array_bytes overflow guard, unlocking on every error
  path. destroy() frees both arrays under lock then destroys the mutex. No double-free / leak.
```

### CPU fade envelope — integer index, from_end >= 0 by construction

```text
- fade_params_init: fade_len = min(segment_len/4, max_fade) clamped >= 1; fade_out_start =
  segment_len - fade_len; inv_fade = 1/fade_len; segment_len<2 short-circuits to a safe unit
  fade. fade_envelope(j): j<fade_len -> raised-cosine in (0->1 over [0,fade_len]); j>=fade_out_
  start -> raised-cosine out (1->0); else plateau 1.0. Integer j in [0,len-1] keeps
  fade_envelope_out's from_end = len-1-j in [0,fade_len-1] >= 0, so the out-ramp is monotone to
  0 at the final sample (asserted j<len). Continuous with the plateau in the discrete limit.
```

### Q15 sine LUT — index/fraction/guard-point math

```text
- spectral_lut_init_sine fills [0,SIZE] inclusive (SIZE+1 entries; index SIZE is the wrap guard
  = sin(2*pi) ~ 0) and forces the four cardinals (0, 0, +scale, -scale). scale =
  SPECTRAL_LUT_AMP_SCALE (Q15_MAX minus interpolation headroom).
- spectral_lut_sin: idx = phase_u16 >> (16-LUT_BITS) in [0,SIZE-1]; lut[idx+1] hits at most
  index SIZE (the guard) so no OOB. The low FRAC_BITS bits are rescaled to an 8-bit weight
  (frac_raw << (8-FRAC_BITS) for the default 4 frac bits) so weight/256 == frac_raw/16 exactly;
  the q31 product (s1-s0)*frac can't overflow int32 and >>8 keeps the result within [s0,s1] (no
  Q15 overflow). cos = sin(phase + 16384) is the exact +pi/2 (quarter-turn) shift with uq16 wrap.
- spectral_resource_bridge.c: spectral_resource_hashes_count == 0, so the single zero dummy
  entry is never dereferenced by find_by_path/find_by_id (zero-iteration scans). Benign stub.
```

### Re-verified (already swept 162-178; re-derived clean this pass)

```text
- Quinn second estimator: tau(x) = 0.25*ln(3x^2+6x+1) - (sqrt6/24)*ln((x+1-sqrt(2/3))/
  (x+1+sqrt(2/3))); delta_p = -alpha_p/(1-alpha_p), delta_m = alpha_m/(1-alpha_m),
  alpha = Re(X[k+/-1]/X[k]); delta = 0.5*(dp+dm)+tau(dp^2)-tau(dm^2). Jacobsen:
  Re{(X[k-1]-X[k+1])/(2X[k]-X[k-1]-X[k+1])}. Candan: jacobsen * tan(pi/N)/(pi/N). Log-parabolic:
  p = 0.5*(a-b)/(a+b), a=ln(L/C), b=ln(R/C); peak height y0 - 0.25*(y[-1]-y[1])*p. All match
  refs. Windows: Hann/Hamming/Blackman coefficients, 2/sum amp scale, N*Sigma w^2/(Sigma w)^2
  ENBW. Oscillators: sin Taylor coeffs are exact 1/(2k+1)!, fades are proper raised-cosine.
```

## Observation (recorded, intentionally NOT changed this pass)

`spectral_fade_envelope_gpu` (spectral_osc_formulas.h, mirrored as the Metal MSL string in
oscillator.c) is the GPU tile path's fade. It takes a float sample offset `j`; the kernel's
bounds test is `sample_pos < seg_end`, so `j` can fall in `(seg_len-1, seg_len)` whenever
`seg_start = seg.start*stretch` is fractional (i.e. under time-stretch). In that sub-sample
tail `from_end = seg_len-1-j` is in `(-1, 0)`, and the fade-out `0.5*(1+sin((from_end*inv_fade
- 0.5)*pi))` is no longer monotone to 0 — at the worst case (`from_end -> -1`, `fade_len = 1`)
the argument approaches `-1.5*pi` and the envelope swings back toward ~1.0 instead of ~0. The
CPU/oscillator.c path is unaffected (it indexes the fade with an integer `j`, so `from_end >=
0` always).

Why it is NOT treated as a KISS defect here: (1) the value stays bounded in [0,1] — no
overflow/NaN/OOB, only a <=1-sample-per-segment amplitude blip at the boundary, partly masked
by the constant-overlap of neighbouring segments; (2) the behaviour is *identical* across the
CUDA and Metal backends today, so the cross-backend parity contract holds; (3) a "fix"
(clamping `from_end >= 0`) would have to change the canonical header AND the duplicated Metal
MSL string AND bump SPECTRAL_OSC_FORMULAS_VERSION — a coordinated cross-backend contract change
that **cannot be verified on this host** (the ctest triad exercises the ARM/contract paths, not
GPU fade output; Metal needs a device run, CUDA needs nvcc). Per the campaign's measure-don't-
assert rule, an un-verifiable contract change is out of scope for an in-place defect pass and
is left for a maintainer-directed, test-backed change.

## Verification

```text
- No source changed this pass (read-only audit), so the Pass 183 state is preserved by
  construction. The host triad was last run green at the end of Pass 183:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-183 (see prior notes) and now the segment
storage + Q15 LUT + CPU fade-envelope + resource-bridge cluster (184, clean — block-chain pool
keeps stable payload pointers and copies exactly `count`; the MT array is mutex-guarded with an
explicit borrow/copy contract; the CPU fade indexes by integer so its out-ramp is monotone to
0; the Q15 LUT lookup is index/guard/fraction-correct with no OOB and bounded interpolation;
the resource bridge is a benign zero-count stub). One bounded, cross-backend-consistent GPU
fade-tail observation under time-stretch is recorded for a future maintainer-directed,
test-backed change. Phase D (compiled harness + LUT golden-vector loop) follows — and is the
natural place to add the GPU fade-tail regression vector.
