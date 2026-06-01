# Patch notes — Pass 185: CTF sweep increment 25 — support/utility math cluster: hash lifecycle + resource-path canonicalization + alloc/overflow helpers + perf cost-model + Q15 primitives + error/peak-model (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the previously
**un-swept support/utility cluster** — the files that the 161-184 sweep referenced
only incidentally (0-2 patch-note mentions each), cross-referenced to confirm none had
a dedicated pass:

```text
- core/spectral_hash_xx32_xx3.c       hash-method lifecycle (init/reset/update/digest/
                                      consume_file {stream,full_direct,mmap}/destroy/oneshot)
- core/spectral_resource_fs.c         resource-path canonicalization (5-phase) + FNV-1a ID,
                                      checked byte-for-byte against the Python reference
- core/spectral_common.c              spectral_aligned_alloc (C11 aligned_alloc wrapper)
- runtime/spectral_utils.h            overflow-checked size add/mul + array alloc helpers
- runtime/spectral_perf_model.c       profile-driven embedded cycle cost model
- synth/math/spectral_q15.c + .h      Q15 fixed-point primitives + Q30->Q15 bulk convert
- core/spectral_error.c               error-code string table + bounded log formatting
- analysis/spectral_peak_model.c      peak-model policy/capability resolution
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol a
clean audit is a legitimate result and a defect must not be fabricated. The most
defect-prone surfaces here — the Q15 saturation/rounding primitives and the
host/embedded path-canonicalization parity — were re-derived against first principles
and against the pure-Python reference respectively, and are correct.

## What was checked and is correct

### Resource-path canonicalization — byte-exact with the Python reference

The C `spectral_resource_path_canonical` (5 phases: lowercase / control-strip /
slash-normalize, then component resolution, then generalized RLE) must produce a byte
sequence identical to `compress_path()` in
`tools/spectral_tools/testing/resource_hash_reference.py`, or embedded FNV-1a ID
lookups silently fail. Re-derived phase-by-phase:

```text
- Phase 1 (p1 vs _phase1to3_normalize_path_bytes): both strip 0x00-0x1f + 0x7f, ASCII-
  lowercase, map '\\'->'/' (lowercase BEFORE the backslash test in both — 0x5c is not a
  letter so order is moot but matches). Truncation boundary is IDENTICAL: C writes while
  `w + 1 < sz` (sz=1024) so scratch caps at 1023 bytes; Python breaks when
  `len(scratch) >= SPECTRAL_CANONICAL_MAX_BYTES` where MAX_BYTES = PATH_SIZE - 1 = 1023.
  Python's `-1` is exactly C's reserved null terminator, so both cap at 1023.
- Phase 4 (p2 vs _phase4_resolve_components): trailing-space strip BEFORE the ".." test
  (NTFS "dot-dot-space" bypass), ".." pops the last accepted component, "."/empty skipped,
  trailing dots stripped, all-dots ("...") components skipped, component cap 256 in both
  (C `comp_count < 256u` / `comp_starts[256]`; Python MAX_COMPONENTS = 256). comp_starts[i]
  records the position BEFORE the separator in both, so a pop rewinds identically.
- Phase 5 (p3 vs _phase5_generalized_rle): a run of N equal bytes -> repeated `chunk =
  min(remaining,255)`; chunk>=2 emits the 4-byte token `\x01 <byte> <hi_hex> <lo_hex>`,
  chunk==1 emits a literal — NEVER an RLE->literal fallback for runs>=2. A run of exactly
  256 yields token(255)+literal(1) on BOTH sides (verified the trailing-remainder case that
  could have diverged). All buffer-bound checks again mirror the C null-terminator reserve
  (C `w+4 < out_size` / `w+1 < out_size`; Python `len+4 > 1023` / `len+1 > 1023`).
- FNV-1a: offset basis 2166136261, prime 16777619, 32-bit; identical to the Python
  reference and applied over the canonical bytes only. NULL path -> 0.
```

### Q15 fixed-point primitives — saturation/rounding/overflow correct

```text
- spectral_float_to_q15 / _to_q31: clamp |f|>=1 to MAX/MIN, NaN/Inf -> 0, else f*scale; the
  open interval (-1,1) keeps the cast in range (f<1 strictly so f*32768 < 32768).
- spectral_phase_rad_to_q15: fmodf into (-1,1), +1 if negative, then the defensive
  `if (n >= 1.0f) n -= 1.0f` that catches the case where `n += 1.0f` rounds a tiny negative
  up to exactly 1.0f — keeps n in [0,1) so (n-0.5)*65536 stays in [-32768,32768) and the
  int16 cast cannot be OOB.
- spectral_omega_to_q88: <=0/non-finite -> 0; >255 divided by 4 then clamped to 255; max
  encoded = 255*256 = 65280 < 65536 (fits uint16). Lossy >255 path is documented (decode *4).
- spectral_smlad portable fallback: a0*b0 + a1*b1 + acc computed in uint32 wraparound — two
  Q15 products can sum to exactly 2^31 (= INT32_MAX+1), so a signed accumulate would be UB;
  the unsigned wrap matches ARM __smlad's non-saturating accumulator bit-for-bit. Each
  (q31_t)ai*bi product has magnitude <= 2^30 so the int32 multiply itself never overflows.
- spectral_mul_q15: ssat16(a*b >> 15) saturates the Q15_MIN*Q15_MIN = 2^30 -> (>>15)=32768
  case to 32767 (Q15 1.0). spectral_qadd/qsub16/32 + ssat16 all clamp to the correct
  [MIN,MAX]. spectral_q30_to_q15_scaled uses the >>15 (Q30->Q15, CMSIS MAC) shift, NOT >>16
  (the -6 dB bug fixed in pass 145), and null-guards only when count>0.
```

### Allocation + overflow helpers — correct guards

```text
- spectral_size_mul guards the divide with `a != 0` before `b > SIZE_MAX/a`; spectral_size_add
  uses the `a > SIZE_MAX - b` form; both prefer __builtin_*_overflow when available. The
  array wrappers (malloc/calloc/realloc_array, next_capacity_3_over_2) route every size through
  these, so no unchecked count*element multiply reaches malloc.
- spectral_aligned_alloc: rejects size 0 and `size > SIZE_MAX - (ALIGN-1)` BEFORE rounding up,
  and the rounded aligned_size is a multiple of SPECTRAL_CACHE_ALIGN as C11 aligned_alloc
  requires; pairs with plain free().
```

### Hash lifecycle, perf model, error table, peak model

```text
- spectral_hash full_direct: guards the (end-start) subtraction (`end_pos < start_pos`),
  the int64 cast (`start_pos > INT64_MAX`), and the size_t narrowing (`total_len_u64 >
  SIZE_MAX`, seeking back to stream on a too-large region). Empty files skip the update
  (len==0 short-circuit). Descriptor table available-flags gate FULL_MMAP off (returns
  BACKEND_UNAVAIL) until implemented. reset() rejects a zero-initialized (type==COUNT) object.
- spectral_perf_model: pure heuristic cost model, all sums accumulated in uint64 from uint32
  inputs x small cycle constants (no realistic overflow); threshold subtractions are guarded
  by the `active > threshold` if; NULL profile falls back to the M7 worst-case default; id
  range-checked (`(int)id < 0 || id >= COUNT`).
- spectral_error: pure switch->string table + bounded vsnprintf(context, 512, ...) with paired
  va_start/va_end. spectral_peak_model: policy/capability resolution, OR-composed flags,
  validate() enforces INTERP_BOUNDED -> window->peak_magsq present and rejects rectangular +
  log-parabolic peak-height; no arithmetic.
```

## Verification

```text
- No source changed this pass (read-only audit), so the Pass 184 state (itself == Pass 183
  green) is preserved by construction. Re-ran the host triad on the current tree to confirm:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-184 (see prior notes) and now the
support/utility math cluster (185, clean — the resource-path canonicalizer is byte-exact
with the Python reference across all five phases including the run-of-256 RLE remainder and
the null-terminator truncation boundary; the Q15 primitives saturate/round/wrap correctly
including the Q15_MIN*Q15_MIN and two-product-2^31 edge cases; the size add/mul/alloc helpers
guard every multiply; the hash full-direct path guards subtraction, int64 cast and size_t
narrowing; the perf model is overflow-safe and threshold-guarded; error/peak-model are
table/policy code with no arithmetic). All major compute AND support surfaces are now audited.
Phase D (compiled harness + LUT golden-vector loop) follows — still the natural home for the
deferred GPU fade-tail regression vector noted in Pass 184.
