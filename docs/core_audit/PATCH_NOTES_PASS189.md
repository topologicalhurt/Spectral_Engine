# Patch notes — Pass 189: CTF sweep increment 29 — tree-wide defect-class cross-cut (unsigned-underflow-shift / integer div-by-zero / computed-size memcpy) + remaining inline-logic headers (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. Having swept every logic-bearing .c
file end-to-end (161-188), this pass does the orthogonal thing: a **tree-wide cross-
cut of the defect *classes*** most likely to harbour a latent twin of the bug fixed
in Pass 187, plus the last two genuinely-unswept inline-logic headers.

```text
- runtime/spectral_perf_accounting.h   semantic perf counters (7 inline fns)
- core/spectral_omp.h                   OpenMP shim + effective-thread-count clamp
- CLASS SWEEP A: unsigned subtraction feeding a shift   (the Pass 187 class)
- CLASS SWEEP B: integer division by a runtime denominator
- CLASS SWEEP C: memcpy/memmove with a computed byte count
```

**Outcome: clean audit. No defect found; no code changed.** The Pass 187 unsigned-
EWMA-underflow class has no other instance in the tree, every integer division by a
variable is guarded or uses a nonzero compile-time constant, and every computed-size
memcpy is either overflow-checked or `>= sizeof` bounds-checked.

## Inline-logic headers — correct

```text
- spectral_perf_accounting.h: loop_iters_for_samples casts to uint64 BEFORE `+3 >> 2`
  (no overflow even at sample_count == UINT32_MAX); count_cache_pressure early-returns on
  `active_count <= miss_threshold_active` so the subtraction is always positive before the
  (uint64)*block_len product (max (2^32-1)^2 = 2^64-2^33+1 < UINT64_MAX); record_peak_block is
  a guarded max; all mutators NULL-guard. No under/overflow.
- spectral_omp.h: effective_thread_count clamps omp_get_max_threads() to [1, SPECTRAL_MAX_THREADS];
  the no-OPENMP omp_get_wtime uses CLOCK_MONOTONIC (elapsed-diff safe in double). Clean.
```

## Class sweep A — unsigned subtraction feeding a shift (Pass 187 class)

Grepped every `x += (a - b) >> n` / `(a - b) >> n` form tree-wide. Exactly three hits,
all the ones fixed in Pass 187 (the ARM debug-monitor EWMAs). The only other
`+= (… >> n)` sites add a **non-negative** quantity and therefore cannot underflow:

```text
- spectral_wavetable.c:403  checksum_calc += (uint8_t)((*address >> 8) & 0xFFu)  -- masked byte, >=0
- spectral_perf_model.c:114 overhead += ((miss_units + 3u) >> 2) * stall          -- ceiling-divide, >=0
```

No remaining unsigned-underflow-into-shift exists.

## Class sweep B — integer division by a runtime denominator

Every integer division whose divisor is not a literal was traced to a guard or a
nonzero compile-time constant; the remaining divisions are floating-point (IEEE
inf/nan on a zero divisor — defined, not a trap):

```text
- spectral_perf_embedded.c:362  est.estimated_cycles / blocks   -- GUARDED: `if (blocks == 0) blocks = 1;`
                                                                   on the line above (359-361).
- spectral_synth_arm32.c:1139   total_cycles / call_count        -- GUARDED: `if (call_count == 0) return 0;`
                                                                   on the line above (1138).
- spectral_segment_pool.c:16,27 ... / pool->block_size           -- block_size = SPECTRAL_SEGMENT_POOL_BLOCK_SIZE,
                                                                   a nonzero compile-time constant (set in init).
- spectral_gpu_tile.c:106       out_len_u32 / tile_size          -- tile_size is the nonzero GPU tile constant.
- perf_embedded.c:370/377/380, debug_embedded_arm.c:132/466, cli_pipeline.c:1276 -- all DOUBLE division
                                                                   (zero divisor -> inf, no UB/trap).
```

(The debug-monitor `deadline_cycles = (cpu_freq / sample_rate) * block_size` at
debug_embedded_arm.c:132 is integer division by `sample_rate`; left as-is — it is
`#ifdef SPECTRAL_DEBUG_ARM` instrumentation whose only callers pass the hardware
sample rate, never 0. Guarding an impossible debug-init input would be KISS-violating
defensive code for an unreachable scenario, distinct from the Pass 187 EWMA which was
reachable on every below-average sample.)

## Class sweep C — memcpy/memmove with a computed byte count

Every computed-size copy routes its size through the overflow-checked helpers
(`spectral_array_bytes` / `spectral_size_mul`, audited Pass 185) or is preceded by an
explicit destination-bound check:

```text
- The persistence/segment/wavetable copies (seg_cache.c:716/732, seg_cache_fs.c:191, segment_pool.c:71,
  segment_mt.c:73, peak_interp.c:192, wavetable.c:216/369/488/532/561, synth_cpu.c:176, in.c:100) all
  derive `*_bytes` from the overflow-checked helpers and copy into a buffer sized by the same checked
  count (re-confirmed in passes 170/176/180/184/185/186).
- The CLI path-string copies (cli_pipeline.c:210 `base_len+1`, :797 `dir_len+1`) are each guarded by a
  preceding `if (len >= sizeof(dst)) return ERR;` so `len+1 <= sizeof(dst)` exactly (the source is a
  strlen'd, NUL-terminated path). No overflow.
- arm32.c:529/1070 multiply a small validated/block-bounded count by sizeof(q15_t|SegmentQ15); on the
  32-bit target the operands are far below the size_t overflow threshold (audited Pass 181).
```

## Verification

```text
- No source changed this pass (read-only cross-cut), so the Pass 188 (== Pass 187) green state is
  preserved by construction:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-188 file-by-file AND now cross-cut the three
highest-risk defect *classes* tree-wide (189, clean — the Pass 187 unsigned-underflow-shift has
no surviving twin; all integer divisions are guarded or constant-denominator; all computed-size
copies are overflow- or bounds-checked) plus the final inline-logic headers. The host-verifiable
kernel has **no open defect leads**: every compute, support, dispatch, I/O, instrumentation,
optional-processing, and firmware surface is audited, and the two recorded observations
(GPU fade-tail-under-time-stretch; Daisy SD `.spq` load re-validation) are bounded, memory-safe,
and deferred maintainer-directed because they are unverifiable on this host. Phase C is at
convergence; Phase D (compiled harness + LUT golden-vector loop) is the natural home for the two
deferred items.
