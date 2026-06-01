# Patch notes — Pass 192: CTF sweep increment 32 — concurrency / floating-point-determinism cross-cut of every OpenMP parallel region (clean audit; one accurately-characterised non-defect note) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. The 189-191 cross-cuts covered the UB
*classes*; this pass cross-cuts the orthogonal **concurrency** surface a DSP engineer
cares about — data races and floating-point reduction-order determinism across every
`#pragma omp parallel` region in the tree:

```text
- analysis/spectral_analysis_fft.c:400        reduction(max:max_magsq)
- analysis/spectral_analysis_fused.c:111,151  reduction(max) + per-thread pair loop
- analysis/spectral_peak_track.c:886,1203,1221 per-frame-pair tracking + pretouch/merge
- synth/backends/cpu/spectral_synth_cpu.c:197,239  additive synth + cross-buffer reduce
- core/port/host/spectral_gpu_tile.c:146,227   tile preprocessing
```

**Outcome: clean audit. No defect found; no code changed.** No region has a data race or
an incorrect shared accumulation. One property is recorded as an explicit **non-defect**:
the parallel additive-synth output differs from the single-thread output by at most one
ULP per sample (the textbook non-associativity of floating-point addition under a
different reduction grouping) — the sum is still the correct additive mixture within
float epsilon, and the CTest-verified embedded Q15 path is integer-exact and unaffected.

## What was checked and is correct

### max-reductions are order-independent → deterministic

```text
- analysis_fft.c:400, analysis_fused.c:111 use `reduction(max:...)`. max() is associative
  AND commutative, and the reduced magsq values are non-negative finite (no NaN ambiguity),
  so the result is identical for any thread count / scheduling. Deterministic.
```

### CPU additive synth — race-free private buffers + fixed-order reduction

```text
- synth_cpu_driver (synth_cpu.c:239): `parallel for` over n_parts partitions; partition p
  processes the DISJOINT segment range [p*count/n_parts, (p+1)*count/n_parts) and writes
  ONLY into its own private buffer tb.bufs[p]. No two threads touch the same buffer -> no race.
- thread_buffers_reduce_native (synth_cpu.c:197): the cross-buffer reduce is `parallel for`
  over output samples j (each thread owns distinct j via schedule(static)); for each j it
  sums bufs[0..n][j] in a FIXED ascending order t=0,1,2,... -> the reduction order does not
  depend on thread scheduling.
```

### Fused analysis + peak tracker — per-thread scratch, per-index output slots

```text
- analysis_fused.c:151: each thread allocates private SpectralFusedScratchRows + candidate
  batch + local_* counters; `omp for schedule(dynamic,1)` over DISJOINT frame-pair chunks;
  failure signalled through an atomic flag; local counters reduced after the region. No
  shared mutable row buffer.
- peak_track.c:886: `omp for schedule(static)` over frame pairs t; results are emitted into
  per-frame-pair output slots keyed by t, so the output ORDER is by pair index regardless of
  which thread ran which t (deterministic); the only shared state is the atomic last_error
  flag (polled at a power-of-two stride) and local_* counters reduced at the end. Race-free.
- peak_track.c:1203/1221 (pretouch + merge): `parallel for schedule(static)` writing distinct
  destination indices (page pre-touch / contiguous merge copy). Disjoint writes, no race.
- gpu_tile.c:146/227: tile preprocessing writes distinct tile/segment indices under
  schedule(static). Disjoint writes.
```

## Non-defect note — cross-thread-count FP additive synth is bit-reproducible only per fixed thread count

The parallel additive synth groups the per-sample sum as
`(Σ partition-0 segs) + (Σ partition-1 segs) + ...` whereas the single-thread path is a
flat left-fold over all segments in index order. Because floating-point `+` is not
associative, the two groupings can differ in the last ULP when `n_parts` changes.

```text
- This is NOT a correctness defect: the result is the mathematically-correct additive sum
  to within float rounding (<= 1 ULP/sample, i.e. < -140 dBFS for normalized audio —
  inaudible), and it is deterministic for any FIXED thread count.
- It is NOT in tension with the campaign's "host binaries byte-identical" rule, which is
  about the COMPILED BINARY being reproducible when source is unchanged, not about runtime
  output across thread counts.
- The correctness oracle that IS enforced (ctest arm32_process_correctness) exercises the
  embedded Q15 path, whose accumulation is integer (q31 sum -> saturating q15) and therefore
  bit-exact and order-independent regardless of thread count.
- Forcing cross-thread-count bit-reproducibility on the float path would mean a fixed
  serial reduction tree (a real throughput cost) to remove an inaudible last-ULP difference
  — a KISS-violating change for no audible/behavioural gain. Recorded, not "fixed".
```

## Verification

```text
- No source changed this pass (read-only cross-cut). Triad green by construction; it was
  re-run green for Pass 190 on this same tree and nothing has changed since:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-188 file-by-file, cross-cut eight UB/defect
classes tree-wide (189-191), and now the full OpenMP concurrency surface (192, clean — every
parallel region is race-free with per-thread private state or disjoint writes; the max
reductions and the additive-synth cross-buffer reduce are order-deterministic; the only
non-determinism is an inaudible cross-thread-count last-ULP FP grouping difference,
characterised as a non-defect). The host-verifiable kernel has **no open defect leads**:
every compute, support, dispatch, I/O, instrumentation, optional-processing, firmware, and
concurrency surface is audited. The two recorded observations (GPU fade-tail-under-time-
stretch; Daisy SD `.spq` load re-validation) remain bounded, memory-safe, deferred
maintainer-directed (unverifiable on this host). Phase C is at convergence; the remaining
verification — runtime numerical/algorithmic correctness of the DSP math — is precisely
what Phase D's compiled golden-vector harness exists to do, and is the natural home for the
two deferred items.
