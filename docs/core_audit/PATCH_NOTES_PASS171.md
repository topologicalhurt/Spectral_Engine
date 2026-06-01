# Patch notes — Pass 171: CTF sweep increment 11 — host GPU-tile preprocess concurrency (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` (and the CLI/converter layer) and fix it in place.
This pass sweeps the **host GPU-tile preprocessing** kernel
(`core/port/host/spectral_gpu_tile.c`) — the two-pass parallel counting-sort that
maps analyzed segments to output tiles for the Metal/CUDA GPU synth backends.

The one defect is a **heap out-of-bounds read+write** caused by a per-thread
scratch array that is sized to a *clamped* thread count while the OpenMP region
that indexes it is left to spawn the runtime's *unclamped* default team.

`gpu_tile_preprocess` allocates one private histogram per thread:

```c
/* core/port/host/spectral_gpu_tile.c (pre-fix) */
int n_threads = spectral_omp_effective_thread_count();           /* line 118 */
thread_counts = spectral_calloc_array((size_t)n_threads, sizeof(uint32_t*));
for (int t = 0; t < n_threads; t++)
    thread_counts[t] = spectral_calloc_array(num_tiles, sizeof(uint32_t));

#pragma omp parallel                       /* line 141 — NO num_threads clause */
{
    int tid = omp_get_thread_num();
    uint32_t* my_counts = thread_counts[tid];   /* OOB if team size > n_threads */
    #pragma omp for schedule(static)
    for (size_t i = 0; i < sa.count; i++) { ... my_counts[tt]++; ... }
}
```

`spectral_omp_effective_thread_count()` (spectral_omp.h:32) is **clamped**:

```c
int n = omp_get_max_threads();
if (n < 1) return 1;
if (n > SPECTRAL_MAX_THREADS) return SPECTRAL_MAX_THREADS;   /* = 256 */
return n;
```

But an unqualified `#pragma omp parallel` uses the `nthreads-var` ICV — i.e.
`omp_get_max_threads()` threads (dynamic adjustment can only *lower* this, never
raise it). So on a host where `omp_get_max_threads() > SPECTRAL_MAX_THREADS` (a
machine with >256 hardware threads — dual-socket EPYC-class — or simply
`OMP_NUM_THREADS=512` in the environment; both user-reachable), the picture is:

- `thread_counts` is sized to `n_threads == 256`.
- the parallel region spawns up to `omp_get_max_threads() > 256` threads.
- threads with `tid in [256, omp_get_max_threads())` evaluate
  `thread_counts[tid]` — an **out-of-bounds read** of the pointer array, yielding a
  garbage `uint32_t*` — then dereference and increment through it
  (`my_counts[tt]++`): an **out-of-bounds write** to an arbitrary address. Heap
  corruption / crash.

This is inconsistent with the project's own established pattern: **every** other
per-thread-array parallel region in the codebase pins the team explicitly —
`peak_track.c` (`num_threads(tracker->n_threads)`), `analysis_fft.c`
(`num_threads(res->n_threads)`), `analysis_fused.c` (`num_threads(actual_threads)`),
`synth_cpu.c` (`num_threads(n_parts)`). The GPU-tile histogram region was the lone
omission, making it both a real reachable defect and a divergence from the
surrounding convention — the campaign's recurring defect class.

## Change

```text
1. Missing num_threads pin on the histogram parallel region  (concurrency / OOB)
   core/port/host/spectral_gpu_tile.c
   - gpu_tile_preprocess: added num_threads(n_threads) to the `#pragma omp parallel`
     that indexes thread_counts[omp_get_thread_num()]. This bounds the team to exactly
     n_threads (== the array's allocated length), so omp_get_thread_num() is always a
     valid index. Dynamic adjustment may still use FEWER threads (safe — the unused
     thread_counts[t] slots stay zero and the reduction sums them as zeros).
```

The **second** parallel region (the fill, `#pragma omp parallel for schedule(static)`
at the former line 222) was audited and **left unchanged**: it indexes no per-thread
array — it claims output positions with `#pragma omp atomic capture` into
`tile_cursors[tt]` and bounds-checks every `write_index` against `total_refs` before
storing — so it is correct for any team size. Adding a clause there would be a
non-fix style change, out of scope by KISS.

## Why this is correct and behaviourally inert on the verification host

On the little-endian, <256-thread build/test host `omp_get_max_threads()` is the
core count (well under `SPECTRAL_MAX_THREADS = 256`), so
`spectral_omp_effective_thread_count()` returns `omp_get_max_threads()` unchanged.
The pre-fix unqualified region already spawned exactly `omp_get_max_threads()`
threads; the post-fix `num_threads(n_threads)` requests exactly
`num_threads(omp_get_max_threads())` — the **same team size**. With the same team and
the same `#pragma omp for schedule(static)` partition, every per-thread increment and
the subsequent reduction are identical, so `tile_counts`, `tile_ranges`, `total_refs`
and `num_tiles` are bit-for-bit unchanged. The fill region is byte-unchanged. The
behaviour change is confined to hosts where `omp_get_max_threads() > 256`, where the
kernel now stays within its allocation instead of corrupting the heap.

## Finding

Audited and left unchanged (no defect) — the rest of the kernel is solid:
- `spectral_gpu_segment_tile_span` — full finite/positive gauntlet on stretch and
  seg start/length, double-precision span math, clamps start/end into
  `[0, num_tiles*tile_size]`, and derives `[start_tile, end_tile]` with
  floor/ceil-1 plus explicit `< num_tiles` bounds before the uint32 casts, so a
  hostile segment can never produce an out-of-range tile index.
- the histogram → reduction → prefix-sum → fill sequence is a correct two-pass
  counting sort: the reduction is overflow-checked per tile (`sum < tile_counts[i]`),
  the prefix sum is overflow-checked (`total_refs > UINT32_MAX - count`), and the
  fill bounds every `write_index` against both `tile_ranges[tt].count` and
  `total_refs` before storing.
- the `tile_cursors[t] == tile_counts[t]` post-condition loop proves every tile was
  filled exactly to its counted length (catches any miscount as
  SPECTRAL_ERR_FILE_CORRUPT rather than emitting a short/!long array).
- `gpu_tile_preprocess_scratch_free` frees `tile_counts`, `tile_cursors`, and each
  `thread_counts[t]` for `t in [0, n_threads)` then the outer array — matches the
  allocation exactly, and is reached on every error path via `goto cleanup` (the
  `out` payload pointers `tile_ranges`/`tile_segment_ids` are freed separately on the
  error path and transferred to `out` only on success, so no leak or double-free).
- all size arithmetic routes through `spectral_size_mul` / `spectral_calloc_array` /
  `spectral_malloc_array` (builtin-overflow guarded), and the `sa.count > UINT32_MAX`
  / `out_len > UINT32_MAX` guards run before any narrowing cast.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float. Only the pre-existing benign -mavx2 /
  -mno-avx512f unused-command-line-arg notes on host; no new warnings.
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- functional parity (the fix changes host codegen — it adds the num_threads argument
  to the fork call — so byte-identical-binary does not apply; verified by output
  parity instead). Exercised the GPU-tile path via --cache mode on a 340-segment
  fixture (desktop, sin_440hz.wav, n_fft=1024 hop=256 thresh=-70, cache cleared each
  run so the cache-miss path actually runs gpu_tile_preprocess):
    * FIXED build run1 vs run2  -> cmp BYTE-IDENTICAL  (path is deterministic
      run-to-run on this input, so an output cmp is a valid parity test).
    * HEAD (git-stashed) build vs FIXED build  -> cmp BYTE-IDENTICAL out_c.wav.
  Confirms the team-size pin is a no-op on the <256-thread host.
```

## Scope (Phase C increment)

Host GPU-tile preprocessing, one defect fixed: the histogram `#pragma omp parallel`
now pins `num_threads(n_threads)` so `thread_counts[omp_get_thread_num()]` can no
longer index past its `SPECTRAL_MAX_THREADS`-clamped allocation on a >256-thread
host — closing a heap OOB read+write and aligning the kernel with the codebase's
universal "pin the team that indexes a per-thread array" convention. The span math,
counting-sort reduction/prefix-sum/fill bounds, the cursor==count post-condition, and
the scratch-free path were audited and are clean. With this increment the Phase C
sweep has cleared fixed-point (161), analysis/peak-track (162), port/SIMD/out (163),
hashing/parsing/path (164), DSP-math/FFT-scaling + alloc/cache (165), synth-backends
+ analysis-orchestration (166), CLI/orchestration (167), embedded fade envelope (168),
core synth dispatch/internal helpers (169), the binary-deserialization/converter
surface (170), and the host GPU-tile concurrency kernel (171). Phase D (compiled
harness + LUT golden-vector loop) follows.
