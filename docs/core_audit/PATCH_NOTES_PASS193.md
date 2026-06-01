# Patch notes — Pass 193: CTF sweep increment 33 — memory-safety lifecycle cross-cut (use-after-free / double-free / leak-on-error-path / uninitialized read) at the highest-risk cleanup sites (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. The final defect *category* not yet
cross-cut as its own pass is **manual heap lifecycle** — the class that the
file-by-file sweep touched per-file but that benefits from a dedicated look at the
densest, multi-exit cleanup sites across the tree:

```text
- double-free: a pointer reachable by free() on two converging control paths
- use-after-free: read/write through a pointer past its free()
- leak-on-error: an allocation not freed on an early-return / goto error path
- uninitialized read: a pointer/aggregate consumed before it is assigned
```

There are 141 `free()` sites tree-wide; this pass audits the ones with the genuinely
risky shape — an inline free plus a converging `fail:`/`cleanup:` label, or a single
buffer freed across many early returns.

**Outcome: clean audit. No defect found; no code changed.** The risky patterns are all
disarmed by the correct idioms: null-after-free before a converging label, block-scoped
single-free-before-goto, and single-owner mutually-exclusive early returns.

## What was checked and is correct

### spectral_cli.c — inline free + converging `fail:` label (the classic double-free trap)

```text
- `skip` is freed inline at :493 and IMMEDIATELY nulled at :494 (`skip = NULL;`). Every
  `goto fail` that can execute after :493 therefore hits a NULL-safe `free(skip)` at :512.
  goto fail sites before :493 (the argc-overflow / OOM branches at :477/:482) free the
  still-live `skip` exactly once at :512. Single free on every path.
- `eff_argv_heap` is freed on the SUCCESS path at :507 and the function returns at :509
  WITHOUT falling into `fail:`. The label's `free(eff_argv_heap)` at :513 is reachable only
  via a `goto fail`, all of which precede :507. So it is freed exactly once per path
  (success: :507; failure: :513). No double-free, no leak.
```

### spectral_seg_cache.c — block-scoped scratch + `goto append_error`

```text
- The endian-swap path allocates two DISTINCT block-scoped buffers — `Segment* scratch`
  (:524, inside `if(needs_swap)`) and `SegmentGpu* scratch` (:549, a separate block). Each
  is freed once at the END of its own block (:531, :558) BEFORE any `goto append_error`,
  and is out of scope at the label, so the label cannot double-free or leak it. The
  malloc-failure `else` branches stream element-by-element with no allocation to free.
- The `entries` buffer in the lookup path is freed on ~14 DISTINCT early-return error
  branches (:259..:395). Each is a mutually-exclusive `free(entries); return;` (single-owner
  on its own exit), and the success path frees once at :395. `segs` (:365/:377) is the
  paired free on the two branches that had also allocated it. No path frees the same pointer
  twice.
- spectral_seg_cache_result_free (:411) is the single owner-release for a lookup result:
  it frees segs (guarded by `capacity > 0` so mmap'd results are not free()'d — the comment
  at :838 documents the mmap/heap distinction), tile_ranges, tile_segment_ids. Idempotent
  shape (caller-owned one-shot).
```

### spectral_cli_pipeline.c / segment_mt.c / segment_pool.c — paired alloc/free

```text
- cli_pipeline cleanup (:1290 `cleanup:`) calls spectral_seg_cache_result_free once; the
  resources free helper (:139/:170/:172) frees mono/output/segs each guarded (segments freed
  only when `capacity > 0`, i.e. heap-owned, never the mmap'd cache view).
- segment_mt_free / segment_pool_free are symmetric teardowns: free the array + pending
  (mt) or every block + the block index (pool) exactly once; the pool's per-block frees
  (:83) iterate distinct allocations and the index free (:85) is once.
- gpu_tile_data_free is called under an `owns_td_store` ownership flag (cli_pipeline
  :1017/:1031/:1055) so a borrowed (cache-mmap'd) tile payload is never freed.
```

### Uninitialized-read shape

```text
- The aggregates that drive cleanup are zero-initialized at declaration (`ThreadBuffers tb =
  {0}`, `GpuTileData td_store = {0}`, `SpectralSegCacheLookupResult` zeroed before fill), so
  an early goto to a cleanup label that inspects them frees NULL, not garbage. Ownership
  flags (`owns_td_store`, `cache_built_this_run`) are initialized to 0 before the first
  branch that could goto.
```

## Verification

```text
- No source changed this pass (read-only cross-cut). Triad green by construction (re-run
  green for Pass 190 on this same tree; unchanged since):
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status — convergence

This increment closes the last defect *category* available to static adversarial review.
The sweep has now cleared 161-188 file-by-file AND cross-cut, tree-wide, every high-risk
class a DSP-engineer / mathematician / programmer review targets:

```text
- 189  unsigned-underflow-shift / integer div-by-zero / computed-size memcpy
- 190  float->int out-of-range conversion / signed-left-shift UB / transcendental NaN
- 191  strict-aliasing type-pun / signed-integer-overflow
- 192  OpenMP concurrency: data-race freedom + reduction determinism
- 193  heap lifecycle: use-after-free / double-free / leak-on-error / uninitialized read
```

All clean. The host-verifiable kernel has **no open defect leads** on any axis reachable
without execution. The two recorded observations (GPU fade-tail-under-time-stretch
non-monotonicity; Daisy SD `.spq` load skipping segment re-validation) remain bounded,
memory-safe, and deferred maintainer-directed because they are unverifiable on this host.

What remains is exactly the work static review *cannot* do: runtime numerical/algorithmic
correctness of the DSP math (phase-vocoder reconstruction accuracy, peak-interpolation
precision, window coherent-gain, cross-implementation parity). That is the explicit purpose
of **Phase D** — the compiled golden-vector harness — which is also the natural home for the
two deferred observations. Phase C is converged.
