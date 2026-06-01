# Patch notes — Pass 176: CTF sweep increment 16 — SpectralTracker lifecycle / per-thread storage / OpenMP reduction (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **SpectralTracker
lifecycle, per-thread segment storage, and the parallel finalize/merge reduction**
in `analysis/spectral_peak_track.c` (+ the `TrackSegment`/`Segment` layout contract
in `core/spectral_common.h` and `analysis/spectral_peak_track_internal.h`, and the
allocator in `core/spectral_common.c`). This is the concurrency + allocation half of
the peak-tracking subsystem (the DSP-math half was Pass 175; the host GPU-tile
concurrency was Pass 171):

```text
- spectral_tracker_create / _destroy / _free_segment_storage   alloc + teardown
- spectral_tracker_emit_segment (spectral_peak_interp.c)        per-thread grow + store
- spectral_tracker_process                                     OMP parallel pair loop
- spectral_tracker_finalize                                    prefix-sum merge reduction
- spectral_tracker_frame_time_from_index                       frame-index -> t_hop
- spectral_aligned_alloc / free pairing                        alloc-class consistency
- TrackSegment(32B) / Segment(64B) layout                      memcpy-prefix + pad clear
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol
a clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

### Allocation & teardown (`create` 727-826, `free_segment_storage` 282-295)

```text
- All sizing is overflow-checked: thread_slots = n_threads*CACHE_LINE_STRIDE,
  thread_slots_bytes, init_seg_bytes via spectral_size_mul; init_cap!=0 verified.
- seg_arrays is calloc'd (n_threads pointers, zero-initialised) so the goto-fail
  path can free() every slot safely even if the per-thread alloc loop aborts
  midway — un-allocated slots are NULL and free(NULL) is a no-op.
- Two distinct index conventions are used *consistently* everywhere:
    seg_arrays[tid]                                  (plain — array of n_threads ptrs)
    seg_counts/seg_capacities[tid*CACHE_LINE_STRIDE] (padded — false-sharing guard)
  verified identical in create (813-814), emit (171-214), finalize (1175,1223-1228),
  and free (285-287). No stride/plain mismatch.
- spectral_aligned_alloc wraps C11 aligned_alloc(CACHE_ALIGN, round_up(size)) with a
  SIZE_MAX-(ALIGN-1) overflow guard and a size==0 -> NULL contract; C11 aligned_alloc
  memory IS releasable with free(), so the seg_arrays[t]/seg_counts/seg_capacities/
  grown-array alloc(aligned)->free(plain) pairing is correct.
```

### Per-thread grow + store (`emit_segment` spectral_peak_interp.c:171-216)

```text
- count and capacity read from the padded slot; growth (count>=cap) doubles cap with
  full overflow guards: old_cap!=0, new_cap>=old_cap (wrap check), and both
  new_bytes=new_cap*sizeof(TrackSegment) and copy_bytes=count*sizeof(TrackSegment)
  via spectral_size_mul. memcpy(count segs) -> free(old) -> swap -> cap=new_cap.
- Each tid is owned by exactly one OMP thread, so seg_arrays[tid]/seg_counts[slot]
  are single-writer: the grow+store is race-free without a lock.
- Every stored field is finite/range-validated (spectral_tracker_emitted_segment_valid)
  before the slot is written; count advanced only after a successful store.
```

### Parallel pair loop (`process` 847-1147)

```text
- Scalar/finite guards on all precomputed steps; chunk_bins overflow-checked.
- n_pairs = chunk_n_frames-1, or chunk_n_frames iff overlap_magsq_row!=NULL. So the
  t==chunk_n_frames-1 "else" branch (next_row=overlap_magsq_row) is ONLY reached when
  overlap_magsq_row is non-NULL => next_row is NEVER NULL on any path.
- next_phase_row is left NULL on the overlap pair (there is no overlap_phase_row); this
  is safe — it only flows into spectral_peak_estimate_phase_advance, which returns 0 on
  !next_phase_row, and the default phase policy is IGNORE. No unconditional deref.
- global_frame_offset+t overflow pre-checked (global_frame_offset > SIZE_MAX-(n_pairs-1)
  -> ERR); frame_time_from_index forms frame_index*hop in double with finite/>=0/<=FLT_MAX
  guards. Failure flag polled at a power-of-two stride; per-thread local_failed fences
  the rest of the block; stats merged via #pragma omp atomic; seg_count>UINT32_MAX ->
  ERR. A mid-run failure makes finalize return empty (no partial-garbage SegmentArray).
```

### Prefix-sum merge reduction (`finalize` 1149-1349)

```text
- Per-thread offsets[t] are a running prefix sum of seg_counts (spectral_size_add
  checked); total_segs>UINT32_MAX -> fail; merge_bytes via spectral_array_bytes.
- Because offsets are a prefix sum, the parallel copy ranges [offsets[t],
  offsets[t]+count[t]) are pairwise DISJOINT — the #pragma omp parallel for copy is
  race-free; every slot in [0,total_segs) is covered exactly once.
- Layout contract (compile-time _Static_assert): offsetof start & width match between
  TrackSegment(32B: start@0..da@24,width@28) and Segment(64B: start@0..da@24, union@28
  with width@28 and _pad_w@32..63); sizeof(TrackSegment)<=sizeof(Segment). Therefore
  memcpy(sizeof(TrackSegment)=32) lands start..width in Segment bytes 0..31 (each field
  in its matching slot) and memset(&_pad_w,0,32) clears bytes 32..63 — full 64-byte
  init, width@28 set by memcpy and NOT clobbered by the memset (which begins at 32), no
  uninitialised heap bytes leak to the segment cache.
- segs/offsets are NULL-initialised so the fail path's free() is safe pre-allocation;
  every goto-fail precedes the success-path free at 1241 (no double-free); on success
  free_segment_storage NULLs the pointers so the later destroy() is a safe no-op.
  total_segs==0 -> segs stays NULL, valid empty result.
```

## Verification

```text
- No source changed this pass, so the Pass 175 green state is preserved by construction
  (host binaries byte-identical). Re-ran the full triad to formally close the pass:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared fixed-point (161), analysis/peak-track scan
(162), port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling +
alloc/cache (165), synth-backends + analysis-orchestration (166), CLI/orchestration
(167), embedded fade envelope (168), core synth dispatch/internal helpers (169),
binary-deserialization/converter (170), host GPU-tile concurrency (171), the oscillator
asin domain guard (172), the host SIMD quantized domain guard (173), the file-I/O + CLI
untrusted-input boundary cluster (174, clean), the peak frequency-estimation cluster
(175, clean), and the SpectralTracker lifecycle/per-thread-storage/OpenMP-reduction
cluster (176, clean — alloc/free class consistency, per-thread single-writer grow,
overlap-row NULL-safety, prefix-sum disjoint merge, and the TrackSegment/Segment
memcpy-prefix + _pad_w layout all verified). Phase D (compiled harness + LUT
golden-vector loop) follows.
