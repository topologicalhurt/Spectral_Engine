# Patch notes — Pass 180: CTF sweep increment 20 — segment-cache persistence cluster (spectral_seg_cache.c, clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **on-disk segment
cache** — the richest remaining untrusted-input + mmap surface in the tree. It turns a
cache key into a persisted `SegmentArray` (+ optional GPU tile blob) and reads it back
across a process boundary, so every field it trusts comes from disk:

```text
- core/spectral_seg_cache.c   seg_cache_key, seg_cache_bsearch (Java ~insertion conv.),
                              seg_cache_tile_blob_bytes (overflow-checked sizing),
                              seg_cache_entry_metadata_valid (per-field disk validation),
                              seg_cache_validate_data_extent (file-size bound),
                              seg_cache_validate_tile_blob (header vs index cross-check),
                              spectral_seg_cache_lookup (mmap fast path + heap fallback),
                              spectral_seg_cache_store (append data, rewrite index).
Consumer/producer contract traced into:
- core/spectral_contracts.h          spectral_segment_array_payload_valid,
                                     spectral_gpu_tile_layout_words_valid
- core/spectral_seg_cache_fs.*       data_append_* / index_load/write / data_map_ro
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol a
clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

### Every disk-sourced field is validated before it is narrowed or trusted

```text
- seg_cache_entry_metadata_valid (125-154) runs on the looked-up entry BEFORE any
  field is narrowed: sample_rate in [MIN,MAX]; stretch finite-positive <= MAX_STRETCH;
  pitch finite in [MIN,MAX]; output_length round-trips uint64->size_t->uint64; and the
  two cross-field invariants seg_count==0 => tile fields 0, and tile_count==0 iff
  tile_total_refs==0. A corrupt entry returns SPECTRAL_ERR_FILE_CORRUPT, not a read.
- The seg_count==0 hit short-circuits (288-292) with hit=1 and no data read — a valid
  empty-segment cache entry, not a degenerate path.
```

### Sizing is overflow-checked end-to-end, then bounded by the real file size

```text
- Lookup computes seg_bytes/gpu_seg_bytes via spectral_array_bytes and tile bytes via
  seg_cache_tile_blob_bytes (itself array-checked + two spectral_size_add), then folds
  them with spectral_size_add into total_data_bytes (294-301) — any overflow => OVERFLOW.
- seg_cache_validate_data_extent (156-181) re-checks size_t->uint64 round-trip, fetches
  the actual data-file size, and rejects unless data_file_size >= data_offset +
  total_data_bytes with an explicit `> UINT64_MAX - data_offset` pre-add guard. So the
  subsequent mmap/read of [data_offset, +total_data_bytes) is provably in-file.
```

### mmap fast path and heap fallback both re-validate payload before accepting

```text
- Fast path (310+, little-endian + mmap): maps RO, then runs
  spectral_segment_array_payload_valid on the mapped Segments AND
  spectral_segment_gpu_array_matches_segments on the packed GPU mirror; any failure
  unmaps and rejects. The GPU segs live at data_ptr+seg_bytes and tiles at
  +seg_bytes+gpu_seg_bytes — all inside the validated extent.
- Heap fallback (big-endian / no mmap): reads, endian-swaps, and runs the same payload
  validation; result_free is mmap-aware so neither path double-frees or unmaps a heap buf.
```

### Tile-blob acceptance cross-checks the on-disk header against the index entry

```text
- seg_cache_validate_tile_blob (184-238): requires tile_data_bytes >=
  sizeof(header); recomputes expected_bytes from the DISK header (num_tiles,total_refs)
  and demands expected_bytes == tile_data_bytes; demands th.tile_size==GPU_TILE_SIZE,
  th.num_tiles==e->tile_count, th.total_refs==e->tile_total_refs; then locates
  ranges_base/refs_base inside the blob and runs spectral_gpu_tile_layout_words_valid
  against e->seg_count. Only after all four gates does it publish pointers into the map.
  A size or metadata mismatch warns-once and skips tile data (segments still returned).
```

### Store path — validate-before-write, overflow-checked append, safe index insert

```text
- spectral_seg_cache_store (459-744): seg_cache_store_metadata_valid +
  spectral_segment_array_payload_valid + (when a tile blob is supplied)
  spectral_gpu_tile_layout_words_valid all run BEFORE the first byte is appended
  (478-492). seg_bytes/gpu_seg_bytes/tile_ranges_bytes/tile_refs_bytes are each
  spectral_array_bytes-checked (502-514).
- Append is transactional: data_append_begin -> writes (segments, packed GPU segs,
  optional tile header+ranges+refs) -> data_append_end; any write error jumps to
  append_error -> data_append_abort, so a half-written record never updates the index.
  Big-endian hosts swap into a scratch buffer, with a per-record scalar fallback when
  the scratch malloc fails (522-638) — no unswapped bytes ever hit disk.
- Index insert (661-739): bsearch hit updates in place (all fields overwritten, no
  stale data). Miss decodes ins=~pos, rejects ins>count as FILE_CORRUPT, then computes
  new_count=count+1 in size_t, rejects new_count>UINT32_MAX, array-checks prefix/suffix
  byte spans, allocates, and memcpy-splices prefix|new|suffix. The comment at 690-693
  documents exactly why `(uint32_t)(count+1)` would wrap and is therefore avoided. A
  failed index_load (652-659) rebuilds from empty rather than corrupting.
```

## Verification

```text
- No source changed this pass (read-only audit), so the Pass 179 green state is
  preserved by construction — host binaries remain byte-identical.
  Re-confirmed the gate:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-179 (see prior notes) and now the
segment-cache persistence cluster (180, clean — every disk-sourced field is validated
before narrowing, sizing is overflow-checked then bounded by the real file size, both
the mmap and heap read paths re-validate the segment + GPU-mirror payload before
accepting, the tile blob's on-disk header is cross-checked against the index entry and
`spectral_gpu_tile_layout_words_valid` before any pointer is published, and the store
path validates-before-write with a transactional overflow-checked append and an
overflow-guarded sorted index insert). Phase D (compiled harness + LUT golden-vector
loop) follows.
