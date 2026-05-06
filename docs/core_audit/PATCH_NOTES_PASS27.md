# Core audit pass 27: segment cache append blob byte-count contract

## Summary

Pass 27 fixes the data-append side of the same persistence contract hardened by
Pass 26.

Pass 26 made segment-cache index insertion safe. The append path still wrote
segment, GPU-segment and tile blobs using raw byte products in several branches:

```c
(size_t)sa->count * sizeof(Segment)
(size_t)sa->count * sizeof(SegmentGpu)
(size_t)tile_count * sizeof(uint32_t) * 2u
(size_t)tile_total_refs * sizeof(uint32_t)
```

Those byte counts define the physical cache-data layout that later lookup uses
for heap reads or mmap-backed views. They must be checked once and reused.

## Bug

The append path could prove allocation success in one branch and then recompute
the write length with raw arithmetic in another branch. That violates the
kernel rule that allocation/copy/write byte counts must come from the same
checked size contract.

The path also used:

```c
(tile_ranges && tile_segment_ids) ? tile_count : 0
```

for index metadata. That can advertise tile metadata even when no tile blob was
written, for example when `tile_count > 0` but `tile_total_refs == 0`.

## Fix

The append path now computes checked byte counts before opening the append
writer:

```text
seg_bytes
gpu_seg_bytes
tile_ranges_bytes
tile_refs_bytes
```

All bulk append writes reuse those checked values. Big-endian scratch writes and
little-endian direct writes now share the same byte-count contract.

The tile metadata condition is now the same condition used to write the tile
blob:

```text
tile_count > 0 && tile_total_refs > 0 && tile_ranges && tile_segment_ids
```

## Reviewer Walkthrough

1. If `sa->count > 0`, the code checks both the `Segment` and `SegmentGpu`
   byte counts before opening the data-file append writer.
2. If tile data is present, the code checks both the range-pair byte count and
   the segment-id byte count before opening the writer.
3. The segment and pre-packed GPU segment append writes use `seg_bytes` and
   `gpu_seg_bytes`, not recomputed raw products.
4. Big-endian tile scratch writes use `tile_ranges_bytes` and `tile_refs_bytes`.
5. Little-endian direct tile writes use the same checked values.
6. Index metadata only records tile counts when the tile blob was actually
   written.

## Why this is critical

The segment cache is a persistence boundary. A wrong append length corrupts the
layout read by later lookup and mmap paths. A metadata/data mismatch can make a
future cache lookup interpret the wrong bytes as segment, GPU-segment or tile
layout data.
