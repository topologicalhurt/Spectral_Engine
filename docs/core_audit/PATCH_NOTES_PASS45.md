# Core audit pass 45: GPU tile fill cursor contract

## Summary

Pass 45 hardens the second phase of GPU tile preprocessing.

GPU tile preprocessing is a two-pass algorithm:

```text
count pass:  count how many segment references each tile needs
write pass:  fill tile_segment_ids using per-tile atomic cursors
```

The allocation sizes are derived from the count pass. The write pass must never
write outside those count-derived ranges.

## Bug

The write pass used:

```c
pos = tile_cursors[tt]++;
tile_segment_ids[tile_ranges[tt].start + pos] = (uint32_t)i;
```

without checking that `pos < tile_ranges[tt].count` or that the final write
index remained inside `total_refs`.

The counting and write passes should be deterministic, but the kernel contract
cannot rely on that assumption when the consequence is an out-of-bounds write.

## Fix

The write pass now validates:

```text
pos < tile_ranges[tt].count
tile_ranges[tt].start + pos < total_refs
```

before writing.

After the parallel fill, the code checks every tile cursor equals the original
tile count:

```text
tile_cursors[t] == tile_counts[t]
```

A cursor overrun is `SPECTRAL_ERR_OVERFLOW`; a cursor mismatch after fill is
`SPECTRAL_ERR_FILE_CORRUPT`, because the produced tile layout does not match
the counted layout.

## Reviewer Walkthrough

1. The count pass remains unchanged.
2. The write pass still uses atomic capture for per-tile cursor increments.
3. Each captured cursor is bounds-checked before the write.
4. Any overflow is recorded through a shared flag and handled after the parallel
   region.
5. After fill, every cursor is compared to its original count.
6. Tile data is only published if count and fill phases agree exactly.

## Why this is critical

GPU tile data feeds Metal/CUDA dispatch. If a tile fill overflows its
count-derived slice, the backend can upload corrupted segment IDs or overwrite
adjacent tile metadata. Count/fill agreement is the core correctness invariant
of the tile preprocessing algorithm.
