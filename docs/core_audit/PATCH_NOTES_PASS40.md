# Core audit pass 40: segment cache payload validity contract

## Summary

Pass 40 completes the segment-cache persistence boundary.

Passes 26-31 hardened cache identity, index shape, data extents and scalar
metadata. The remaining hole was the actual payload: a cache entry could have a
valid index and valid byte extent while still carrying corrupt `Segment`,
`SegmentGpu` or tile-reference data.

## Bug

Lookup could return a cache hit after proving only this:

```text
index entry is well-shaped
data_offset + total_data_bytes is inside seg_cache_data.bin
```

It did not prove that the segment payload itself was valid before publishing:

```text
result->segments.segs
result->gpu_segs
result->tile_ranges
result->tile_segment_ids
```

The mmap path was especially dangerous because it returns direct pointers into
the cache file. A corrupt cache file could therefore inject NaN/Inf segment
fields or mismatched GPU pre-packed segments into synthesis without copying.

The store path had the mirror issue: it validated scalar metadata but could
persist corrupt segment arrays or tile references.

## Fix

The cache now validates payloads at both ingress and egress.

Store now rejects:

```text
non-finite or structurally invalid Segment payloads
invalid tile range layouts
tile segment IDs outside seg_count
```

Lookup now validates:

```text
mmap-backed Segment payload before publishing result->segments
mmap-backed SegmentGpu payload against spectral_segment_pack_gpu(Segment)
heap fallback Segment payload after endian conversion
tile range contiguity and tile segment IDs before publishing tile cache views
```

If the required Segment/GPU payload is corrupt, lookup returns
`SPECTRAL_ERR_FILE_CORRUPT`. Optional tile data remains an optimization: invalid
tile layout is not published.

## Reviewer Walkthrough

1. `seg_cache_segment_valid()` mirrors the segment-domain assumptions used by
   the synthesis hot path: finite fields, non-negative start/length, and
   non-negative omega.
2. `spectral_seg_cache_store()` validates the caller's segment array before
   opening the append writer.
3. Store also validates tile ranges are contiguous and tile segment IDs are
   within `sa->count`.
4. mmap lookup validates the mapped `Segment` array before assigning it to the
   public result.
5. mmap lookup validates the mapped `SegmentGpu` array exactly matches
   `spectral_segment_pack_gpu()` for the validated Segment payload.
6. heap fallback validates the copied/swapped Segment array before publishing it.
7. tile blobs are accepted only when ranges are contiguous and all referenced
   segment IDs are in range.

## Why this is critical

A cache hit is trusted as analysis output. Shape validation alone is not enough:
the bytes can be the right length and still contain invalid DSP state. Segment
and GPU-segment payload validation prevents a corrupted cache file from becoming
a silent synthesis input.
