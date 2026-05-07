# Core audit pass 50: GPU tile cache validation contract

## Summary

Pass 50 hardens the process-local GPU tile cache.

Pass 40 validates tile layouts when they come from the segment cache, and Pass
45 validates freshly preprocessed tile fills. The cache accessors themselves
still accepted and returned tile-layout pointers without validating the cached
layout against the current segment count.

## Bug

`gpu_tile_cache_try_get()` could return cached tile data based only on:

```text
tile size
stretch
out_len
```

It did not prove:

```text
ranges pointer is present
segment IDs pointer is present when refs > 0
tile ranges are contiguous
total refs matches range sum
every segment ID is < current segment count
```

A stale or incorrectly set process-local cache could therefore feed invalid tile
IDs into Metal/CUDA dispatch.

## Fix

The shared synth internals now include:

```c
gpu_tile_data_refs_valid(...)
```

`gpu_tile_cache_set()` fails closed for obviously invalid pointer/count
combinations.

`gpu_tile_preprocess_cached()` validates any cache hit against the current
`SegmentArray` before returning it. Invalid cache data is cleared and the code
falls back to fresh preprocessing.

## Reviewer Walkthrough

1. Cache set rejects `num_tiles == 0` with nonzero refs.
2. Cache set rejects refs without both ranges and segment IDs.
3. Cache hit still checks tile size/stretch/output length.
4. Before returning a hit, it validates range contiguity and segment ID bounds.
5. Invalid cached data is cleared.
6. Fresh preprocessing remains the fallback path.

## Why this is critical

The tile cache is an optimization, not an authority. GPU backends must not trust
process-global cached pointers unless they still describe the current segment
array and GPU tile layout.
