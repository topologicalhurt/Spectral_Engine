# Core audit pass 88: GPU tile cache one-shot layout identity contract

## Summary

Pass 88 fixes the same identity problem for the process-local GPU tile cache.

Pass 50 validated cached tile ranges and segment IDs against the current segment
count. That still does not prove tile layout identity, because tile layout also
depends on segment start/length.

## Bug

The cache lookup matched only:

```text
tile size
stretch
output length
```

and then validated IDs were in range. Two different segment arrays with the same
count and output shape can have completely different tile coverage.

A stale tile cache can therefore dispatch the wrong segment IDs per tile while
all IDs remain numerically in range.

## Fix

`Gpu_tile_cache_try_get()` is now a one-shot handoff. A successful lookup copies
the cached pointers into the output struct and immediately clears the cache.

Shape mismatches also clear stale cache state.

## Reviewer Walkthrough

1. Cache set remains available for the pipeline's immediate backend handoff.
2. Try-get still checks the known shape fields.
3. Try-get copies the layout into a local `GpuTileData`.
4. The process-local cache is cleared before returning.
5. Later calls cannot accidentally reuse stale tile coverage by shape alone.

## Why this is critical

Tile layout is a spatial acceleration structure, not just a shape. Reusing it
without segment identity can synthesize wrong output silently.
