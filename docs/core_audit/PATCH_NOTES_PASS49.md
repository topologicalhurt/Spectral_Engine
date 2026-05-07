# Core audit pass 49: zero-reference GPU tile dispatch contract

## Summary

Pass 49 fixes a GPU backend edge case.

GPU tile preprocessing can produce a valid tile layout with zero segment
references: for example, all segments are outside the output range or rejected by
tile-span validation. CPU synthesis naturally reduces zeroed thread buffers in
that case. GPU backends should not dispatch kernels that require tile-ID buffers
when there are no tile IDs.

## Bug

Metal and CUDA proceeded into normal GPU dispatch even when:

```text
td.total_refs == 0
```

That can force zero-length tile-ID buffers through backend APIs. Even if the
shader would produce zeros for every tile, the dispatch setup itself may fail or
copy stale persistent output.

## Fix

After tile preprocessing, Metal and CUDA now handle zero-reference layouts as a
successful silent render:

```text
zero output buffer
record synthesis timing
skip GPU dispatch
return SPECTRAL_OK
```

CUDA synchronizes the stream first because segment upload may already have been
queued before tile preprocessing.

## Reviewer Walkthrough

1. Tile preprocessing still runs.
2. If it produces no segment references, the backend zeroes the output buffer.
3. Metal skips command-buffer construction.
4. CUDA synchronizes any already-submitted segment upload before returning.
5. Existing cleanup still releases owned tile data.

## Why this is critical

A zero-reference tile layout is a valid synthesis result: silence. It should not
be forced through backend dispatch paths that expect non-empty tile-ID buffers.
