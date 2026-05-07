# Core audit pass 48: GPU tile preprocess segment-ID narrowing contract

## Summary

Pass 48 fixes a public GPU tile preprocessing boundary.

`gpu_tile_preprocess()` stores segment references as `uint32_t` IDs, but accepts
a `SegmentArray` whose `count` is `uint32_t` in the public struct today only by
convention at call sites. The helper itself still needs to defend the narrowing
contract because it writes:

```c
tile_segment_ids[write_index] = (uint32_t)i;
```

## Bug

GPU tile preprocessing may be called independently of synth preflight. If a
future caller supplies a segment array with a count outside the `uint32_t`
segment-ID domain, the tile ID write would silently truncate the segment index.

## Fix

`gpu_tile_preprocess()` now rejects:

```text
sa.count > UINT32_MAX
```

before count/fill passes.

The write-side cast remains, but only after the function proves every `i` in the
parallel loop is representable as a tile segment ID.

## Reviewer Walkthrough

1. Basic output/tile/stretch validation remains.
2. The function validates segment pointer consistency.
3. It now checks `sa.count` against `UINT32_MAX` before parallel preprocessing.
4. The count pass and fill pass then operate inside the same segment-ID domain
   used by Metal/CUDA tile buffers.

## Why this is critical

Tile segment IDs are part of the GPU ABI. A host `size_t` loop index must not be
silently truncated into a `uint32_t` GPU reference.
