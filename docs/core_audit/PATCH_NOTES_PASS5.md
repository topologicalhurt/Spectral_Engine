# Core audit pass 5 — GPU tile-span canonicalization

## Scope

This pass is in the architecture/minimality and hardware-correctness lane of the original implementation plan. It keeps the synthesis math contract unchanged and focuses on the GPU tile preprocessing support code.

## Findings addressed

1. `gpu_tile_preprocess()` duplicated segment-to-tile span calculation in the counting pass and the fill pass. Duplicate float/index logic is high-risk in GPU dispatch code because a future correction can update one pass but not the other.
2. Tile bounds were computed by converting `float` tile coordinates through `int`. Very large finite segment positions can make that cast implementation-defined or undefined before later range checks can protect it.
3. Segment end positions exactly on a tile boundary were assigned to the next tile, creating unnecessary tile references.
4. A valid input with zero active tile references could reach `malloc(0)` for `tile_segment_ids`; implementations may return `NULL`, which the old code treated as an allocation failure.
5. Cached tile-layout reuse did not explicitly gate on the canonical cache tile size, even though `gpu_tile_preprocess_cached()` accepts a `tile_size` argument.

## Changes

- Added a single `spectral_gpu_segment_tile_span()` helper.
- Uses sample-domain bounds with `ceil(start)` and `ceil(end)` so tiles cover exactly integer output sample indices that may be affected by a segment.
- Replaced both GPU tile count/fill loops with calls to the same helper.
- Added a zero-reference path that returns a valid tile range table with `segment_ids == NULL` and `total_refs == 0`.
- Gates cached tile-layout reuse to `SPECTRAL_GPU_TILE_SIZE`.
- Added static audit checks and `tests/core_math/test_core_pass5_static.py`.

## Validation

```sh
python3 tools/core_audit/core_static_audit.py .
python3 tests/core_math/test_core_pass5_static.py
make clean && make configure CMAKE_BUILD_TYPE=Debug
make desktop
```
