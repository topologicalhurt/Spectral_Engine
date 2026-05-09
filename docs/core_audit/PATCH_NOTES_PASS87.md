# Core audit pass 87: GPU segment cache one-shot identity contract

## Summary

Pass 87 fixes a process-local GPU segment cache identity bug.

The cache stores a pre-packed `SegmentGpu*` pointer so Metal/CUDA can avoid
repacking mmap-backed segment-cache payloads.

## Bug

The cache lookup keyed only on segment count:

```c
gpu_seg_cache_try_get(count, &cached_gpu_segs)
```

A later synthesis call with the same segment count but different segment content
could reuse a stale `SegmentGpu*` pointer if the cache was not explicitly reset
by the caller.

Count is not identity.

## Fix

`Gpu_seg_cache_try_get()` is now one-shot. On successful retrieval it copies the
pointer to the caller and immediately clears the process-local cache.

It also clears the cache on count/pointer mismatch.

## Reviewer Walkthrough

1. `gpu_seg_cache_set()` still installs a pointer for the next backend call.
2. `gpu_seg_cache_try_get()` still requires the expected segment count.
3. A successful lookup clears the cache before returning.
4. A failed lookup also clears stale state.
5. No backend can accidentally reuse a previous render's prepacked segments by
   count alone.

## Why this is critical

A pre-packed GPU segment pointer is a payload identity, not just a shape. Reusing
it by count can synthesize the wrong segment set while looking like a valid fast
path.
