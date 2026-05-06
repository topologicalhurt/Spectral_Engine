# Core audit pass 26: segment cache index insertion overflow

## Summary

Pass 26 fixes a segment-cache index insertion overflow.

The segment cache stores its index entry count as `uint32_t`. On cache store,
inserting a new key used:

```c
(size_t)(count + 1)
```

where `count` is `uint32_t`.

If `count == UINT32_MAX`, the addition wraps in `uint32_t` before the cast to
`size_t`, producing zero. The code could then allocate an undersized index array
and copy/write entries through it.

## Bug

The old insertion path also copied prefix/suffix slices with raw byte products:

```c
ins * sizeof(SpectralSegCacheEntry)
(count - ins) * sizeof(SpectralSegCacheEntry)
```

Those copies are part of the persistent cache index mutation. They must use the
same checked byte-count discipline as the analysis/tracker/synthesis buffers.

## Fix

The insertion path now:

- validates `ins <= count`;
- computes `new_count` with `spectral_size_add((size_t)count, 1u, ...)`;
- rejects `new_count > UINT32_MAX`, preserving the on-disk count contract;
- checks the full index byte count with `spectral_array_bytes()`;
- checks prefix and suffix copy byte counts with `spectral_array_bytes()`;
- assigns `count = (uint32_t)new_count` after the checked mutation.

## Reviewer Walkthrough

1. `seg_cache_bsearch()` returns Java-style `~insertion_point`. The insertion
   point is converted to `uint32_t ins`.
2. `ins > count` is now treated as file/index corruption.
3. `new_count` is computed in `size_t`, not in `uint32_t`, and then checked
   against `UINT32_MAX` because the index format cannot represent more entries.
4. The allocation request and both `memcpy()` lengths are derived from checked
   byte counts.
5. Only after the new array has been allocated and populated does the code
   assign `count = (uint32_t)new_count`.

## Why this is critical

This is not a cache-only nicety. The segment cache is part of the kernel's
persistence path. A wrapped index count can underallocate the index array and
corrupt the cache's persistent metadata. Once corrupted, later lookups can mmap
or heap-read the wrong segment/tile regions.
