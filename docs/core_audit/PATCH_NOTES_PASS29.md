# Core audit pass 29: segment cache lookup metadata and data extent contract

## Summary

Pass 29 closes the remaining segment-cache lookup trust boundary.

Passes 26–28 made index insertion, data append, and index loading validate their
byte-count contracts. Lookup still trusted scalar fields from the on-disk index
before narrowing or using them:

```c
result->sample_rate = (int)e->sample_rate;
result->output_length = (size_t)e->output_length;
```

and it did not prove that the declared data extent was contained in the data
file before returning a hit on the heap fallback path.

## Bug

A corrupt index entry could declare:

```text
sample_rate outside runtime bounds
non-finite stretch/pitch
output_length not representable as size_t
tile_count/tile_total_refs mismatch
data_offset + total_data_bytes outside seg_cache_data.bin
```

The mmap path usually catches the data extent through the filesystem mapper, but
the heap fallback path only reads `seg_bytes` and can still return a cache hit
even when the GPU-segment/tile region is truncated or inconsistent.

## Fix

Lookup now validates entry metadata before writing the public result:

```text
sample_rate in [SPECTRAL_MIN_SAMPLE_RATE, SPECTRAL_MAX_SAMPLE_RATE]
stretch finite, positive, <= SPECTRAL_MAX_STRETCH
pitch finite and within configured pitch bounds
output_length representable as size_t
tile metadata internally consistent
```

After computing the checked data layout:

```text
seg_bytes + gpu_seg_bytes + tile_data_bytes
```

lookup validates:

```text
data_offset + total_data_bytes <= actual data file size
```

before attempting mmap or heap fallback.

## Reviewer Walkthrough

1. The lookup path finds an index entry with binary search exactly as before.
2. `seg_cache_entry_metadata_valid()` rejects corrupt scalar metadata before any
   narrowing assignment to the public lookup result.
3. The existing checked layout calculation still derives `seg_bytes`,
   `gpu_seg_bytes`, `tile_data_bytes`, and `total_data_bytes`.
4. `seg_cache_validate_data_extent()` opens the data file, reads its length, and
   proves the declared byte range is contained in the file.
5. Only after metadata and extent validation does lookup proceed to mmap or heap
   fallback.
6. Heap fallback can no longer return a hit for a truncated data file that
   happens to contain only the leading `Segment` bytes.

## Why this is critical

The segment cache is persistent input. Once data is on disk, it must be treated
as untrusted: it can be stale, truncated, externally modified, or produced by a
different build. Returning a cache hit from unvalidated metadata can propagate
invalid sample rates, invalid output lengths, or stale GPU/tile layout into the
render path.
