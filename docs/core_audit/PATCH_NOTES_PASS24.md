# Core audit pass 24: tracker segment-storage allocation overflow

## Summary

Pass 24 closes the tracker segment-storage allocation boundary exposed by the
Pass 21 cleanup work.

Pass 21 centralized tracker segment cleanup and switched the per-thread segment
pointer table to `spectral_calloc_array()`. The remaining tracker storage still
used raw byte products for:

```c
(size_t)n_threads * SPECTRAL_CACHE_LINE_STRIDE * sizeof(size_t)
init_cap * sizeof(TrackSegment)
new_cap * sizeof(TrackSegment)
count * sizeof(TrackSegment)
```

Those products define the actual storage later indexed and copied by the peak
tracker. They must be checked before allocation or copy.

## Bug

The kernel could prove neither:

```text
n_threads * padded_counter_stride * sizeof(size_t)
```

nor:

```text
segment_capacity * sizeof(TrackSegment)
```

before allocating. If either product wrapped, the tracker could allocate a
smaller buffer than the hot path later writes through.

The growth path had the same issue: it checked `new_cap < old_cap`, but still
passed `new_cap * sizeof(TrackSegment)` to `spectral_aligned_alloc()` and
`count * sizeof(TrackSegment)` to `memcpy()` without checked byte counts.

## Fix

`SpectralTracker` creation now computes:

```text
thread_slots
thread_slots_bytes
init_seg_bytes
```

with `spectral_size_mul()` before allocation. The padded count/capacity arrays
and each initial segment array use those checked byte counts.

The segment growth path now computes:

```text
new_bytes
copy_bytes
```

with `spectral_size_mul()` before allocation and copy. Arithmetic failure is
reported as `SPECTRAL_ERR_OVERFLOW`; allocation failure remains
`SPECTRAL_ERR_MEMORY`.

## Reviewer Walkthrough

1. `spectral_tracker_create()` converts `n_threads` to the padded counter-slot
   count using checked multiplication.
2. It then checks the byte count for the padded `seg_counts` and
   `seg_capacities` arrays before calling `spectral_aligned_alloc()`.
3. It separately checks the initial per-thread `TrackSegment` array byte count.
4. Both padded arrays are zeroed with the checked `thread_slots_bytes` value.
5. `spectral_tracker_emit_segment()` checks the doubled segment capacity and the
   resulting allocation/copy byte counts before reallocating a thread segment
   array.
6. Any arithmetic failure is classified as overflow, not memory pressure.

## Why this is critical

The tracker emits segments from the hot analysis path. Underallocating
per-thread segment storage is not a recoverable DSP approximation; it is a
memory safety violation in the kernel.
