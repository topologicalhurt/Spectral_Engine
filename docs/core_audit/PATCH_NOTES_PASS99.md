# Core audit pass 99: tracker frame-time helper consolidation

## Summary

Pass 99 centralizes frame-index-to-sample-time conversion.

Both fused analysis and incremental tracker processing convert frame/pair indices
into `float t_hop`. They had duplicate double-domain finite checks.

## Fix

Pass 99 introduces:

```c
spectral_tracker_frame_time_from_index()
```

in `spectral_peak_track_internal.h`.

Both fused analysis and tracker processing use the helper.

## Why this is critical

Frame time is segment start state. Full/fused/incremental paths should not carry
separate conversion logic for the same tracker time coordinate.
