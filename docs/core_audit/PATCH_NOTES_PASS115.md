# Core audit pass 115: tracker flush API encapsulation

## Summary

Pass 115 removes the long candidate-flush declaration from the internal header.

`Spectral_tracker_flush_candidate_batch()` is an implementation detail of
`spectral_peak_track.c`; no other module should call it. Keeping its long
parameter list in the internal header makes the argument-sprawl look like a
supported API.

## Fix

The function is made `static` inside `spectral_peak_track.c` and its declaration
is removed from `spectral_peak_track_internal.h`.

## Why this is critical

Phase E is about reducing candidate-flow API surface. Hiding the flush helper is
a low-risk cleanup that prevents future modules from depending on the old long
argument chain.
