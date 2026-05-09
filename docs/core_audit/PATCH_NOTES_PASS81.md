# Core audit pass 81: tracker emitted-segment validity contract

## Summary

Pass 81 hardens the final tracker emission boundary.

The estimator already validates a large amount of local peak state, but
`spectral_tracker_emit_segment()` still wrote a `TrackSegment` directly from
derived fields without one final emitted-segment contract check.

## Bug

The emit path assigned:

```c
seg->start = t_hop;
seg->length = hop_float;
seg->phase = phase_row[cf];
seg->amp = estimate.amp;
seg->da = estimate.da;
seg->omega = estimate.omega;
seg->df = estimate.df;
```

without proving the complete emitted segment would satisfy the synthesis/cache
payload contract:

```text
finite non-negative start
finite positive length
finite phase
finite non-negative amplitude
finite non-negative omega
finite df/da
valid per-thread storage domain
```

A single invalid phase row or malformed estimator output could therefore enter
the segment array even though later synthesis now assumes valid segment payloads.

## Fix

`Spectral_tracker_emit_segment()` now checks the final emitted fields through:

```c
spectral_tracker_emitted_segment_valid(...)
```

before touching the per-thread segment array.

Invalid emitted segment state records `SPECTRAL_ERR_PARAM` through the tracker
first-error channel and fails the emission path.

## Reviewer Walkthrough

1. The estimator still computes `SpectralPeakEstimate` as before.
2. Before allocating/growing segment storage, the emitter validates the thread
   domain and storage pointers.
3. It validates all fields that will be written into `TrackSegment`.
4. Only then does it read the per-thread segment count and write the segment.
5. Failure preserves the tracker root-cause error rather than emitting corrupt
   segment state.

## Why this is critical

The tracker is the only producer of analysis segments. Segment validity must be
proven at the producer boundary, not merely at later cache/synthesis consumers.
