# Core audit pass 82: tracker stats accumulation overflow contract

## Summary

Pass 82 hardens tracker statistics accumulation.

Tracker statistics are used for bandwidth/timing diagnostics and byte-estimate
reporting. They are accumulated from worker-local counters into shared tracker
fields.

## Bug

The old accumulation path used OpenMP atomic additions directly:

```c
tracker->total_pairs += local_pairs;
tracker->total_candidates += local_candidates;
tracker->total_segments += local_segments;
tracker->process_time_total += local_track_time;
```

Those additions had no overflow or finiteness checks. A wrap in `total_segments`
can later corrupt byte-estimate calculations, and non-finite timing can poison
analysis summaries.

## Fix

Stats accumulation now:

```text
rejects non-finite/negative local timing
checks uint64_t additions before mutating shared counters
checks double timing additions before mutating shared timing
records overflow/parameter errors through the tracker first-error channel
```

Because stats are accumulated once per worker, the code uses a small OpenMP
critical section rather than unchecked atomic increments.

## Reviewer Walkthrough

1. `spectral_tracker_u64_add_checked()` protects counter additions.
2. Local timing values must be finite and non-negative.
3. Shared updates happen inside one named critical section.
4. Each counter is checked before assignment.
5. Timing sums are checked before assignment.
6. Debug timing fields use the same contract when enabled.

## Why this is critical

Diagnostic counters still influence byte estimates and final reporting. They
must not wrap or become non-finite merely because the hot path processed a very
large input.
