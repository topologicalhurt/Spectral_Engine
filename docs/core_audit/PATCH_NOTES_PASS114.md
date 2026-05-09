# Core audit pass 114: tracker worker-stats context

## Summary

Pass 114 groups per-worker tracker counters/timers into one reusable context.

The fused path accumulated:

```text
local_pairs
local_candidates
local_segments
local_track_time
debug timing fields...
```

and then passed a long argument list into `spectral_tracker_accumulate_stats()`.

## Fix

Adds:

```c
SpectralTrackerWorkerStats
spectral_tracker_worker_stats_commit()
```

The fused worker now commits through that context object.

## Why this is critical

Tracker candidate flow is no longer just rows and batch state. Worker-local
stats are another repeated argument group. Giving them a named owner makes the
hot path easier to refactor without losing timing/accounting behavior.
