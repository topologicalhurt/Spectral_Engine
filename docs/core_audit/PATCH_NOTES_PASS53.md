# Core audit pass 53: audio window time-domain contract

## Summary

Pass 53 hardens audio window extraction.

The audio read/write passes validated sample and frame-count contracts. The
windowing helper still accepted some invalid time-domain inputs, especially
`NaN` start times.

## Bug

The old window logic treated `NaN` start times as zero because:

```c
(start_sec > 0.0f) ? ... : 0.0
```

is false for `NaN`.

That silently converted invalid caller state into "start at zero". The helper
also accepted any positive sample rate rather than the engine's configured
sample-rate domain, and it could leave output pointers unchanged on parameter
failure.

## Fix

`Spectral_audio_window()` now:

```text
clears outputs before validation
requires total_frames > 0
validates sample_rate in configured bounds
requires finite start_sec
requires end_sec either negative sentinel or finite
keeps existing clamp/window behavior for valid inputs
```

## Reviewer Walkthrough

1. Output pointers are nulled/zeroed once pointer parameters are valid.
2. Sample rate is checked against canonical bounds.
3. `start_sec` must be finite; it is no longer silently interpreted as zero when
   it is NaN.
4. `end_sec < 0` remains the "to end of buffer" sentinel.
5. Non-negative `end_sec` must be finite.
6. The existing frame clamp and non-empty window checks remain.

## Why this is critical

Window extraction defines the analysis interval. Invalid time-domain parameters
must fail closed; silently converting NaN into a real frame offset can analyze
the wrong audio region while looking successful.
