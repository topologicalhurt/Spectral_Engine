# Core audit pass 101: tracker stats accumulation helper consolidation

## Summary

Pass 101 consolidates tracker statistics accumulation.

Pass 82 made the stats path safe, but left repeated blocks for every counter and
timing field. That is readable enough for three counters, but debug timing adds a
large repeated section.

## Fix

Pass 101 introduces:

```c
spectral_tracker_accumulate_counter_checked()
spectral_tracker_accumulate_time_checked()
```

`Spectral_tracker_accumulate_stats()` now calls those helpers for each field.

## Why this is critical

Stats are diagnostic, but their overflow/error policy should have one reusable
implementation. This pass preserves the safety from Pass 82 while making the
code maintainable.
