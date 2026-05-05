# Core audit pass 14: sub-bin temporal slope for `df`

## Intuition

Pass 11 made the current-frame frequency estimate sub-bin accurate:

```text
omega = (current_bin + current_offset) * bin_step
```

But the temporal slope still used only coarse bin movement:

```text
df ~= best_next_bin - current_bin
```

That means a stationary off-bin tone can have a good `omega` but a weak
frequency-slope model.

## Change

Pass 14 adds optional next-frame sub-bin estimation:

```text
current peak: current_bin + current_offset
next peak:    best_next_bin + next_offset
df:           (next_peak - current_peak) * freq_step_df
```

If the next-frame estimate is not safe, the code preserves the previous coarse
fallback:

```text
df = (best_next_bin - current_bin) * freq_step_df
```

## Why the fallback matters

The next-frame best bin is selected from a local search around the current bin.
It is not guaranteed to be a local maximum in its own expanded next-frame
neighborhood. Pass 14 requires that before using a next-frame offset. Otherwise
it keeps the old coarse model rather than inventing precision from bad data.

## What this does not solve yet

This is still magnitude-track slope, not phase-vocoder phase-continuity. The
next major DSP pass should use expected phase advance / phase unwrapping to
check or refine instantaneous frequency.
