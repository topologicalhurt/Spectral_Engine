# Core audit pass 95: tracker peak-model mutation consolidation

## Summary

Pass 95 consolidates tracker peak-model setter wiring.

Four public setters repeated the same pattern:

```text
load current model
mutate one field
resolve/apply full model
ignore error in void public setter
```

That duplication makes it easy for future estimator/window/phase policy changes
to update one setter but miss another.

## Fix

Pass 95 introduces one internal mutation helper:

```c
spectral_tracker_update_peak_model_field(...)
```

All four public setters now route through that helper.

## Why this is critical

Peak-model resolution is a coupled contract: window, estimator, phase policy and
amplitude policy are not independent after resolution. Mutations should pass
through one full-model resolver.
