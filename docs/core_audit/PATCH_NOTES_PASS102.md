# Core audit pass 102: peak clamp consolidation regression repair

## Summary

Pass 102 repairs the double-domain offset clamp helper introduced during peak
estimator consolidation.

## Bug

The intended helper is the single policy for:

```text
finite double offset
clamp to [-0.5, 0.5]
narrow to float
store
```

A bad consolidation can leave the helper recursively calling itself:

```c
return spectral_peak_store_clamped_offset_d(out_offset, offset_d);
```

inside its own body. That turns every estimator using the helper into infinite
recursion.

## Fix

The helper body is replaced with the actual clamp/narrow/store implementation.
The estimators continue calling the shared helper.

## Why this is critical

This is a real regression, not a stylistic cleanup. Consolidation helpers must be
audited as carefully as the code they replace.
