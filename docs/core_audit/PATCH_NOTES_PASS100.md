# Core audit pass 100: peak offset clamp consolidation

## Summary

Pass 100 centralizes double-domain peak offset clamping.

Jacobsen, magnitude-parabolic and Quinn estimators all used the same pattern:

```text
check finite double offset
clamp to [-0.5, 0.5]
narrow to float
store
```

## Fix

Pass 100 adds:

```c
spectral_peak_store_clamped_offset_d()
```

and routes double-domain estimators through it.

## Why this is critical

The estimator family should have one narrowing/clamp policy. Repeating it in
every estimator makes future offset-domain changes fragile.
