# Core audit pass 68: fused-analysis frame-time float contract

## Summary

Pass 68 hardens fused-analysis frame-time conversion.

The tracker's per-frame context stores `t_hop` as `float`, while the fused path
derives it from:

```c
pair * hop
```

where `pair` is `size_t` and `hop` is `int`.

## Bug

The old fused path used:

```c
float t_hop = (float)pair * (float)hop;
```

without proving the product was finite and representable as `float`.

For pathological but representable host-side frame indices, this can produce
Inf or a silently invalid tracker time coordinate. That value enters segment
start-time derivation.

## Fix

The fused path now computes the product in `double`, checks it is finite,
non-negative and `<= FLT_MAX`, then narrows to `float`.

On failure, it records `SPECTRAL_ERR_OVERFLOW` through the tracker first-error
setter and stops processing.

## Reviewer Walkthrough

1. `spectral_analysis_fused.c` includes the tracker internal setter and
   `<float.h>`.
2. Each pair computes `t_hop_d = (double)pair * (double)hop`.
3. Invalid or unrepresentable frame time records overflow.
4. `t_hop` is assigned only after the representability proof.
5. The rest of the fused frame context remains unchanged.

## Why this is critical

The tracker emits segment start positions in sample units. A fused path must not
convert a host-side frame index into an invalid float time coordinate and then
treat the resulting segment as valid analysis output.
