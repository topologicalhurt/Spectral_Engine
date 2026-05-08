# Core audit pass 80: fast sqrt finite-input contract

## Summary

Pass 80 hardens the public fast square-root helpers.

The engine keeps approximation gates disabled by default, but the helpers remain
public kernel utilities. They should not return NaN for invalid input when their
documented failure behavior for non-positive values is zero.

## Bug

`fast_inv_sqrt()` and `fast_sqrt()` checked only:

```c
x <= 0.0f
```

For `NaN`, that comparison is false, so the helpers returned NaN through
`sqrtf(NaN)` or approximation arithmetic. For `Inf`, `fast_sqrt()` returned Inf.

Several estimator paths defensively validate inputs, but public helpers should
still define their own finite-input contract.

## Fix

Both helpers now reject:

```text
NaN
Inf
non-positive values
```

with `0.0f`.

Approximation branches also validate their result before returning it.

## Reviewer Walkthrough

1. The first branch uses `!(x > 0.0f) || !isfinite(x)` to reject NaN, Inf and
   non-positive values.
2. Exact paths remain exact for valid finite positive inputs.
3. Approximation paths check the computed result is finite and positive.
4. Callers that already validate inputs see no behavior change.

## Why this is critical

These helpers are low-level DSP utilities. Returning NaN/Inf from public math
helpers makes invalid input harder to contain and can poison downstream vectors
or segment estimates.
