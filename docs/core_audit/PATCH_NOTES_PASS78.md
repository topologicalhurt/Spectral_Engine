# Core audit pass 78: magnitude-parabolic estimator finite-product contract

## Summary

Pass 78 hardens the magnitude-domain parabolic peak estimator.

This estimator is an explicit diagnostic policy. It takes square roots of the
local magnitude-squared triplet and fits a quadratic in magnitude space.

## Bug

The old implementation computed the quadratic denominator and offset in
`float`:

```c
denom = left - 2.0f * center + right;
offset = 0.5f * (left - right) / denom;
```

For large finite magnitudes, the subtraction chain can lose the finite-product
contract before the final offset clamp.

## Fix

The estimator now:

```text
checks sqrt outputs are finite
computes numerator and denominator in double
checks denominator finiteness and floor
computes offset in double
clamps before narrowing to float
```

## Reviewer Walkthrough

1. Input magnitude-squared values remain finite/nonnegative-checked.
2. Sqrt outputs are checked before fitting.
3. The quadratic denominator uses double arithmetic.
4. The numerator and offset use double arithmetic.
5. The final offset is clamped before float narrowing.

## Why this is critical

Even diagnostic estimator policies should be numerically safe. A large finite
triplet should not overflow estimator algebra before the clamp stage.
