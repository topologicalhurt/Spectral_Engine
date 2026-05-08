# Core audit pass 77: Quinn second estimator finite-product contract

## Summary

Pass 77 hardens Quinn's second frequency estimator.

Quinn's estimator is useful but algebraically dense: it forms dot products,
rational corrections, squares of intermediate deltas, and tau-log corrections.
The old implementation performed those products in `float`.

## Bug

Finite reconstructed bins can still overflow float dot products:

```c
ap = dot(xp, x0) / |x0|^2
am = dot(xm, x0) / |x0|^2
```

The correction terms then square `dp` and `dm` before passing them to tau. If
those products overflow, the estimator can report an invalid offset or fall back
for the wrong reason.

## Fix

The Quinn path now computes dot products, denominators, `dp`, `dm`, squared
terms, and the final offset in `double`.

It checks:

```text
dot product ratios finite
denominators nonzero
dp/dm finite
dp²/dm² finite and representable as float before tau
tau outputs finite
final offset finite
```

The final offset is clamped in double before narrowing.

## Reviewer Walkthrough

1. Complex triplet reconstruction remains unchanged.
2. `mag0`, `ap`, `am`, `den_p`, `den_m`, `dp`, `dm` use double.
3. Squared correction inputs must fit the float-domain tau helper.
4. Tau outputs must be finite.
5. The final offset is computed and clamped in double.

## Why this is critical

Quinn's estimator is an explicit high-accuracy policy. Its correctness is
dominated by its algebra, so the intermediate products must be protected against
overflow and invalid narrowing.
