# Core audit pass 74: peak amplitude gain-bound finite-product contract

## Summary

Pass 74 hardens the peak-amplitude gain bound.

Window-aware peak-height callbacks can estimate a peak magnitude-squared value
above the center-bin magnitude. The estimator bounds that gain with
`SPECTRAL_PEAK_AMP_MAX_GAIN`.

## Bug

The bound used float multiplication:

```c
peak_magsq <= center_magsq * max_gain
```

If `center_magsq * max_gain` overflowed to Inf, every finite candidate
`peak_magsq` would pass the bound. That turns a safety bound into an always-true
predicate for large finite inputs.

## Fix

The gain limit is now computed in `double` and checked for finiteness before the
candidate is accepted:

```c
gain_limit = (double)center_magsq * (double)max_gain
```

The candidate comparison is also performed in `double`.

## Reviewer Walkthrough

1. Center and candidate magnitudes must still be finite and positive.
2. Invalid configured gain falls back to 1.0.
3. The gain limit is computed in double.
4. A non-finite or non-positive limit rejects the candidate.
5. Only then is `peak_magsq` compared against the checked limit.

## Why this is critical

Amplitude estimation is part of the emitted segment contract. A gain limiter
must not become ineffective because its own limit arithmetic overflowed.
