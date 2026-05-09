# Core audit pass 103: peak magnitude triplet helper consolidation

## Summary

Pass 103 removes duplicated left/center/right magnitude-squared neighborhood
loading and validation from the peak estimator.

## Problem

Multiple estimator paths loaded the same local triplet shape:

```text
row[bin - 1]
row[bin]
row[bin + 1]
```

and repeated finite/non-negative checks. This made callback-boundary validation
and estimator-specific logic harder to distinguish.

## Fix

Pass 103 introduces:

```c
SpectralPeakMagsqTriplet
spectral_peak_load_magsq_triplet()
```

and routes these paths through it:

```text
log-parabolic offset
magnitude-parabolic offset
next-frame offset
current-frame peak-height callback
next-frame peak-height callback
```

## Why this is critical

The three-bin neighborhood is a reusable estimator primitive. Centralizing it
keeps extension-point callback dispatch clean and makes future estimator policies
less error-prone.
