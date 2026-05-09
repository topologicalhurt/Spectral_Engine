# Core audit pass 85: tracker candidate best-next integer-domain contract

## Summary

Pass 85 hardens candidate validation before it narrows frequency-bin indices
into the estimator's signed `best_next_bin` field.

## Bug

`Spectral_tracker_validate_candidate()` computes:

```c
*out_best_next = (int)cf + best_idx - 1;
```

where `cf` is `size_t`. The tracker constructor now rejects huge frequency
counts, but this helper is an internal boundary and should still prove the
specific candidate index is representable before narrowing.

## Fix

Candidate validation now rejects:

```text
cf > INT_MAX
```

before computing `best_next_bin`.

`spectral_peak_interp.c` now includes `<limits.h>` explicitly for the integer
domain proof.

## Reviewer Walkthrough

1. Existing pointer, threshold and zero-bin checks remain.
2. The candidate index must fit the signed estimator field domain.
3. Neighbor magnitudes are then loaded and validated as before.
4. `best_next_bin` is computed only after the representability proof.

## Why this is critical

The estimator API uses an `int` for `best_next_bin`. Every path that narrows a
frequency-bin index into that field must prove the index domain first.
