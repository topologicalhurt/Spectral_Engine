# Core audit pass 83: tracker creation thread/frequency index-domain contract

## Summary

Pass 83 hardens `SpectralTracker` creation domains.

Analysis dispatch now clamps OpenMP thread count, but `SpectralTracker` is still
an internal/public incremental API. Its constructor must defend its own resource
and index domains.

## Bug

`Spectral_tracker_create()` accepted any positive `n_threads` and any
`n_freqs` that passed coarse shape checks. The tracker later stores candidate
bin IDs as `uint32_t` and passes `best_next_bin` as `int` into the estimator.

That means the constructor must reject frequency counts outside the estimator's
integer index domain. It also must reject thread counts outside the configured
engine thread domain, not merely depend on callers to clamp.

## Fix

Tracker scalar derivation now rejects:

```text
n_freqs > INT_MAX
```

and the constructor rejects:

```text
n_threads > SPECTRAL_MAX_THREADS
```

before allocating per-thread segment storage.

## Reviewer Walkthrough

1. `n_freqs` is validated in the same helper that derives tracker scalars.
2. Frequency bin counts now fit the estimator's signed `best_next_bin` domain.
3. Thread counts below one are normalized to one as before.
4. Thread counts above `SPECTRAL_MAX_THREADS` fail closed.
5. Allocation shape and OpenMP team shape remain aligned.

## Why this is critical

Tracker creation defines the integer domains used throughout peak scanning,
candidate batching and estimator dispatch. Those domains need to be explicit at
construction time.
