# Core audit pass 59: vector complex-interleaved index contract

## Summary

Pass 59 hardens vector helpers that operate on interleaved complex arrays.

Several vector helpers accept a logical complex count but index the source array
as:

```c
interleaved[i * 2]
interleaved[i * 2 + 1]
```

## Bug

The helpers did not prove that `count * 2` is representable before using those
indices. High-level FFT callers allocate checked buffers, but the vector helpers
are public core utilities and must defend their own index domain.

Affected helpers:

```text
spectral_deinterleave
spectral_magsq_phase
spectral_magsq_only
```

## Fix

A shared helper now validates:

```text
count <= SIZE_MAX / 2
```

before any complex-interleaved indexing. The magnitude/phase helpers also clear
`*max_magsq` on invalid/empty input so callers do not observe stale output.

## Reviewer Walkthrough

1. `spectral_complex_interleaved_count_valid()` defines the index-domain proof.
2. `spectral_deinterleave()` returns before indexing if the count is invalid.
3. `spectral_magsq_phase()` clears `max_magsq`, validates pointers, and rejects
   invalid complex counts before indexing.
4. `spectral_magsq_only()` does the same for magnitude-only scans.

## Why this is critical

Complex interleaved arrays are twice the logical bin count. A helper that accepts
logical counts must prove the `2 * count` index domain itself, not rely on every
caller to have done it.
