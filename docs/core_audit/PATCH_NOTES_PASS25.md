# Core audit pass 25: tracker error propagation preserves root cause

## Summary

Pass 25 fixes a tracker error-contract violation: later generic failure paths
could overwrite the first, more precise tracker error.

The concrete failure mode is:

```text
segment growth detects capacity/byte-count overflow
emit path stores SPECTRAL_ERR_OVERFLOW
batch/fused caller sees failure and stores SPECTRAL_ERR_MEMORY
final tracker state reports memory failure instead of overflow
```

That collapses a caller-contract bug into a resource-pressure bug.

## Bug

Before this pass, several paths wrote directly to `tracker->last_error`:

```c
atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_OVERFLOW, ...);
atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_MEMORY, ...);
```

That means the final error depended on which layer observed failure last, not
which layer discovered the root cause.

## Fix

The tracker now exposes one internal error setter:

```c
spectral_tracker_set_error(tracker, error)
```

It commits only the first non-`SPECTRAL_OK` error using an atomic compare/exchange
from `SPECTRAL_OK` to `error`.

`Spectral_tracker_set_failed()` remains the compatibility helper for external
allocation failures, but it now calls:

```c
spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY)
```

instead of overwriting the field.

The emit path and batch-flush path now use the same setter.

## Reviewer Walkthrough

1. `spectral_tracker_set_error()` rejects `SPECTRAL_OK`; success is not an error
   event.
2. It uses `atomic_compare_exchange_strong_explicit()` with
   `expected = SPECTRAL_OK`, so only the first failing layer records the error.
3. Segment growth overflow records `SPECTRAL_ERR_OVERFLOW`.
4. If an outer batch/fused path subsequently observes failure and tries to set
   `SPECTRAL_ERR_MEMORY`, the compare/exchange fails because the tracker already
   contains `SPECTRAL_ERR_OVERFLOW`.
5. If the first failure really is allocation failure, `SPECTRAL_ERR_MEMORY` is
   still recorded.

## Why this is critical

Overflow and memory exhaustion are different kernel contracts. Overflow means
the requested shape or growth is not representable. Memory failure means the
shape is representable but allocation failed. Collapsing those into whichever
error was written last makes diagnostics and caller recovery incorrect.
