# Core audit pass 70: analysis OpenMP thread-count domain contract

## Summary

Pass 70 bounds analysis thread counts before allocating per-thread resources.

The analysis paths used `omp_get_max_threads()` directly. That value can be
controlled by process/global OpenMP configuration and should not be allowed to
drive unbounded per-thread FFT/tracker allocations.

## Bug

Both full and fused analysis derive resource counts from:

```c
omp_get_max_threads()
```

The engine already has a canonical bound:

```c
SPECTRAL_MAX_THREADS
```

but analysis resource allocation was not enforcing it.

An extreme OpenMP environment could request excessive per-thread FFT scratch,
tracker segment arrays, and fused row buffers before the engine applies any
kernel-specific limit.

## Fix

Analysis now uses one helper:

```c
spectral_analysis_effective_thread_count()
```

which clamps:

```text
< 1 -> 1
> SPECTRAL_MAX_THREADS -> SPECTRAL_MAX_THREADS
```

Full and fused analysis both use this helper before allocating FFT resources or
trackers.

## Reviewer Walkthrough

1. The helper lives in the shared analysis module.
2. The internal analysis header exposes it to full/fused implementations.
3. Full-matrix analysis uses the bounded count for FFT resources.
4. Fused analysis uses the bounded count for FFT resources, tracker creation and
   OpenMP region team sizes.
5. Backend-specific thread binding from Pass 67 remains local and explicit.

## Why this is critical

Thread count is an allocation shape. It must be bounded by the engine's
configured domain before allocating per-thread kernel resources.
