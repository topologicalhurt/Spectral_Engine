# Core audit pass 96: shared OpenMP effective-thread contract

## Summary

Pass 96 moves effective OpenMP thread-count clamping into the OpenMP shim.

Analysis had its own clamp helper:

```c
spectral_analysis_effective_thread_count()
```

while GPU tile preprocessing still read `omp_get_max_threads()` directly. That
split the thread-domain contract across modules.

## Fix

Pass 96 adds:

```c
spectral_omp_effective_thread_count()
```

to `spectral_omp.h`.

Analysis and GPU tile preprocessing now use the same helper. The analysis-local
wrapper is removed.

## Why this is critical

Thread count is an allocation shape. Every subsystem allocating per-thread
scratch must apply the same domain clamp before allocation and OpenMP dispatch.
