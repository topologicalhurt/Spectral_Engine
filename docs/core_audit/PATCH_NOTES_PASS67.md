# Core audit pass 67: fused-analysis OpenMP resource-thread contract

## Summary

Pass 67 binds fused-analysis OpenMP teams to the allocated FFT resource thread
domain.

Pass 61 fixed `spectral_fft_frames()` by using `num_threads(res->n_threads)`.
The fused path still relied on process-global `omp_set_num_threads()` before
parallel regions.

## Bug

The fused path allocates FFT resources for `actual_threads`. Inside each OpenMP
region, `omp_get_thread_num()` indexes those per-thread resources.

Using `omp_set_num_threads(actual_threads)` mutates global OpenMP state and
relies on later parallel regions inheriting it correctly. The correctness
contract should be local to the parallel region.

## Fix

Both fused parallel regions now use explicit OpenMP clauses:

```c
#pragma omp parallel reduction(max:pass1_max) num_threads(actual_threads)
#pragma omp parallel num_threads(actual_threads)
```

The global `omp_set_num_threads()` calls are removed.

## Reviewer Walkthrough

1. `actual_threads` remains the allocation count for FFT resources.
2. Pass-1 max discovery binds its team size to `actual_threads`.
3. Pair-processing binds its team size to `actual_threads`.
4. `omp_get_thread_num()` is now local-region bounded by the resource allocation.
5. The pass stops mutating global OpenMP thread state.

## Why this is critical

Per-thread FFT scratch arrays are indexed by OpenMP thread ID. The parallel
region itself must prove the thread-ID domain matches the allocation domain.
