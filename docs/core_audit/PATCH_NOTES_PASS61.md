# Core audit pass 61: FFT frame dispatch range/thread contract

## Summary

Pass 61 hardens shared FFT frame dispatch.

`Spectral_fft_resources_alloc()` allocates per-thread FFT buffers for a fixed
`res->n_threads`. `spectral_fft_frames()` then used `omp_get_thread_num()` to
index those arrays, but did not bind the OpenMP team size to the resource count.

## Bug

If the process-wide OpenMP thread count changes after resource allocation,
`omp_get_thread_num()` can exceed the number of allocated FFT scratch slots.

The function also derived:

```c
local_n_frames = frame_end - frame_start
```

without first proving `frame_end >= frame_start`, and it did not validate the
phase-output pointer when phase output was requested.

## Fix

Both vDSP and FFTW frame-dispatch implementations now validate:

```text
resource/audio/window/output pointers
positive hop
frame_end >= frame_start
nonzero frame span
phase output pointer when magsq_only == 0
local_n_frames * n_freqs representability
```

The OpenMP parallel region now uses:

```c
num_threads(res->n_threads)
```

so the thread ID domain matches the allocated resource arrays.

## Reviewer Walkthrough

1. Invalid resource or pointer state returns a zero max.
2. Reversed frame ranges fail closed instead of underflowing.
3. The output matrix shape is checked before pointer arithmetic.
4. OpenMP team size is bound to the resource allocation count.
5. Existing per-frame transform logic remains unchanged.

## Why this is critical

FFT scratch buffers are per-thread resources. The dispatch thread domain must
match the allocation thread domain, or a correct-looking analysis request can
index past the FFT resource arrays.
