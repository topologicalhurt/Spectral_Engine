# Core audit pass 69: FFT single-frame helper boundary contract

## Summary

Pass 69 hardens `spectral_fft_single_frame()`.

The caller paths now bind thread domains correctly, but the helper itself is an
internal shared boundary and still dereferenced `res`, per-thread arrays and FFT
plans before validating its arguments.

## Bug

Both vDSP and FFTW definitions started by reading:

```c
res->n_fft
res->thread_in[tid]
res->fft_plans[tid]
```

or their vDSP equivalents. If a future internal caller passed an invalid `tid`,
uninitialized resources, null audio/window/output pointers, or invalid hop, the
helper could crash or write through invalid scratch state.

## Fix

A shared argument validator now checks:

```text
resource pointer
thread ID inside allocated resource domain
audio/window/output pointers
positive hop
nonzero FFT shape
backend-specific per-thread scratch/plan pointers
```

Invalid calls zero the output row/phase row when possible, set frame max to zero,
and return without touching backend FFT state.

## Reviewer Walkthrough

1. `spectral_fft_single_frame_args_valid()` owns the resource/tid checks.
2. `spectral_fft_single_frame_clear()` defines fail-closed output behavior.
3. Both vDSP and FFTW implementations guard before reading `res->n_fft`.
4. Valid hot paths execute the same FFT logic as before.

## Why this is critical

This function is the boundary between frame dispatch and backend FFT resources.
It should not rely solely on every caller proving the same resource/tid contract.
