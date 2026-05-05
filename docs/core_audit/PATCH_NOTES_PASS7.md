# Core audit pass 7: FFT allocator boundary safety

## Intent

Close the remaining FFT resource ownership gap left after pass 6.

## Changes

- `spectral_fft_resources_alloc()` now zeroes the resource state before any shape-failure return.
- `spectral_fft_resources_alloc()` now rejects FFT sizes larger than `INT_MAX`, because the FFTW planner API takes an `int` length.
- The full-matrix analysis path initializes `SpectralFftResources res = {0};` before allocation attempts.
- Static audit checks cover the new allocator contract.

## Rationale

Pass 6 made cleanup safe for partial allocations, but the allocator still returned before zeroing `res` on invalid shape. Existing callers can call `spectral_fft_resources_free(&res)` after allocation failure, so the failure path must leave `res` in a known zero state.
