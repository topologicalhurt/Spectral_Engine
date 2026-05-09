# Core audit pass 97: FFT frame dispatch consolidation

## Summary

Pass 97 removes duplicate `spectral_fft_frames()` implementations.

The vDSP and FFTW branches need different single-frame kernels, but the frame
range validation and OpenMP dispatch loop are identical. Keeping two copies makes
future scheduler/range changes easy to apply to only one backend.

## Fix

The two backend-local `spectral_fft_frames()` definitions are removed and a
single backend-agnostic implementation is placed after the vDSP/FFTW
`spectral_fft_single_frame()` definitions.

## Why this is critical

Backend-specific code should contain backend-specific work only. Dispatch policy
is shared architecture and should have one owner.
