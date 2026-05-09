# Core audit pass 84: tracker byte-estimate STFT accounting overflow contract

## Summary

Pass 84 fixes an overflow-check omission in tracker byte-estimate accounting.

`Spectral_tracker_estimate_bytes()` estimates memory traffic for analysis
logging. It already uses checked arithmetic for most terms, but the fused STFT
read/write contribution had one unchecked addition result.

## Bug

The code computed:

```c
if (spectral_size_mul(stft_writes, 2u, &fft_mags_phases)) {
    spectral_size_add(total_read_floats, fft_mags_phases, &total_read_floats);
}
```

The return value from `spectral_size_add()` was ignored. If that addition
overflowed, the estimate silently kept stale `total_read_floats` and continued
as if the estimate were valid.

## Fix

The STFT accounting block now returns failure if either multiplication or
addition fails.

## Reviewer Walkthrough

1. `pairs * n_freqs` remains checked.
2. `stft_writes * 2` remains checked.
3. The addition into `total_read_floats` is now checked.
4. Estimate failure propagates through the existing caller path.

## Why this is critical

Even diagnostic estimates must not silently wrap. These byte counts are used to
reason about hot-path bandwidth and should fail closed when outside the
representable domain.
