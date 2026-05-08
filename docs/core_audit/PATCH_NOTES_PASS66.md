# Core audit pass 66: fused-analysis scratch allocation byte-count contract

## Summary

Pass 66 hardens the fused STFT+tracking scratch allocation boundary.

The fused path allocates per-thread FFT row buffers for `n_freqs` floats. The
full-matrix path already derives allocation byte counts with checked helpers,
but the fused path still used raw products:

```c
spectral_aligned_alloc(n_freqs * sizeof(float))
```

## Bug

`n_freqs` is a `size_t`, and these scratch buffers are written by
`spectral_fft_single_frame()`. If `n_freqs * sizeof(float)` wrapped, the fused
path could allocate a smaller buffer than the FFT helper writes into.

The allocation occurs inside OpenMP regions, so the failure path also has to
fail closed through the tracker rather than producing partial output.

## Fix

The fused path now computes:

```c
n_freqs_f32_bytes
```

once with `spectral_array_bytes(n_freqs, sizeof(float), ...)` before allocating
any fused scratch row.

Every per-thread row allocation uses that checked byte count:

```text
pass-1 scratch magsq row
current magsq row
next magsq row
current phase row
next phase row
```

## Reviewer Walkthrough

1. `n_fft_f32_bytes` remains the checked window byte count.
2. `n_freqs_f32_bytes` is derived once before any FFT resource allocation.
3. Pass-1 max-discovery scratch rows use the checked value.
4. Pair-processing row/phase buffers use the checked value.
5. No fused scratch allocation uses `n_freqs * sizeof(float)` anymore.

## Why this is critical

The fused path is the large-input kernel. Its per-thread scratch rows are direct
FFT outputs. Their allocation byte count must be proven before the FFT helper
writes into them.
