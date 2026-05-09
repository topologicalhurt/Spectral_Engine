# Core audit pass 98: full-analysis STFT matrix ownership

## Summary

Pass 98 introduces a reusable owner for full-matrix STFT buffers.

The full path previously managed `magsq`, `phases`, `total_bins`, and
`total_bytes` independently. That ownership pattern is reusable and should not be
encoded as raw pointer/free choreography inside the analysis function.

## Fix

Pass 98 introduces:

```c
SpectralAnalysisStftMatrix
spectral_analysis_stft_matrix_alloc()
spectral_analysis_stft_matrix_free()
```

`Spectral_analysis_run_full()` now owns the matrix through this struct.

## Why this is critical

Full-matrix analysis is the simpler reference path. Its memory ownership should
be obvious and reusable before further full/fused parity work.
