# Core audit pass 93: analysis window context consolidation

## Summary

Pass 93 removes duplicated analysis-window setup from the full and fused analysis
paths.

Both paths independently performed the same sequence:

```text
compute n_fft * sizeof(float)
allocate window buffer
generate Hann
derive window metrics
apply endpoint/positive-bin FFT scales
bind Hann descriptor to tracker
free window buffer
```

That is a wiring smell. The window is part of the analysis contract; setup should
have one owner.

## Fix

Pass 93 introduces:

```c
SpectralAnalysisWindowContext
spectral_analysis_window_context_init()
spectral_analysis_window_context_free()
spectral_analysis_window_context_apply_magsq_scales()
```

Full and fused analysis now use the shared context.

## Why this is critical

Window generation, metric calibration and tracker descriptor binding are a
single contract. If one path changes and the other does not, full/fused parity
breaks. Centralizing this setup makes future window extensibility less fragile.
