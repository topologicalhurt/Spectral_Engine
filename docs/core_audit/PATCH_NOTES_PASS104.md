# Core audit pass 104: analysis shape contract consolidation

## Summary

Pass 104 centralizes analysis input/shape derivation.

`Analyze_audio()` mixed public input validation, FFT-shape derivation, total-bin
overflow checks, path selection and path-name formatting in one function.

## Fix

Pass 104 introduces:

```c
SpectralAnalysisShape
spectral_analysis_shape_init()
spectral_analysis_path_name()
```

`Analyze_audio()` now delegates shape derivation to the helper and only performs
dispatch/reporting.

## Why this is critical

Analysis shape is a reusable contract. Future streaming/window/path changes
should not duplicate sample-rate, FFT, frame-count and total-bin logic.
