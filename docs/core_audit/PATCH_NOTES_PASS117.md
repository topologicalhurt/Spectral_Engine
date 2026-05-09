# Core audit pass 117: analysis path policy object

Pass 117 starts Phase F by making full/fused path selection explicit and reusable.

The analysis path decision was previously embedded as `total_bins > SPECTRAL_STFT_CHUNK_THRESHOLD`.
This pass introduces:

```c
SpectralAnalysisPathMode
SpectralAnalysisPathDecision
spectral_analysis_path_decide()
```

`SpectralAnalysisShape` now stores the decision object. This creates the reusable seam needed for controlled full/fused parity tests.
