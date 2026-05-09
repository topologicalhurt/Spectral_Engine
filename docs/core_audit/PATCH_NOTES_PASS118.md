# Core audit pass 118: explicit analysis path-mode entry point

Pass 118 adds the controlled entry point needed for full/fused behavioral parity tests:

```c
analyze_audio_with_path_mode(..., SPECTRAL_ANALYSIS_PATH_FULL, ...)
analyze_audio_with_path_mode(..., SPECTRAL_ANALYSIS_PATH_FUSED, ...)
```

The existing `analyze_audio()` delegates to AUTO mode.
