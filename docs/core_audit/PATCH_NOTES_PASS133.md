# Patch notes — Pass 133: ARM32 chirp contract

## Problem

`SpectralSegmentQ15` can carry `df_q15` when `SPECTRAL_HAS_CHIRP` is enabled, but the current ARM32 hot path advances phase with a constant `freq_inc`. That silently renders chirped segments as non-chirped segments.

## Change

Add `spectral_arm32_segment_chirp_supported()` and make `spectral_arm32_load()` reject nonzero `df_q15` until the ARM32 process loop has a real, tested chirp implementation.

## Why minimal

This does not attempt a performance implementation. It closes the logical contract hole first: unsupported DSP features must fail closed, not degrade silently.
