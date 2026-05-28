# Patch notes — Pass 135: ARM32 load invariants

## Problem

The streaming ARM32 process loop assumes segment order and bounded active voice count, while `spectral_arm32_load()` accepted arbitrary segment arrays. `spectral_arm32_seek()` also relies on monotonic end positions.

## Change

Add load-time validation for checked segment ends, nonzero length, start/end monotonicity, chirp support, and worst-case active segment count against `SPECTRAL_ARM32_MAX_ACTIVE`.

## Why minimal

This keeps the hot callback branch-free and bounded while making the required input shape explicit at load time.
