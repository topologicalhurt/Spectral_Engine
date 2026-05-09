# Full/fused analysis parity harness

## Purpose

This document defines the behavioral parity harness for analysis.

The goal is not bit-identical output. Full and fused analysis have different
scheduling and allocation shapes, but they should agree on segment content within
well-defined tolerances for deterministic fixtures.

## Entry points

After Pass 118:

```c
analyze_audio_with_path_mode(..., SPECTRAL_ANALYSIS_PATH_FULL, ...)
analyze_audio_with_path_mode(..., SPECTRAL_ANALYSIS_PATH_FUSED, ...)
```

## Deterministic fixtures

```text
silence
single sinusoid, bin-centered
single sinusoid, off-bin
two separated sinusoids
short chirp
amplitude ramp
```

## Comparison rules

Segment arrays should be sorted by:

```text
start, omega, amp
```

Suggested initial tolerances:

```text
count: exact or documented one-segment edge tolerance
start: <= 1 sample
length: <= hop
omega: relative <= 1e-4 or absolute <= 1e-6
amp: relative <= 1e-3 or absolute <= 1e-5
df/da: relative <= 1e-3 or absolute <= 1e-5
```

## Required harness behavior

```text
generate fixture
run full path
run fused path
sort segment arrays
compare with tolerances
print per-field max errors
return nonzero on failure
```

## Forbidden

```text
testing only AUTO threshold behavior
string-only parity tests
allowing fused-only/full-only path to change estimator/window policies
```
