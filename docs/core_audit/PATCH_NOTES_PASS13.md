# Core audit pass 13: estimator validation harness

## Intuition

Pass 11 added estimator candidates. Pass 12 hardened estimator contracts. Pass
13 makes estimator quality measurable against synthetic ground-truth offsets.

The existing `peak_estimator_bench.c` is useful, but it primarily measures
throughput and internal consistency on generated triplets. It is not enough to
decide whether an estimator is good for the engine's STFT signal model.

## Added harness

Pass 13 adds:

```text
tools/core_audit/peak_estimator_sweep.c
tests/core_math/test_core_pass13_peak_estimator_sweep.py
```

The sweep harness:

1. generates a known fractional-bin sinusoid;
2. applies Hann, Hamming, Blackman or Rectangular windows;
3. computes the three local DFT bins directly;
4. runs all estimator candidates;
5. reports RMS offset error, max offset error, fallback rate, complex-use rate
   and approximate runtime.

## Why this matters

Estimator papers usually state assumptions: isolated tone, DFT coefficient
access, noise model, and often rectangular/unwindowed conditions. The engine's
real model is different: windowed STFT rows with calibrated magnitude-squared
and phase. The sweep harness gives us a way to check whether those assumptions
survive contact with the engine.

## Current hard guard

The test keeps one strict correctness guard:

```text
Hann + log-parabolic + high SNR => max error <= 0.035 bins
```

That protects the current conservative default without prematurely requiring
the advanced estimators to win under assumptions they may not satisfy.

## Validation matrix

The sweep is intentionally separate from the micro-benchmark: it answers
whether an estimator tracks the engine's current STFT signal model, not only
whether it is internally self-consistent on generated three-bin triplets.

### Offset sweep

The harness evaluates interior-bin tones with known fractional-bin offsets. For
each case it:

1. generates a single sinusoid at a known bin plus offset;
2. applies the selected analysis window;
3. computes the local DFT bins directly;
4. calls the selected `SpectralPeakEstimatorType`;
5. compares the emitted offset with the known ground-truth offset.

The current matrix covers:

```text
windows:    Hann, Hamming, Blackman, Rectangular
estimators: log-parabolic, magnitude-parabolic, Jacobsen, Candan, Quinn second
SNRs:       120 dB, 60 dB, 30 dB
metrics:    cases, valid rate, RMS error, max absolute error, fallback rate,
            complex-use rate, average nanoseconds per estimate
```

### Required guard

The default engine profile remains Hann plus log-power parabolic interpolation.
For that profile, the high-SNR sweep must satisfy:

```text
Hann + log-parabolic + 120 dB SNR => max_abs_err <= 0.035 bins
fallback_rate == 0.0
```

This bound is deliberately local to the current default. The advanced complex
estimators remain explicit because their source assumptions differ from the
engine's default windowed magnitude-squared tracker contract.

### Interpretation

A failing hard guard blocks the pass. Other sweep rows are diagnostic: they are
used to compare estimator assumptions, approximation flags, and future window
profiles before changing defaults.
