# Core audit pass 13: estimator validation harness

## Intuition

Pass 11 added estimator candidates. Pass 12 hardened estimator contracts. Pass
13 makes estimator quality measurable against synthetic ground truth.

The existing `peak_estimator_bench.c` is useful, but it primarily measures
throughput and internal consistency on generated triplets. It is not enough to
decide whether an estimator is good for the engine's STFT signal model.

## Added harness

Pass 13 adds:

```text
tools/core_audit/peak_estimator_sweep.c
tests/core_math/test_core_pass13_peak_estimator_sweep.py
docs/core_audit/PASS13_ESTIMATOR_VALIDATION_MATRIX.md
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
