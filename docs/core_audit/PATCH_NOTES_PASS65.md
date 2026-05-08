# Core audit pass 65: window metric finite-input contract

## Summary

Pass 65 hardens the window metric boundary.

Built-in windows produce finite samples, but the window API also supports
caller-provided/future generated windows. Metric helpers must not let NaN/Inf
window samples propagate into calibration flags or scale derivation.

## Bug

`Spectral_window_sum()` and `spectral_window_energy()` accumulated whatever was
in the window span. A single NaN/Inf sample could make the metric result
non-finite.

The metrics path mostly checked derived values later, but the public sum/energy
helpers themselves did not define fail-closed behavior for invalid window
samples.

## Fix

A shared validator now checks every window sample is finite before metric
accumulation.

`Spectral_window_sum()` and `spectral_window_energy()` now also reject
non-finite or float-unrepresentable accumulated totals and return zero.

`Spectral_window_metrics()` refuses invalid spans before deriving coherent gain,
RMS gain, ENBW, or calibration scales.

## Reviewer Walkthrough

1. Built-in windows remain unchanged.
2. Caller-provided metric spans are validated sample-by-sample.
3. Sum accumulation is bounded to finite float-representable output.
4. Energy accumulation is bounded to finite float-representable output.
5. Metrics return default invalid/no-flag state for invalid spans.

## Why this is critical

Window metrics define amplitude and magnitude-squared calibration. Invalid
window samples must not produce valid-looking calibration state or public metric
values.
