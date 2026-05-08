# Core audit pass 71: tracker scalar-domain derivation contract

## Summary

Pass 71 hardens `SpectralTracker` creation and threshold updates.

The tracker derives core scalar state from public analysis parameters:

```text
threshold scale
threshold power
inverse hop
frequency step omega
frequency step df
hop as float
```

Those values are consumed by the peak-estimation hot path.

## Bug

`Spectral_tracker_create()` accepted only coarse positive checks for sample
rate, FFT size and hop. It then computed:

```c
powf(10.0f, db_thresh / 20.0f)
thresh_linear_sq * max_magsq
(float)sr / n_fft
1.0f / hop
```

without proving every derived scalar was finite, positive where required, and
representable as `float`.

`Spectral_tracker_update_threshold()` also multiplied a cached scale by a new
frame maximum without validating the new maximum or product.

## Fix

A single derivation helper now validates:

```text
sample-rate domain
power-of-two FFT size
n_freqs == n_fft / 2 + 1
finite db threshold
finite non-negative max magnitude
finite threshold scale
finite threshold power
finite inverse-hop and frequency-step scalars
```

Tracker creation stores only derived values returned by that helper.

Threshold updates now reject non-finite/negative frame maxima and non-finite
threshold products through the tracker first-error setter.

## Reviewer Walkthrough

1. `spectral_tracker_derive_create_scalars()` derives everything in `double`.
2. Every value is checked before narrowing to `float`.
3. `spectral_tracker_create()` no longer recomputes threshold/frequency scalars.
4. `spectral_tracker_update_threshold()` uses the same finite-product discipline.
5. Invalid threshold updates preserve root-cause errors through
   `spectral_tracker_set_error()`.

## Why this is critical

Tracker scalar state drives every emitted segment. It is not enough for raw
inputs to be positive; the actual scalars consumed by the estimator and segment
emitter must be finite and inside the engine domain.
