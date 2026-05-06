# Core audit pass 17b: window-aware amplitude estimation

## Why this replaces the previous Pass 17 assumption

The fitted parabolic peak-height formula itself is a property of a parabola:

```text
y_peak = y_center - 0.25 * (y_left - y_right) * offset
```

But the decision to model a windowed spectral peak as a parabola in log-power
is window-dependent.

Hann, Hamming and Blackman can reasonably use the local log-parabolic model as
a bounded approximation. A rectangular window has a sinc-shaped main lobe, so
the descriptor defaults to center-bin amplitude instead of pretending the same
log-parabolic height model is canonical.

## Change

Window descriptors now own two policies:

```text
interp_magsq: sub-bin frequency offset estimator
peak_magsq:   sub-bin peak-height estimator
```

The peak estimator no longer hard-codes log-parabolic peak-height correction.
It calls the active window descriptor's `peak_magsq` callback and then applies
the existing bounded-gain safety check.

## Current built-ins

```text
Hann        -> log-parabolic peak height
Hamming     -> log-parabolic peak height
Blackman    -> log-parabolic peak height
Rectangular -> center-bin peak height
```

This is conservative and extensible. Future windows can register an exact or
better validated amplitude estimator without changing the tracker.
