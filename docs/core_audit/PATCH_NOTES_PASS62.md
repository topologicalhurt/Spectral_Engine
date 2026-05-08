# Core audit pass 62: FFT magnitude scaling finite-output contract

## Summary

Pass 62 hardens FFT magnitude-squared calibration.

Window calibration scales are finite and positive, but the scaled magnitude
values themselves were not checked. A finite FFT magnitude can still overflow
when multiplied by a large finite calibration scale.

## Bug

The scaling path did this:

```c
magsq[i] *= positive_scale;
```

and then recomputed the frame maximum.

If the product overflowed to Inf, thresholding and tracking could observe an
infinite frame maximum. That is not a valid DSP measurement; it is a contract
failure in the scaled magnitude domain.

## Fix

Magnitude scaling now routes every bin through:

```c
spectral_fft_scaled_magsq(value, scale)
```

The helper rejects:

```text
non-finite input magnitude
negative magnitude
invalid scale
non-finite scaled product
scaled product outside FLT_MAX
```

Invalid scaled bins become zero. The trackable max scan now ignores non-finite
values defensively.

## Reviewer Walkthrough

1. Scale validation remains unchanged.
2. Endpoint bins use the endpoint scale through the checked helper.
3. Interior bins use the positive-bin scale through the checked helper.
4. Trackable frame max only considers finite positive interior bins.
5. The function still has no error channel, so invalid bins fail closed to zero.

## Why this is critical

The analysis kernel's thresholding logic consumes scaled magnitude-squared
values. It must not treat Inf as a valid peak power or frame maximum.
