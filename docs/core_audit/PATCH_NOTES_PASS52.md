# Core audit pass 52: float normalization finite-state contract

## Summary

Pass 52 hardens public float normalization.

The audio output boundary now rejects non-finite samples before writing, but
`Spectral_normalize_float()` is a public core helper used before output. It
still accepted non-finite buffers or invalid headroom and could scale a buffer
with NaN/Inf.

## Bug

The old function computed `max_amp`, then used:

```c
scale = headroom / max_amp
```

without proving:

```text
headroom is finite and non-negative
every input sample is finite
max_amp is finite
scale is finite
```

If any of those failed, normalization could turn a recoverable input-contract
error into a buffer full of NaN/Inf.

## Fix

The function now rejects:

```text
non-finite or negative headroom
non-finite input samples
non-finite/negative max amplitude
non-finite scale
```

before applying vector or scalar multiplication.

## Reviewer Walkthrough

1. `spectral_float_buffer_all_finite()` validates the input span.
2. Headroom must be finite and non-negative.
3. The vector and scalar max paths both validate `max_amp`.
4. `scale` is checked before applying it.
5. Invalid inputs leave the buffer unchanged and return `0.0f`.

## Why this is critical

Normalization is often the last DSP operation before persistence. It must not
introduce non-finite output or mask invalid upstream data with undefined scaling.
