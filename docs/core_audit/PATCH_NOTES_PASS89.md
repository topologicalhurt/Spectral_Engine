# Core audit pass 89: oscillator width and quantized-cast contract

## Summary

Pass 89 fixes a real undefined-behavior risk in the oscillator formulas.

`Spectral_osc_quantized()` maps phase to quantized steps using an integer cast.

## Bug

The old formula did:

```c
return (float)(int)(rads * width) * (1.0f / width);
```

without proving:

```text
rads is finite
width is finite and positive
rads * width is finite
rads * width is inside int range
```

Converting a NaN, Inf, or out-of-range finite float to `int` is undefined
behavior in C.

`Spectral_osc_pwm()` also treated NaN width as the "width <= 0" branch, hiding
invalid oscillator state as a constant waveform.

## Fix

Quantized oscillator now validates the full cast domain before converting to
`int`.

PWM now rejects non-finite phase/width with zero.

The oscillator formula version is bumped so backend parity guards catch stale
formula mirrors.

## Reviewer Walkthrough

1. `spectral_osc_formulas.h` includes `<limits.h>` for `INT_MIN/INT_MAX`.
2. Quantized oscillator rejects invalid phase/width.
3. The scaled phase is checked for finiteness and int representability.
4. The reciprocal width is checked before multiplication.
5. Formula version checks in CPU/Metal sources are updated.

## Why this is critical

This is not cosmetic hardening. Float-to-int conversion outside the integer
domain is C undefined behavior. Oscillator formulas are kernel math and must not
contain UB for caller-provided segment widths.
