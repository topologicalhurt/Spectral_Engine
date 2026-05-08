# Core audit pass 76: Jacobsen/Candan complex-offset finite-product contract

## Summary

Pass 76 hardens the complex DFT peak-offset estimator used by both Jacobsen and
Candan policies.

The estimator reconstructs three complex bins, then evaluates a complex quotient
whose numerator and denominator are dot products of those bins.

## Bug

The old Jacobsen path computed the quotient in `float`:

```c
den_mag2 = den_re * den_re + den_im * den_im;
offset = (num_re * den_re + num_im * den_im) / den_mag2;
```

Finite reconstructed bins can still overflow these intermediate float products.
If the denominator product overflows or the numerator product overflows, the
candidate can fall into fallback behavior for the wrong reason or produce an
invalid offset.

Candan's estimator depends on this Jacobsen offset before applying its finite-N
correction, so the same bug affects both policies.

## Fix

The Jacobsen quotient is now computed in `double`:

```text
num_re / num_im
den_re / den_im
den_mag2
offset_d
```

The denominator magnitude and quotient are checked for finiteness. The offset is
clamped in double before narrowing to float.

## Reviewer Walkthrough

1. Complex triplet reconstruction remains unchanged.
2. All quotient arithmetic is performed in double.
3. `den_mag2` must be finite and above the existing tiny-denominator floor.
4. `offset_d` must be finite.
5. The final clamp happens before any float narrowing.

## Why this is critical

Complex estimators are exposed as explicit high-accuracy policies. Their
intermediate algebra must not overflow merely because a finite STFT bin has a
large magnitude.
