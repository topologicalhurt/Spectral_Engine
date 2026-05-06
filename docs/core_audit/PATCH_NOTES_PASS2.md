# Core audit pass 2: approximation gates and validation workflow

## Summary

This patch is intentionally correctness-first. Some exact default paths may be slower than the previous approximate paths. That is deliberate: this engine is intended as a reusable kernel, so approximations must be opt-in, named, benchmarked and bounded.

## Changes

New approximation gates:

- `SPECTRAL_ENABLE_APPROX_TRIG`
- `SPECTRAL_ENABLE_APPROX_ATAN2`
- `SPECTRAL_ENABLE_APPROX_INV_SQRT`
- `SPECTRAL_METAL_FAST_MATH`

Default value for all four is `0`.

The correct optimization workflow is now:

1. keep exact defaults;
2. add a benchmark fixture;
3. add an error-bound fixture;
4. enable one approximation at a time;
5. record both error and speedup.
