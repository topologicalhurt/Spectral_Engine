# Core audit pass 1: foundational math and fast-math cleanup

## Summary

This bundle's apply script performs targeted edits only.

### `spectral_osc_formulas.h`

- Bumps `SPECTRAL_OSC_FORMULAS_VERSION` from 1 to 2.
- Replaces phase wrap formula that mapped `0 -> -pi`.
- Fixes canonical fade sign.

### `spectral_envelope.c`

- Fixes runtime fade sign for fade-in and fade-out.

### `oscillator.c`

- Updates Metal formula static assert to version 2.
- Fixes Metal phase wrap formula.
- Fixes Metal fade sign.

### `oscillator_simd.c`

- Fixes SIMD phase wrap formula.

### `spectral_fast_math.c`

- Makes `fast_inv_sqrt` and `fast_sqrt` exact by default.
- Gates legacy inverse-sqrt approximation behind `SPECTRAL_ENABLE_APPROX_INV_SQRT`.
- Uses two Newton iterations if legacy approximation is explicitly enabled.

### `spectral_windows.c`

- Replaces empirical power-domain sub-bin interpolation default with log-power parabolic interpolation.
- Keeps an explicit opt-in macro for future rational estimators.
- Clamps output to `[-0.5, 0.5]`.

### `spectral_windows.h`

- Corrects window-normalization documentation.
- Corrects interpolation documentation.

### `spectral_config.h`

- Defaults interpolation to log domain regardless of fast-math mode.
- Adds `SPECTRAL_TRACK_INTERP_POWER_RATIONAL` opt-in macro.
- Adds bounded `SPECTRAL_TRACK_INITIAL_SEG_CAP` macro.

### `spectral_peak_track.c`

- Replaces Linux-overcommit initial segment capacity with bounded macro.

## Validation

```sh
python3 tests/core_math/test_core_math_contract.py
python3 tools/core_audit/core_static_audit.py .
git diff --check
make clean && make configure CMAKE_BUILD_TYPE=Debug
make desktop
```

## Expected output changes

The output may change because previous phase and fade formulas were mathematically wrong. Validate against synthetic fixtures, not against the previous output as ground truth.
