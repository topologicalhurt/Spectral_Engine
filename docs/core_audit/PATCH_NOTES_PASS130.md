# PATCH_NOTES_PASS130 — LUT lookup domain contract

## Problem

The oscillator LUT lookup assumes a 16-bit unsigned phase domain, but `SPECTRAL_OSC_LUT_BITS` had no explicit range guard.  Values above 16 make the fractional-bit arithmetic underflow and can create invalid shifts.  The hot lookup also trusted the LUT pointer unconditionally.

## Change

- Adds a preprocessor guard requiring `SPECTRAL_OSC_LUT_BITS` to be in `[1,16]`.
- Guards `spectral_lut_sin()` against NULL LUT pointers.
- Uses explicit unsigned shift constants in the guarded 16-bit phase domain.
- Makes `spectral_lut_cos()` wrap the quarter-cycle phase offset explicitly through `uq16_t`.

## Validation

- `tests/core_math/test_core_pass130_lut_lookup_contract.py`
- `tools/core_audit/core_static_audit.py`

## Risk

Low.  Valid LUT configurations preserve the same lookup math.  Invalid build-time configurations now fail at preprocessing instead of producing invalid shifts.
