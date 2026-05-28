# PATCH_NOTES_PASS129 — Fixed-point finite narrowing contract

## Problem

Several Q15/Q31 helpers converted floats to integer domains without first rejecting NaN/Inf.  Converting a non-finite float, or a finite float outside the representable target integer range, is not a safe kernel boundary.  The phase and omega helpers had the same risk before Q15/Q8.8 narrowing.

## Change

- Adds finite-checked `spectral_float_to_q15()` and `spectral_float_to_q31()` helpers.
- Routes `FLOAT_TO_Q15` and `FLOAT_TO_Q31` through the helpers, avoiding repeated macro expression evaluation and rejecting NaN/Inf before integer casts.
- Guards phase-radian and omega-to-Q8.8 conversion helpers.
- Adds null-span guards to both NEON and portable Q31-to-Q15 bulk converters.

## Validation

- `tests/core_math/test_core_pass129_fixed_point_finite_conversion.py`
- `tools/core_audit/core_static_audit.py`

## Risk

Low.  Non-finite conversion now returns silence/zero instead of relying on undefined narrowing behavior.  Valid finite inputs preserve the old saturation policy.
