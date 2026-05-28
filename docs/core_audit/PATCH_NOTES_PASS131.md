# PATCH_NOTES_PASS131 — ARM32 process/load bounds

## Problem

The ARM32 embedded process path used raw `start + length` arithmetic for segment ends, performed zero-output writes before block-size clamping, and entered the hot path without proving segment storage and LUT state.  These are small C-level issues, but they sit in the embedded callback path where overflow and unchecked pointers are unacceptable.

## Change

- Adds saturated `spectral_arm32_segment_end_sat_u32()`.
- Adds centralized bounded zero-output helper.
- Clamps requested block length before any zero-fill.
- Clamps processing to the remaining output duration before computing `out_end`.
- Rejects missing segment storage or missing oscillator LUT before entering the hot loop.
- Allows empty ARM32 loads while rejecting missing storage for non-empty loads.
- Uses saturated segment-end math in seek and activation paths.

## Validation

- `tests/core_math/test_core_pass131_arm32_process_bounds.py`
- `tools/core_audit/core_static_audit.py`

## Risk

Moderate-low.  The patch changes only boundary and arithmetic guards; hot-loop synthesis math is left intact for the later dedicated ARM architecture round.
