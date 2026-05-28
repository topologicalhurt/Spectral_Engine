# PATCH_NOTES_PASS132 — ARM simulation preflight and conversion bounds

## Problem

The embedded simulation backend still called the removed legacy `SYNTH_VALIDATE_FLOAT` macro and independently recomputed synth scalars.  Its segment conversion also stretched and cast segment starts through unchecked float-to-uint32 operations.

## Change

- Routes simulation ingress through `synth_preflight_float()`.
- Reuses canonical `SynthParams` for pitch/stretch-derived scalars.
- Rejects output lengths that cannot fit the simulator's uint32 block arithmetic.
- Includes `spectral_contracts.h` and validates every segment conversion through `spectral_segment_valid_for_synth()`.
- Converts stretched start/length in double and rejects out-of-range or non-finite casts before writing fixed-point segment state.
- Removes the later unchecked `sim_segs[i].start * stretch` cast.

## Validation

- `tests/core_math/test_core_pass132_arm_sim_preflight_conversion.py`
- `tools/core_audit/core_static_audit.py`

## Risk

Moderate.  Invalid/out-of-output-range simulation segments become zero-length no-ops.  The real ARM hot loop is not refactored here; this only gets simulation back onto the same boundary contract as the other synth backends.
