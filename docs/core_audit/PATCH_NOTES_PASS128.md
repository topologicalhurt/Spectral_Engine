# PATCH_NOTES_PASS128 — Window generation status contract

## Problem

`spectral_analysis_window_context_init()` treats `spectral_window_generate()` as a status-returning function, but the window API declared and implemented it as `void`.  That leaves the analysis window boundary in an incoherent state and can fail at compile time once the analysis path consumes the status value.

## Change

- Converts `spectral_window_generate()` to return `SpectralError`.
- Includes `spectral_error.h` in the window API.
- Includes `spectral_contracts.h` in the implementation because window metrics and generation now share canonical finite-span validation.
- Makes generation fail closed on invalid buffer, zero length, invalid descriptor, or non-finite generated samples.

## Validation

- `tests/core_math/test_core_pass128_window_generation_status.py`
- `tools/core_audit/core_static_audit.py`

## Risk

Low.  The analysis path already expected this API shape.  Existing call sites that ignored the return value remain source-compatible in C, while boundary-aware call sites can now fail explicitly.
