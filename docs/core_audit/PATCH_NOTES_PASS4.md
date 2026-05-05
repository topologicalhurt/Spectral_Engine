# Core audit pass 4: kernel API hardening

This pass focuses on correctness at public/internal synthesis and analysis boundaries.

## Changes

- Adds centralized synthesis parameter validation for `stretch` and `pitch`.
- Propagates invalid-parameter errors through CPU, CUDA and Metal preflight paths.
- Adds native-output preflight so native synthesis receives the same validation contract as float synthesis.
- Rejects non-finite segment fields before building per-segment loop parameters.
- Rejects invalid `stretch` in GPU tile preprocessing before tile-index arithmetic.
- Strengthens analysis input validation using canonical sample-rate and FFT-size bounds.
- Validates FFT resource shape before allocating plans/buffers.

## Rationale

The engine is intended to act as a reusable kernel. Relying on CLI validation is not
adequate because host code may call analysis/synthesis entry points directly. Invalid
`stretch`, `pitch`, segment fields, sample rate, or FFT shape can otherwise produce
NaNs, infinities, invalid casts, silent empty output, or backend divergence.

## Validation

Run:

```sh
python3 tools/core_audit/core_static_audit.py .
python3 tests/core_math/test_core_pass4_static.py
make clean && make configure CMAKE_BUILD_TYPE=Debug
make desktop
```
