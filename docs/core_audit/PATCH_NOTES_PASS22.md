# Core audit pass 22: synthesis output-size overflow contract

## Summary

Pass 22 fixes a shared synthesis preflight contract violation.

The shared backend preflight used `synth_validate_inputs()` to classify output
byte-count overflow as an early exit. Public backends then returned the
preflight error value, which was still `SPECTRAL_OK`.

That means an impossible output byte count could be reported as successful
synthesis instead of `SPECTRAL_ERR_OVERFLOW`.

## Bug

The problematic path was:

```text
out_len * elem_size overflows
synth_validate_inputs() returns SYNTH_VALIDATE_EARLY_EXIT
SynthPreflight.error remains SPECTRAL_OK
backend returns SPECTRAL_OK
```

This violates the kernel error contract. Overflow is not silence, not empty
input, and not a benign no-op. It is an error.

## Fix

`synth_preflight_common()` now distinguishes true benign early exits from
contract failures before calling `synth_validate_inputs()`:

```text
NULL timing pointer      -> SPECTRAL_ERR_PARAM
zero element size        -> SPECTRAL_ERR_PARAM
output byte-count overflow -> SPECTRAL_ERR_OVERFLOW
```

The CPU synth driver also stopped using:

```c
memset(out_buffer, 0, out_len * elem_size)
```

on allocation failure. It now computes a checked `out_bytes` value once and uses
that for failure zeroing.

## Why this is critical

This shared path is used by CPU/native synthesis and by GPU backends through the
same `SynthPreflight` contract. A synthesis kernel must never return success
for an arithmetic overflow in its output buffer contract.

## Validation

Run:

```sh
python3 tools/core_audit/core_static_audit.py .
python3 tests/core_math/test_core_pass22_synth_overflow_contract.py
git diff --check
make clean && make configure CMAKE_BUILD_TYPE=Debug
make desktop
```
