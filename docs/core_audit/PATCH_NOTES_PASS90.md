# Core audit pass 90: CPU synth finite-output postcondition

## Summary

Pass 90 adds a final postcondition check to float CPU synthesis.

Earlier passes validate segment inputs and output file writes, but CPU synthesis
can still accumulate many finite segment contributions into Inf/NaN if the
request is pathologically large. The backend should not return `SPECTRAL_OK`
with non-finite output.

## Bug

`Synth_cpu_driver()` reduced per-thread buffers into the caller's output and
returned success immediately. There was no final finite-output check for the
float path.

If accumulation overflowed to Inf, later normalization or file-output code would
catch it, but the synthesis backend itself would have already claimed success.

## Fix

The float reduction wrapper now validates the final output buffer after
reduction:

```text
every float output sample is finite
```

If any sample is non-finite, the reducer returns `SPECTRAL_ERR_PARAM`. The
existing driver error path zeros the output and returns the error.

Native/Q15 output is not affected because it is integer-domain.

## Reviewer Walkthrough

1. Float reduction copies and accumulates exactly as before.
2. After all `spectral_vadd()` reductions, the output span is scanned.
3. Non-finite output makes the reducer fail.
4. `synth_cpu_driver()` already zeroes output and returns reducer errors.
5. File output still has its own finite-sample boundary.

## Why this is critical

A synthesis backend's success value must mean "the output buffer is valid kernel
output." Returning success with NaN/Inf samples violates that contract.
