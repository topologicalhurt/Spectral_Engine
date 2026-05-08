# Core audit pass 56: Q15 normalization CMSIS length contract

## Summary

Pass 56 hardens `spectral_normalize_q15()` at the CMSIS boundary.

The public function accepts `size_t len`, but CMSIS-DSP normalization helpers
accept `uint32_t` lengths. The old code cast directly:

```c
arm_absmax_q15(buffer, (uint32_t)len, ...)
arm_shift_q15(buffer, -shift_amt, buffer, (uint32_t)len)
```

## Bug

If `len` is not representable as `uint32_t`, the CMSIS path can analyze and
shift only a truncated prefix while the public function reports a result for the
full span.

That violates the same integer-domain boundary rule used throughout the kernel:
do not allocate, scan, dispatch, or transform in one integer domain while the
callee observes a narrowed domain.

## Fix

`Spectral_normalize_q15()` now rejects lengths above `UINT32_MAX` before any
CMSIS cast.

It also clears `*shift` at entry when provided, so all failure/early-return paths
leave deterministic output state.

## Reviewer Walkthrough

1. `shift` is initialized to zero immediately.
2. Null/empty buffers return zero without touching memory.
3. Lengths outside the CMSIS `uint32_t` domain fail closed.
4. The CMSIS path narrows once into `len_u32` after the proof.
5. Both `arm_absmax_q15()` and `arm_shift_q15()` use that checked value.

## Why this is critical

Q15 normalization is an embedded/public DSP helper. A partial normalization due
to silent length truncation can leave the buffer in a mixed scaling state, which
is worse than a clean failure.
