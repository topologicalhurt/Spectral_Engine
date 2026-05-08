# Core audit pass 51: CPU synth reduction byte-count contract

## Summary

Pass 51 fixes the CPU synthesis reduction byte-count contract.

Earlier passes proved output byte counts in `synth_cpu_driver()` and the
thread-buffer allocator. The float reduction step still recomputed a raw byte
product locally:

```c
memcpy(out_buffer, tb->bufs[0], out_len * sizeof(float));
```

That violates the post-Pass-21 allocation rule: allocation/copy byte counts must
come from the checked contract, not local arithmetic.

## Bug

The driver already computes:

```c
out_bytes = out_len * elem_size
```

with checked arithmetic. Then it calls the reducer, which recomputes one copy
length in a different expression.

If the reduction path ever receives inconsistent element size or length, the
driver had no way to propagate a reducer contract failure. The reducer was
`void`, so it could only copy or silently return.

## Fix

`ReduceFn` now carries the checked `out_bytes` value:

```c
ReduceFn(..., size_t out_len, size_t out_bytes)
```

The float reducer copies exactly that checked byte count. Both float and native
reducers validate `tb->buf_size >= out_bytes` before reading per-thread buffers.

`synth_cpu_driver()` now propagates reducer errors, zeroes the output on
reduction failure, frees thread buffers, and returns the root error.

## Reviewer Walkthrough

1. `synth_cpu_driver()` computes `out_bytes` exactly once with
   `spectral_size_mul()`.
2. The reducer receives that checked byte count.
3. Float reduction uses `memcpy(..., out_bytes)`, not a local product.
4. Native reduction validates the same byte contract before entering the loop.
5. Any reducer contract failure is returned through `synth_cpu_driver()`.

## Why this is critical

Reduction is the final CPU synthesis write boundary. It must use the same byte
shape that allocated the per-thread arenas. Divergent byte arithmetic at the
last copy can invalidate all earlier overflow hardening.
