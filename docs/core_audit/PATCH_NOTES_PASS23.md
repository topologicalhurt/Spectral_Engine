# Core audit pass 23: CPU synth thread-arena stride overflow

## Summary

Pass 23 closes the remaining arithmetic boundary in the CPU synthesis thread
buffer arena. The concrete failure mode is cache-line stride overflow during
per-thread arena layout.

Pass 22 made the shared preflight reject `out_len * elem_size` overflow. The CPU
backend still had a second size calculation:

```c
tb.buf_stride = (tb.buf_size + SPECTRAL_CACHE_ALIGN - 1) & ~(SPECTRAL_CACHE_ALIGN - 1);
```

That raw addition can wrap when `tb.buf_size` is close to `SIZE_MAX`, causing
the per-thread arena stride to become smaller than the required output buffer.

## Bug

The kernel had proved:

```text
out_len * elem_size is representable
```

but had not proved:

```text
align_up(out_len * elem_size, SPECTRAL_CACHE_ALIGN) is representable
```

Those are different contracts.

## Fix

`thread_buffers_alloc()` now:

- receives a `SpectralError* out_error`;
- validates that `SPECTRAL_CACHE_ALIGN` is a nonzero power of two;
- computes stride padding with `spectral_size_add()`;
- checks `tb.buf_stride >= tb.buf_size`;
- computes arena padding with `spectral_size_add()`;
- checks pointer alignment addition before forming the aligned base pointer;
- propagates `SPECTRAL_ERR_OVERFLOW` back to `synth_cpu_driver()`.

The CPU driver now returns the allocator's real error instead of collapsing all
thread-buffer failures into `SPECTRAL_ERR_MEMORY`.

## Reviewer Walkthrough

1. `synth_cpu_driver()` first asks the shared preflight path for a validated
   `out_bytes` value. That proves the output clear operation is bounded, but it
   does not prove the thread arena stride is representable.
2. `thread_buffers_alloc()` recomputes the per-thread byte count as `out_bytes`
   and then performs a checked align-up: `out_bytes + (cache_align - 1)` is
   guarded by `spectral_size_add()` before the power-of-two mask is applied.
3. The arena payload is then `aligned_stride * n_threads`, again checked before
   allocating. The final arena allocation adds only `align_mask` bytes so the
   returned base can be advanced to a cache-line boundary.
4. The manual aligned-base adjustment also has a pointer-add guard:
   `base > UINTPTR_MAX - align_mask` rejects addresses where the integer
   addition would wrap before masking.
5. Every arithmetic failure returns `SPECTRAL_ERR_OVERFLOW`; allocation failure
   returns `SPECTRAL_ERR_MEMORY`. Keeping those distinct matters because an
   overflow means the requested render shape is invalid, not that the machine is
   temporarily out of memory.

## Why this is critical

This is not a style cleanup. It prevents an arithmetic overflow from shrinking
the per-thread accumulation buffer, which would make later segment writes and
reductions operate on an undersized arena.
