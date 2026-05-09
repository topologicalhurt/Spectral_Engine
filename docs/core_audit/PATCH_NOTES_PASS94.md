# Core audit pass 94: fused-analysis scratch row ownership

## Summary

Pass 94 turns the fused analysis row/phase scratch buffers into an owned
structure.

The fused path previously managed four raw local pointers:

```text
row_curr
row_next
phase_curr
phase_next
```

allocation, null checking, rotation and free were spread across the OpenMP worker
body.

## Fix

Pass 94 introduces:

```c
SpectralFusedScratchRows
spectral_fused_scratch_rows_alloc()
spectral_fused_scratch_rows_free()
spectral_fused_scratch_rows_rotate()
```

The OpenMP worker now owns exactly one scratch-row object.

## Why this is critical

The fused path is dense. Scratch row ownership must be obvious so future
optimizations do not leak rows, rotate only half the state, or miss one free on
failure.
