# Core audit pass 42: GPU synth parameter packing contract

## Summary

Pass 42 fixes a GPU backend boundary bug.

Shared `SynthParams` stores `out_len` as `size_t`, but the Metal and CUDA
kernel parameter ABI stores output length and segment counts as 32-bit values.
The old pack helper silently cast:

```c
(uint32_t)sp->out_len
(uint32_t)timbre
```

without making the boundary explicit.

## Bug

A CPU/native synthesis request can represent output lengths wider than the GPU
kernel ABI. If a GPU path silently truncates `out_len`, the kernel can render or
bounds-check against the wrong output length.

Even when tile preprocessing rejects some oversized GPU shapes, the parameter
pack itself is the ABI boundary and must be checked there too.

## Fix

The header now exposes:

```c
gpu_synth_params_pack_checked(...)
```

It validates:

```text
out_len representable as uint32_t
num_segments representable as uint32_t
tile_size nonzero
timbre enum inside supported domain
all derived SynthParams scalars finite and positive
```

Metal and CUDA now call the checked helper and fail closed instead of silently
packing a truncated ABI struct.

## Reviewer Walkthrough

1. `GpuSynthParams` remains the Metal/CUDA ABI struct.
2. `gpu_synth_params_pack_checked()` is the canonical boundary-crossing helper.
3. The old `gpu_synth_params_pack()` remains as a compatibility wrapper, but GPU
   backends no longer use it.
4. Metal and CUDA both return an error before dispatch if checked packing fails.

## Why this is critical

GPU kernels do not see `size_t`. They see a 32-bit ABI. Every crossing from
host-side kernel state into GPU parameter memory must prove representability,
or backend output can silently diverge from CPU/native output.
