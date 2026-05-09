# Core audit pass 107: prepared GPU dispatch plan

## Summary

Pass 107 starts Phase D from `ARCHITECTURE_CLEANUP_STATUS.md`: explicit prepared
GPU dispatch.

Metal and CUDA currently each perform the same host-side dispatch preparation:

```text
choose cached/prepacked SegmentGpu source
preprocess or fetch tile layout
pack GPU params
derive tile byte counts
handle zero-reference silence
```

That is shared GPU architecture, not backend-specific work.

## Fix

Pass 107 introduces:

```c
SpectralGpuDispatchPlan
spectral_gpu_dispatch_plan_init()
spectral_gpu_dispatch_plan_free()
```

The plan owns the prepared host-side GPU dispatch contract:

```text
optional prepacked SegmentGpu source
SegmentGpu byte count
GpuTileData + ownership
zero-output tile layout flag
checked GpuSynthParams
tile id/range byte counts
```

Backends will be wired to consume this plan in the following passes.

## Why this is critical

The process-local segment/tile caches are one-shot handoffs. Their ownership and
identity policy should be consumed by one preparation layer, not reimplemented in
each backend.
