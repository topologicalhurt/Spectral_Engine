# Core audit pass 108: Metal prepared GPU dispatch wiring

## Summary

Pass 108 wires Metal to the `SpectralGpuDispatchPlan` introduced in Pass 107.

## What changes

Metal no longer owns host-side GPU preparation directly:

```text
segment cache try-get
tile preprocessing/cache lookup
GPU params packing
tile byte derivation
zero-reference tile handling
```

Those are now owned by `spectral_gpu_dispatch_plan_init()`.

Metal remains responsible only for Metal-specific work:

```text
Metal buffer growth
NoCopy wrapping for cached SegmentGpu source
Metal command encoder setup
Metal dispatch/completion
copyback
```

## Why this is critical

This is the first real step away from process-local GPU cache plumbing in each
backend and toward an explicit prepared-dispatch architecture.
