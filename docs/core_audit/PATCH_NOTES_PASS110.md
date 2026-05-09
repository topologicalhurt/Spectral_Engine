# Core audit pass 110: GPU cache lookup encapsulation

## Summary

Pass 110 hides process-local GPU cache lookup APIs from backend headers.

After Metal/CUDA consume `SpectralGpuDispatchPlan`, backend code no longer needs:

```c
gpu_seg_cache_try_get()
gpu_tile_cache_try_get()
```

Those are internal implementation details of dispatch-plan preparation.

## Fix

The header still exposes producer-side cache APIs:

```c
gpu_seg_cache_set()
gpu_tile_cache_set()
gpu_*_cache_clear()
```

but no longer exposes try-get lookup APIs.

## Why this is critical

The prepared dispatch plan is now the only legitimate consumer of one-shot cache
lookups. Hiding lookup APIs prevents future backends from bypassing the shared
dispatch preparation layer.
