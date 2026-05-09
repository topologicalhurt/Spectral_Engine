# Core audit pass 109: CUDA prepared GPU dispatch wiring

## Summary

Pass 109 wires CUDA to the `SpectralGpuDispatchPlan`.

CUDA no longer owns host-side dispatch preparation directly:

```text
segment cache try-get
tile preprocessing/cache lookup
GPU params packing
tile byte derivation
zero-reference tile handling
```

CUDA remains responsible for CUDA-specific work:

```text
device buffer growth
host staging when needed
async copies
kernel launch
stream synchronization
copyback
```

## Why this is critical

Metal and CUDA now share the same prepared dispatch contract. Backend divergence
moves back to actual backend mechanics.
