# Core audit pass 91: contract routing and alias-wrapper removal

## Summary

Pass 91 is a cleanup/wiring pass, not another guard pass.

Phase C introduced `spectral_contracts.h`, but preserved several local wrapper
functions so older tests and module-specific names would keep passing. Those
wrappers are now removed where they are pure aliases.

## Problem

Phase C still left patterns like:

```c
static int seg_cache_segments_valid(...) {
    return spectral_segment_array_payload_valid(...);
}
```

and:

```c
static int spectral_audio_samples_finite(...) {
    return spectral_f32_span_finite(...);
}
```

That is better than duplicated loops, but still violates the new maintenance
rule: do not keep alias functions that merely wrap another function.

## Fix

Pass 91 rewires call sites directly to canonical contracts and removes pure
wrappers from:

```text
spectral_seg_cache.c
spectral_synth_internal.c
spectral_in.c
spectral_out.c
spectral_synth_cpu.c
spectral_wavetable.c
spectral_windows.c
```

It also adds:

```text
docs/core_audit/KERNEL_PATCHING_GUIDELINES.md
```

and updates Phase C / legacy static tests so they assert canonical wiring rather
than stale wrapper names.

## Reviewer Walkthrough

1. Segment cache payload checks call `spectral_segment_array_payload_valid()`
   directly.
2. Segment cache GPU payload checks call
   `spectral_segment_gpu_array_matches_segments()` directly.
3. Tile layout checks call `spectral_gpu_tile_layout_words_valid()` directly.
4. Float-span checks call `spectral_f32_span_finite()` directly.
5. The synthetic local wrapper names are removed.
6. Audit/test expectations are updated to enforce direct canonical routing.

## Why this is critical

The previous hardening is only maintainable if the contracts are centralized in
practice, not only in theory. Alias wrappers preserve old naming debt and make
future patches harder to reason about.
