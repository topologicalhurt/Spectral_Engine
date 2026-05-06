# Core audit pass 21: kernel alias and wrapper slimming

## Summary

Pass 21 removes low-value alias functions from the synthesis/model kernel,
deletes dead duplicate kernel files, and tightens audit coverage so future
passes use canonical helpers directly.

## Changes

- Replaced the three public `spectral_peak_model_requires_*` wrappers with one
  generic `spectral_peak_model_has_capability(model, capability)` API.
- Removed local oscillator wrapper functions such as `osc_saw()` that only
  forwarded to `spectral_osc_saw()`. The timbre dispatch table now points at
  canonical waveform formulas directly.
- Removed `compute_phase()` and `compute_amplitude()` aliases from
  `spectral_synth_internal.h`; CPU and SIMD synthesis call
  `spectral_segment_phase_at_f32()` and `spectral_segment_amp_at_f32()`
  directly.
- Removed the unused out-of-line `spectral_segment_math.c` wrappers and their
  source-manifest entry. Segment math is header-only unless a future caller
  needs a real out-of-line ABI for a non-inline reason.
- Lifted duplicated desktop CPU-time accounting out of the macOS/Linux
  `spectral_perf.c` platform split. Only memory and core-count queries remain
  platform-specific.
- Removed the unused `spectral_peak_chain.[ch]` files. They were not in the
  source manifest and duplicated tracker segment-copy logic, so keeping them
  made the kernel look larger than the compiled architecture.
- Added `spectral_tracker_free_segment_storage()` so create/finalize/destroy
  paths share one per-thread segment cleanup policy. The per-thread pointer
  table now uses `spectral_calloc_array()` so partial allocation failure cleanup
  is NULL-safe.
- Centralized GPU tile preprocessing scratch cleanup with
  `gpu_tile_preprocess_scratch_free()` and kept success/error ownership release
  paths identical.
- Removed the redundant `lp.amp = s->amp;` assignment in synth segment loop
  setup. This is intentionally a tiny edit: repeated field writes in hot-path
  setup are exactly the kind of noise Pass 21 is meant to catch.
- Added `segment_array_mt_apply_pending_locked()` so `segment_array_mt_get()`
  and `segment_array_mt_copy()` share one pending-swap commit path and reset the
  pending array through the canonical empty value.
- Replaced remaining local array allocation patterns in touched kernel paths
  with safe allocation helpers: audio input/output buffers, process-mask scratch
  strings, CPU synth thread arenas, and embedded-simulation segment/accumulator
  buffers now route through `spectral_malloc_array()` or
  `spectral_calloc_array()` after explicit overflow checks where error
  reporting needs to distinguish overflow from allocation failure.

## Rule

Do not add a wrapper whose only job is to rename another helper. A wrapper is
acceptable when it enforces validation, hides a backend boundary, records units,
or centralizes a non-trivial policy. Otherwise call the canonical helper.

Do not keep dead duplicate kernel files around as informal references. If a
file is not compiled, not included, and mirrors live kernel logic, delete it or
wire it into the manifest with tests. Repeated ownership cleanup should move
into central cleanup helpers when the same resources are released on success,
failure, and destroy paths.

Do not write local `malloc(count * sizeof(T))`, `calloc(1, bytes)`, or
`strlen(x) + 1` allocation arithmetic when the kernel already has safe
allocation helpers. Use `spectral_size_add()`, `spectral_malloc_array()` and
`spectral_calloc_array()` unless a platform allocator or ABI boundary requires
something else.
