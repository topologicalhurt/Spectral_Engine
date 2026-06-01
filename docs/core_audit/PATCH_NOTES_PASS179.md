# Patch notes — Pass 179: CTF sweep increment 19 — GPU synthesis dispatch cluster + cross-backend timbre gate (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **host-side GPU
synthesis dispatch cluster** shared by the Metal and CUDA backends — the path that
turns a validated `SegmentArray` + `SynthParams` into the buffers a GPU kernel
consumes, plus the cross-backend timbre-support gate:

```text
- core/spectral_synth_internal.h  gpu_timbre_supported, gpu_synth_params_pack_checked,
                                  SpectralGpuDispatchPlan layout
- core/spectral_synth_internal.c  spectral_gpu_dispatch_plan_init / _free,
                                  gpu_tile_preprocess_cached, gpu_seg_cache_*
- synth/backends/gpu/metal/spectral_synth_metal.m   host dispatch (buffer build/upload)
- synth/backends/gpu/cuda/spectral_synth_cuda.cu    timbre gate call site
- core/oscillator.c (Metal MSL oscillator() string)  the 6-waveform GPU formula mirror
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol a
clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

### Cross-backend timbre gate matches the Metal MSL capability (the lead that opened this pass)

```text
- The Metal MSL oscillator() switch (oscillator.c:175-186) implements timbres 0..5
  (SINE,SAW,SQUARE,TRIANGLE,ASIN,PARABOLA) with default->fast_sin; it has NO width
  parameter, so it structurally cannot do QUANTIZED(6)/PWM(7).
- gpu_timbre_supported(timbre) = (int)timbre <= TIMBRE_PARABOLA — i.e. exactly 0..5.
  Both backends call gpu_check_timbre_or_fallback before dispatch (metal:263, cuda:217),
  so 6/7 fall back to the CPU synth (which does support width timbres). Therefore a
  quantized/pwm request never reaches the 6-waveform MSL kernel -> NO silent sine
  substitution, NO cross-backend divergence. (CUDA #includes spectral_osc_formulas.h
  directly so it has all 8 forms, but is gated identically — conservative, not wrong.)
- Out-of-range / invalid timbre id: > 5 -> not supported -> CPU path, which warns-once
  and renders TIMBRE_SINE. Consistent fallback, no OOB.
```

### Dispatch-plan construction — overflow & lifetime

```text
- spectral_gpu_dispatch_plan_init zero-inits the plan, then sizes segment_bytes via
  spectral_array_bytes(sa.count, sizeof(SegmentGpu)) (overflow-checked). sa.count is
  uint32_t (spectral_common.h:90), so the size_t->uint32_t pass into
  gpu_seg_cache_try_get(sa.count) cannot truncate.
- gpu_seg_cache_try_get: returns the pre-packed SegmentGpu pointer ONLY on an exact
  uint32 count match, and clears the cache on every get (single-use). A count-only key
  is explicitly NOT treated as segment identity, so a prior call's pointer can never be
  implicitly reused — comment-documented and code-consistent.
- gpu_tile_preprocess_cached: on a tile-cache hit it re-validates the layout words
  (spectral_gpu_tile_layout_words_valid) and only then reuses; otherwise it recomputes
  via gpu_tile_preprocess (Pass 171) and sets owns_tile_data. tile_ids_bytes /
  tile_ranges_bytes are overflow-checked; total_refs==0 -> zero_output shortcut.
- _free conditionally frees tiles only when owns_tile_data, then zero-inits — no
  double-free of a cache-owned GpuTileData, safe on a partially-built plan.
```

### Boundary pack (`gpu_synth_params_pack_checked`)

```text
- Guards sp/out non-NULL and tile_size!=0; rejects out_len>UINT32_MAX,
  num_segments>UINT32_MAX, timbre outside [SINE,PWM]; requires stretch / inv_stretch /
  inv_stretch_sq / pitch_factor all finite-positive before the 32-bit narrowing pack.
  This is the single sanctioned size_t->uint32 crossing for GPU dispatch.
```

### Metal host dispatch — buffer sizing & NULL segment_source

```text
- plan.segment_source NULL (the common cache-miss case) is handled: the code grows
  g_mtl.segBuf to plan.segment_bytes and packs from sa via
  spectral_segment_pack_gpu_array(sa.segs, sa.count, ...) — it never dereferences a
  NULL source. Non-NULL -> newBufferWithBytesNoCopy of exactly segment_bytes.
- tileIds/tileRanges/output buffers grown to the plan's overflow-checked byte counts
  before the matching memcpy; zero_output memsets exactly output_size. The debug
  avg = total_refs/num_tiles is only reached when total_refs!=0 (=> num_tiles>=1).
```

## Verification

```text
- No source changed this pass (read-only audit), so the Pass 178 green state is
  preserved by construction. Re-confirmed the gate:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-178 (see prior notes) and now the GPU
synthesis dispatch cluster + cross-backend timbre gate (179, clean — the Metal
6-waveform MSL kernel is correctly fenced off from the width-based QUANTIZED/PWM
timbres by gpu_timbre_supported, and the dispatch-plan construction / boundary pack /
Metal buffer sizing are overflow-checked with NULL segment_source handled). Phase D
(compiled harness + LUT golden-vector loop) follows.
