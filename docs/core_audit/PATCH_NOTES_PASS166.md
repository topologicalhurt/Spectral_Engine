# Patch notes — Pass 166: CTF sweep increment 6 — synth backends + analysis orchestration (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass sweeps the
**synth-backend + analysis-orchestration cluster** — the analysis dispatch and
the two STFT paths (`analysis/spectral_analysis.c`,
`analysis/spectral_analysis_full.c`, `analysis/spectral_analysis_fused.c`), the
multi-threaded CPU synthesis backend (`synth/backends/cpu/spectral_synth_cpu.c`)
and the embedded-simulation backend (`synth/backends/sim/spectral_synth_simulation.c`).

The one real defect is a **read-after-reset of the window descriptor** in the
full-matrix analysis path: the window context is freed (which zeroes its struct)
*before* its descriptor is read, so the descriptor argument silently falls back to
a hardcoded default.

## Change

```text
1. Window descriptor read after the context is freed/zeroed  (read-after-reset)
   analysis/spectral_analysis_full.c  (spectral_analysis_run_full)
   The teardown ordering was:
     spectral_analysis_window_context_free(&window_ctx);   // zeroes the struct
     ...
     spectral_track_peaks_with_window_descriptor(...,
         window_ctx.descriptor ? window_ctx.descriptor
                               : spectral_window_descriptor(SPECTRAL_WINDOW_HANN),
         ...);
   spectral_analysis_window_context_free() does `*ctx = (Context){0}`
   (spectral_analysis.c:45), so window_ctx.descriptor is NULL by the time the
   ternary reads it — the first operand is statically dead and the call ALWAYS
   takes the fallback. window descriptors live in a static const table
   (spectral_window_descriptor returns &spectral_window_descriptors[i],
   spectral_windows.c:140), so this is not a dangling read — but it is a real
   latent defect:
     (a) the author's intent (pass the context's actual descriptor) is silently
         defeated; the ternary's true branch can never execute; and
     (b) it is a correctness landmine: spectral_analysis_window_context_init()
         already takes a window TYPE parameter (here pinned to HANN), so the
         moment a non-HANN window is used the magnitude calibration would be
         computed from HANN's descriptor instead of the real one.
   Today it is behaviour-neutral only because the type is hardcoded HANN and the
   fallback reconstructs the identical HANN descriptor pointer.
   Fix: free window_ctx AFTER the track-peaks call (move the free below it, next to
   the stft free), so window_ctx.descriptor is the real, non-NULL descriptor when
   read. Same pointer value on the current HANN path -> identical output; removes
   the dead branch and the non-HANN landmine.
```

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is solid:
- `analysis/spectral_analysis.c` — `spectral_analysis_shape_init` validates
  sr/n_fft(power-of-two, >= MIN)/hop(>0)/db_thresh(finite)/n_samples(>= n_fft) and
  computes n_frames without underflow (n_samples >= n_fft proven first); every
  size product/sum goes through `spectral_size_{mul,add}` overflow guards; the
  AUTO/FULL/FUSED path decision is total. The STFT-matrix and FFT-byte-estimate
  allocators are overflow-checked and free-on-failure.
- `analysis/spectral_analysis_fused.c` — reads `window_ctx.samples` (FFT input)
  and `window_ctx.descriptor` (tracker) while the context is still live; the two
  `window_context_free` sites are both on teardown AFTER the last use. The
  descriptor handed to the tracker is a static-table pointer, valid after the
  context is zeroed. No read-after-reset (this is the positive control that made
  defect 1 visible).
- `synth/backends/cpu/spectral_synth_cpu.c` — `thread_buffers_alloc` is fully
  overflow-/alignment-hardened (checked out_len*elem_size, checked cache-line
  align-up, pointer-add wrap guard near UINTPTR_MAX, calloc-zeroed arena); the
  parallel partition `[p*count/n_parts, (p+1)*count/n_parts)` tiles the segment
  set disjointly into per-thread buffers then sums in reduce; the fade length is
  the profile-selected `SPECTRAL_SYNTH_CPU_FADE_SAMPLES`; empty input zeroes the
  output. Float and native reduce paths both bound-check `buf_size >= out_bytes`.
- `synth/backends/sim/spectral_synth_simulation.c` — `segment_to_q15` is the
  pass-161-hardened saturating converter (double-precision stretch with finite +
  range gates, length clamped to the u16 field, df forced 0 for the chirp-
  rejecting loader). The workload-model block walk (activation / deactivation /
  per-block sample windowing) is overflow-safe: `block_end = out_pos + block_len
  <= out_len <= UINT32_MAX`, and every `blk_end - blk_start` is proven non-negative
  from the activation invariant `seg->start < block_end`. The audio loop drives the
  REAL `spectral_arm32_process` in <=256-sample chunks: its only return-0 path
  reachable here (`output_position >= output_length || num_segments == 0`) zeroes
  the output first (spectral_synth_arm32.c:739), and on success it returns exactly
  `num_samples` and advances `output_position` by the same (lines 1039-1041) — so
  `n = (got>0)?got:want` never copies stale stack data and the cursor cannot stall.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mavx2 /
  -mno-avx512f unused-command-line-arg notes on host).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- functional parity on resources/testing/sin_440hz.wav unchanged vs pass 165:
  desktop and simulate both detect 657 segments; desktop output peak 0.95000
  (-0.45 dBFS) / RMS -5.34 dBFS, simulate peak 0.95000 / RMS -5.42 dBFS (within
  the 0.08 dB float-vs-Q15 quantization gap). The fix passes the identical HANN
  descriptor pointer on the current path, so rendered output is unchanged; only
  the dead ternary branch and the non-HANN landmine are removed.
```

## Scope (Phase C increment)

Synth-backend + analysis-orchestration cluster, one defect fixed: the full-matrix
path now reads the window descriptor while the context is live (removing a
read-after-reset whose true branch was statically dead and which would mis-calibrate
any non-HANN window). With this increment the core/analysis/synth CTF sweep has
cleared fixed-point (161), analysis/peak-track (162), port/SIMD/out (163),
hashing/parsing/path (164), DSP-math/FFT-scaling + alloc/cache (165) and
synth-backends + analysis-orchestration (166). The core/analysis/synth surface is
now swept end-to-end. Phase D (compiled harness + LUT golden-vector loop) follows.
