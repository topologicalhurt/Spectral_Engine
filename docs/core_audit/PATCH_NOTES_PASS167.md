# Patch notes — Pass 167: CTF sweep increment 7 — CLI / orchestration cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` (and the CLI layer that drives them) and fix it in
place. This pass sweeps the **CLI / orchestration cluster** — argument parsing and
validation (`cmd/cli/spectral_cli.c`) and pipeline orchestration
(`cmd/cli/spectral_cli_pipeline.c`) — plus the three audio/synth-helper files
carried over from the previous increment (`core/spectral_synth_internal.c`,
`core/spectral_wavetable.c`, `core/spectral_in.c`).

The one real defect is a **stretch-validation divergence at the CLI boundary**:
`SPECTRAL_MAX_STRETCH` (1000.0) is the documented stretch cap and is enforced by
*every* downstream stretch consumer, yet the CLI entry validator never checks the
upper bound — so an over-cap stretch passes the boundary, wastes the entire
analysis pass, then fails deep in synthesis.

## Change

```text
1. CLI validator missing the stretch upper bound  (boundary/contract divergence)
   cmd/cli/spectral_cli.c  (spectral_cli_validate)
   SPECTRAL_MAX_STRETCH (= 1000.0f, spectral_config.h:330) is the stretch cap, and
   it is enforced at every consumer of `stretch`:
     - synth dispatch    synth_derive_param_scalars  (spectral_synth_internal.c:23
                         -> stretch > MAX -> SPECTRAL_ERR_PARAM)
     - seg-cache         spectral_seg_cache.c:60, :136, :444
     - segment parser    spectral_segment_parser.c:52
     - gpu_tile          core/port/host/spectral_gpu_tile.c:42, :97
     - programmatic API  spectral_config_validate  (spectral_cli.c:684)
   But the CLI entry path spectral_cli_validate() only ran
   cli_validate_common_synth_params(), which checks
   spectral_is_finite_positive_f32(stretch) — i.e. "finite and > 0" — with NO upper
   bound. So a CLI invocation such as `./spectral in.wav 0 2000 0 ...` passed
   validation, and the pipeline then:
     (a) ran the full analysis stage (FFT + peak tracking — potentially many
         seconds on a large input) before
     (b) failing in run_synthesis -> synth_validate_params with
         PIPELINE_ERR_SYNTHESIS and a misleading "Synthesis failed (...)" log,
   instead of failing fast at the boundary with a clear "stretch out of range".
   It was also a direct inconsistency with the sibling programmatic validator
   spectral_config_validate(), which rejects the same value at line 684.
   Fix: after the shared common-validation call (which has already proven stretch
   finite/positive), reject `opts->stretch > SPECTRAL_MAX_STRETCH` with the CLI's
   expected/actual error idiom, mirroring spectral_config_validate()'s two-step
   structure (common helper + separate cap check). The cap is inclusive
   (`> MAX` rejects, `== MAX` accepts) to match every downstream `> SPECTRAL_MAX_STRETCH`
   consumer exactly.
```

## Why this is correct and behaviourally inert for every valid input

For any in-contract stretch (0 < stretch <= 1000) the validator outcome is
unchanged — the new branch is not taken, so analysis, synthesis and the rendered
WAV are byte-for-byte identical to pass 166. The change only affects
*out-of-contract* stretch (> 1000), which previously failed late and confusingly
and now fails immediately at the boundary with `stretch: expected finite float in
(0, 1000], got <value>` and a non-zero exit. This removes wasted analysis compute
on an input that can never synthesize, and aligns the CLI validator with both the
documented `SPECTRAL_MAX_STRETCH` contract and the programmatic validator.

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is solid:
- `cmd/cli/spectral_cli_pipeline.c` — the render path (pipeline_render_and_write)
  guards `out_len = n_samples * stretch` against non-finite / <= 0 / > SIZE_MAX
  before the double->size_t cast (lines 376-383). The two *unguarded* sibling
  computations in cache mode (lines 938, 975) are NOT a reachable UB: cache mode
  only executes when build_cache_key() succeeds, and build_cache_key ->
  spectral_seg_cache_key returns 0 for stretch > SPECTRAL_MAX_STRETCH
  (spectral_seg_cache.c:444), which disables cache mode (key == 0) before those
  lines run. With stretch thereby bounded to <= 1000 and n_samples bounded by the
  input frame count, the product cannot approach SIZE_MAX (~1.8e19) — so no guard
  is warranted there (guarding it would validate an impossible state).
  segment_array_output_length() already finite/range-checks each
  `start + length` before its size_t cast (lines 288-290). argv filtering
  (spectral_cli_parse) overflow-checks the skip/eff_argv allocations via
  spectral_size_mul and bounds the stack/heap argv switch on CLI_MAX_ARGV.
- `core/spectral_synth_internal.c` — segment_loop_params_init clamps
  `start_idx + length <= out_len`; synth_derive_param_scalars validates the full
  stretch/pitch domain (incl. the tiny-stretch `stretch*stretch <= 0` underflow
  and the MAX cap) before any backend consumes the derived scalars.
- `core/spectral_wavetable.c` — lookup_f/lookup_q are bounds-safe (idx+1 <= SIZE
  against a SIZE+1-entry table with the wrap sentinel samples[SIZE]=samples[0]);
  all loaders are overflow/size/finite/checksum hardened.
- `core/spectral_in.c` — spectral_audio_read proves sf_count_t->size_t
  representability before allocating, overflow-checks total_samples/mono_bytes,
  downmixes in double precision with finite/range gates (i*channels provably <
  total_samples); spectral_audio_window clamps start/end to [0, total_frames] with
  finite checks before the size_t casts.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mavx2 /
  -mno-avx512f unused-command-line-arg notes on host).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- functional parity on resources/testing/sin_440hz.wav (valid stretch unchanged):
    * desktop  stretch=1.0 -> 657 segments, output written (peak 0.95);
    * simulate stretch=1.0 -> 657 segments, output written;
    * boundary stretch=1000 (== SPECTRAL_MAX_STRETCH) accepted, exit 0.
- new fast-fail behaviour (over-cap stretch):
    * desktop  stretch=2000 -> "Error: stretch: expected finite float in (0, 1000],
      got 2000" with NO analysis/synthesis run, exit 1;
    * simulate stretch=2000 -> same fast rejection, exit 1.
  Host binaries are not byte-identical (spectral_cli.c is host-compiled and
  changed), so correctness is established by functional parity for all valid
  inputs plus the verified boundary/over-cap behaviour above.
```

## Scope (Phase C increment)

CLI / orchestration cluster, one defect fixed: the CLI entry validator now rejects
`stretch > SPECTRAL_MAX_STRETCH` at the boundary, matching the documented contract
constant, every downstream stretch consumer, and the sibling programmatic
validator — replacing a late, misleading synthesis failure (after a wasted
analysis pass) with an immediate, clear boundary rejection. Behaviourally inert for
every in-contract stretch (0 < stretch <= 1000). With this increment the Phase C
sweep has cleared fixed-point (161), analysis/peak-track (162), port/SIMD/out
(163), hashing/parsing/path (164), DSP-math/FFT-scaling + alloc/cache (165),
synth-backends + analysis-orchestration (166), and the CLI/orchestration layer
(167; synth_internal/wavetable/audio-I/O also audited clean this increment). Phase
D (compiled harness + LUT golden-vector loop) follows.
