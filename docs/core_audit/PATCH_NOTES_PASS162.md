# Patch notes — Pass 162: CTF sweep increment 2 — analysis/peak-track tracker cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass clears the
**analysis / peak-track cluster** — `analysis/spectral_peak_track.c`, the
incremental `SpectralTracker` API and the fused (chunked) analysis driver it
feeds. This path is HOST-only (FFT via vDSP/FFTW3) and the compiled CTest harness
does not exercise it, so defects here are latent-by-design — none of the four
CTests touch tracker create/run, which is exactly why the most severe defect
(an inverted return polarity that silently empties large-input analysis) survived.

## Change

Three defects found and fixed:

```text
1. Memory leak on out-of-range thread count
   analysis/spectral_peak_track.c  (spectral_tracker_create)
   The validation `if (n_threads > SPECTRAL_MAX_THREADS) return NULL;` ran AFTER
   `malloc(sizeof(SpectralTracker))`, so an over-range n_threads leaked the just-
   allocated tracker. Fix: hoist the clamp/reject (n_threads < 1 -> 1;
   > SPECTRAL_MAX_THREADS -> return NULL) ABOVE the malloc so the early return
   frees nothing because nothing is allocated yet. Behaviour-identical for the
   in-range path.

2. Unclamped thread count feeding tracker_create
   analysis/spectral_peak_track.c  (spectral_track_peaks_with_window_descriptor)
   `int n_threads = omp_get_max_threads();` could exceed SPECTRAL_MAX_THREADS (256)
   on a many-core host, which made spectral_tracker_create reject (return NULL)
   and previously leak (defect 1). Fix: use spectral_omp_effective_thread_count()
   (core/spectral_omp.h), which clamps to [1, SPECTRAL_MAX_THREADS] — the SAME
   helper the FFT-resource and fused paths already use, so the three analysis
   entry points are now consistent.

3. Inverted return polarity in the fused per-frame driver  (SEVERE)
   analysis/spectral_peak_track.c  (spectral_tracker_run_fused_frame)
   `local_failed` is 0 on success / 1 on failure throughout the function, but the
   function ended `return local_failed;` — i.e. it returned FALSE on success and
   TRUE on failure. Its single caller (spectral_analysis_fused.c:221) is
   success-truthy: `if (!spectral_tracker_run_fused_frame(...)) { set_failed;
   break; }` (with the comment "follows tracker helper polarity"). So on a
   SUCCESSFUL frame the caller saw 0, declared failure, set the tracker error and
   broke out of the chunk loop -> the entire fused (large-input) analysis returned
   empty segments and never reported an error. Fix: `return !local_failed;` to
   match the success-truthy convention of every sibling helper (queue_candidate /
   flush_candidate_batch / process_bitmask all return 1=success, 0=failure).
```

The fused path is selected only when `total_bins > SPECTRAL_STFT_CHUNK_THRESHOLD`
(32M bins, ~256MB STFT), so defect 3 silently degraded only very large inputs —
small/medium renders use the non-fused path and were unaffected, which is why it
was never noticed.

## Finding

Audited and left unchanged (no defect) — the rest of the analysis cluster is
Campaign-1-hardened numeric code:
- `spectral_peak_estimator.c` — every interpolator (Jacobsen/Candan/Quinn/
  log-parabolic/mag-parabolic) guards each division with a denominator epsilon
  and does every float->int in double with finite/range checks
  (`spectral_peak_store_clamped_offset_d`, `spectral_peak_load_magsq_triplet`
  bounds `center_bin + 1u >= n_freqs`).
- `spectral_peak_interp.c` — `validate_candidate` reads `row[cf±1]` but callers
  guarantee `cf in [1, n_freqs-2]`; `emit_segment` checks `cf + 1u >= n_freqs`
  and uses the overflow-checked realloc path.
- `spectral_analysis_fft.c` — FFT resource alloc/free, single-frame transform
  (vDSP + FFTW variants), magsq scaling; all bounds/overflow correct.
- `spectral_analysis.c` / `_full.c` / `_internal.h` — overflow-hardened; the
  window_ctx-freed-before-descriptor-read in `_full.c` is a non-defect (fallback
  yields the identical hardcoded HANN descriptor).
- `spectral_peak_model.c`, `spectral_peak_track_internal.h`,
  `spectral_processing_chain.c` (mask parse/format, snprintf-truncation guarded),
  and the `proc_{serra_smith_1990,johnston_1988,adaptive_track_density}.c` no-op
  stubs — all clean.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign
  -mno-avx512f / -mavx2 unused-arg notes on host).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- NOTE on coverage: the analysis/peak-track path is HOST-only and is NOT exercised
  by any CTest, so these fixes are verified by build-clean + unchanged ctest, with
  the behaviour change (defect 3) documented here. The embedded targets do not
  compile the analysis path at all; defects 1-3 cannot affect any embedded binary.
- desktop float render is unaffected for small/medium inputs (non-fused path
  untouched); defect 3's fix restores correct output for >256MB-STFT inputs that
  previously returned empty.
```

## Scope (Phase C increment)

Analysis / peak-track tracker cluster only. Defects 1-2 are host-robustness
(leak + many-core consistency); defect 3 is a correctness fix for the large-input
fused path. No change to the non-fused analysis path or any embedded binary.
Next CTF cluster per ULTRAPLAN Phase C: port/SIMD/out
(`core/port/{host,embedded}/*.c` — out kernels, oscillator, windows, lut,
envelope).
