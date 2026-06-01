# Patch notes — Pass 177: CTF sweep increment 17 — STFT analysis FFT driver + orchestration cluster (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **STFT analysis
FFT driver and its orchestration cluster** — the layer that turns windowed audio
frames into the one-sided magnitude-squared / phase matrix consumed by the peak
tracker, plus the path-decision and (no-op) processing-chain dispatch:

```text
- analysis/spectral_analysis_fft.c        vDSP + FFTW frame transform, one-sided
                                          magsq scaling, vDSP 0.5 rescale, phase,
                                          OMP reduction(max) frame loop, alloc/free
- analysis/spectral_analysis.c            shape/frame-count, path decision, scale wiring
- analysis/spectral_analysis_full.c       full-matrix path
- analysis/spectral_analysis_fused.c      two-pass concurrent fused FFT+track path
- analysis/spectral_processing_chain.c    process-mask parse + stage dispatch
- analysis/spectral_peak_model.c          peak-model policy/capability resolution
- analysis/spectral_proc_*_apply          johnston_1988 / serra_smith_1990 /
                                          adaptive_track_density (no-op stubs)
Consumer/producer contracts traced into:
- core/spectral_windows.c                 magsq-scale derivation (amp^2)
- core/port/host/spectral_vector_ops.c    spectral_magsq_only / _phase
- analysis/spectral_peak_track.c          threshsq dB->linear power math
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol
a clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

### One-sided magsq scaling is applied exactly once, identically on both backends

```text
- spectral_magsq_only / spectral_magsq_phase (FFTW path) and the vDSP path both
  compute RAW magsq = re^2+im^2 per bin and a raw per-frame max; the one-sided
  convention is then applied solely by spectral_fft_apply_magsq_scales, which
  overwrites frame_max with spectral_fft_trackable_magsq_max (interior bins only).
  So the doubling is never double-counted, and both backends funnel through the
  same scale application.
- apply_magsq_scales: DC[0] and Nyquist[n_freqs-1] take endpoint_scale; interior
  [1 .. n_freqs-2] take positive_scale. Matches SciPy one-sided periodogram (only
  interior bins doubled) and FFTW r2c endpoint packing.
- Scale derivation (spectral_window_metrics): positive_bin_amp_scale = 2/Σwindow,
  endpoint_bin_amp_scale = 1/Σwindow; the *magsq* scales are the SQUARES
  ((2/Σ)^2 vs (1/Σ)^2 — a 4:1 ratio, not 2:1). This is the amplitude-recovery
  convention: sqrt(scaled_magsq) = amp_scale*|X[k]| = A, consistent with the
  estimator's magsq->amp path. Every scale only takes effect behind a
  finite-&-positive guard (spectral_fft_magsq_scale_valid); bad scales fall back
  to 1.0.
```

### vDSP packed-real path — the 0.5 rescale and DC/Nyquist handling are exact

```text
- vDSP_fft_zrip(FFT_FORWARD) emits 2x the textbook DFT uniformly across realp[0]
  (DC), imagp[0] (Nyquist) and interior bins; the explicit ×0.5 on realp and imagp
  (n_fft/2 each) undoes it so |X[k]| matches the FFTW r2c branch and the 2/Σ amp
  scales (which assume the unscaled DFT). ×0.5 is uniform on re+im => phases
  unaffected, sign preserved.
- magsq layout: out_magsq[0]=realp[0]^2 (DC), out_magsq[n_freqs-1]=imagp[0]^2
  (Nyquist), interior via vDSP_vsq(realp+1)/vDSP_vsq(imagp+1)/vDSP_vadd over
  mid=n_freqs-2 elements. n_fft>=64 => n_freqs>=33 => mid>=31 (no underflow);
  realp/imagp have n_fft/2 elements and the loads reach index n_fft/2-1 (in
  bounds). All n_freqs bins written exactly once.
- phases: out_phases[0]=(realp[0]>=0)?0:π, out_phases[n_freqs-1]=(imagp[0]>=0)?0:π
  (DC & Nyquist of a real signal are real), interior via
  vvatan2f(out, imagp+1, realp+1, mid) => atan2(imag, real) — correct arg order.
  Equivalent to the FFTW branch's atan2f(im, re) for the same data.
```

### Frame geometry & bounds

```text
- spectral_analysis_shape_init: n_frames = (n_samples - n_fft)/hop + 1 with
  n_samples>=n_fft validated; last frame start = floor((n_samples-n_fft)/hop)*hop
  <= n_samples-n_fft, so src=audio + (n_frames-1)*hop spans [.. +n_fft) in bounds.
  n_fft power-of-two re-checked; n_freqs=n_fft/2+1 with n_freqs>=3 gate; total_bins
  via spectral_size_mul (overflow-checked).
- spectral_fft_frames: OMP reduction(max:max_magsq) with a private per-thread
  max + #pragma omp for schedule(static); each i writes out_magsq+i*n_freqs /
  out_phases+i*n_freqs (disjoint). output_bins = n_frames*n_freqs overflow-checked.
```

### Fused two-pass path concurrency & frame-pair contract

```text
- Pass 1 (max discovery) and Pass 2 both run num_threads(actual_threads); the
  tracker is created with actual_threads slots, so omp_get_thread_num() < team
  size <= actual_threads => per-thread (tid-indexed) storage never overruns even
  if the runtime grants fewer threads.
- Per-chunk frame-pair contract verified by induction: each chunk primes
  row_curr=FFT(pair_start), then for pair in [pair_start,pair_end): computes
  row_next=FFT(pair+1), processes (curr=FFT(pair), next=FFT(pair+1)) with frame
  index = pair, then rotates. So row=magsq[pair], next_row=magsq[pair+1],
  t_hop=pair*hop — identical to the full-matrix pair semantics. The trailing extra
  rotate on the last pair is benign (row_curr is re-primed next chunk).
- Scratch rows + candidate_batch are per-thread (declared inside the parallel
  region), each n_freqs floats, single-writer; freed per thread. fft_time_total is
  accumulated with #pragma omp atomic update; global_max via reduction(max).
- The pass-1 max scans interior bins of ALL frames [0,n_frames); pass-2 sees every
  frame 0..n_frames-1 as some pair's curr/next, so the global-max-derived threshold
  matches the full path. n_frames<2 -> empty (no pairs, no segments — same as full).
```

### dB threshold — power convention (the one factor that could hide a real bug)

```text
spectral_tracker_derive_create_scalars:
  thresh_linear_sq = pow(10, db_thresh/10);  threshsq = thresh_linear_sq*max_magsq
Because magsq IS power (magnitude squared), the dB->linear divisor is 10
(P/Pref = 10^(dB/10)), NOT 20 (which would be the amplitude convention). Using
/20 here would be a classic latent DSP bug; the code is correct. All products in
double with finite / >=0 / <=FLT_MAX guards.
```

### Parse / dispatch / policy (no DSP math, scanned for memory hazards)

```text
- spectral_process_mask_parse: numeric path strtoul(base 0) then reject any bit
  outside ALL_KNOWN (so an overflow to ULONG_MAX is caught by the mask test);
  token path mallocs n+1, strtok_r, rejects "none"+other combinations.
- spectral_process_mask_to_string: snprintf truncation accounting breaks before
  out+used can go out of bounds; size arg clamped to 0 when used>=out_size.
- spectral_peak_model_validate/_resolve: pure policy gating (INTERP_BOUNDED
  requires a peak_magsq callback; rectangular must not inherit the log-parabolic
  peak-height model); no allocation or arithmetic hazards.
- proc_johnston_1988 / serra_smith_1990 / adaptive_track_density: no-op stubs
  returning SPECTRAL_OK; unimplemented requested bits surface via report.pending.
```

## Verification

```text
- No source changed this pass (read-only audit), so the Pass 176 green state is
  preserved by construction (host binaries byte-identical). Re-ran the full triad
  to formally close the pass:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts,
      core_guarantees, core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared fixed-point (161), analysis/peak-track
scan (162), port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling +
alloc/cache (165), synth-backends + analysis-orchestration (166), CLI/orchestration
(167), embedded fade envelope (168), core synth dispatch/internal helpers (169),
binary-deserialization/converter (170), host GPU-tile concurrency (171), the
oscillator asin domain guard (172), the host SIMD quantized domain guard (173), the
file-I/O + CLI untrusted-input boundary cluster (174, clean), the peak
frequency-estimation cluster (175, clean), the SpectralTracker lifecycle/per-thread
storage/OpenMP-reduction cluster (176, clean), and the STFT analysis FFT driver +
orchestration cluster (177, clean — one-sided magsq scaling applied once across both
backends, the vDSP 0.5 rescale + DC/Nyquist packing, frame geometry/bounds, the
fused two-pass frame-pair contract and tid-indexed concurrency, and the threshsq
power-dB /10 convention all verified). Phase D (compiled harness + LUT golden-vector
loop) follows.
