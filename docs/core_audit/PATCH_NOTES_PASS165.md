# Patch notes — Pass 165: CTF sweep increment 5 — DSP-math / FFT-scaling cluster (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass sweeps the
**DSP-math / FFT-scaling cluster** — the analysis FFT front end
(`analysis/spectral_analysis_fft.c`), window generation + magnitude calibration
(`core/spectral_windows.c`), the peak estimators
(`analysis/spectral_peak_estimator.c` + `spectral_peak_interp.c` +
`spectral_peak_model.c`), the fused driver (`analysis/spectral_analysis_fused.c`),
fast-math (`core/spectral_fast_math.c`), the oscillator formula header
(`core/spectral_osc_formulas.h`), and the allocation/pool/cache cluster
(`core/spectral_seg_cache_fs.c`, `core/spectral_perf_model.{c,h}`,
`core/spectral_perf{,_embedded}.c`, `core/spectral_perf_accounting.h`).

The one real defect is a **backend-divergent FFT magnitude calibration**: on the
Apple/vDSP build the forward real FFT is scaled 2× relative to the textbook DFT,
and that 2× was never compensated — so vDSP/macOS analysis reported magnitudes
4× (≈ +6 dB) hotter in magsq than the FFTW/portable path, which is the path the
window amp scales (PASS8) were derived for and the path the contract CTests
exercise (`SPECTRAL_USE_VDSP=0`).

## Change

```text
1. Uncompensated vDSP forward-FFT 2x overscale  (backend calibration divergence)
   analysis/spectral_analysis_fft.c  (spectral_fft_single_frame, vDSP branch)
   vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD) emits the textbook DFT
   scaled by 2 — Apple's packed-real convention, uniform across DC (split.realp[0]),
   Nyquist (split.imagp[0]) and the interior bins (split.realp[1..]/imagp[1..]).
   The vDSP branch squared split.realp/imagp straight into out_magsq with NO
   compensation, while:
     (a) the window amp scales are derived for the UNSCALED textbook DFT
         (PASS8: positive_bin_amp_scale = 2/Sigma_window,
          endpoint_bin_amp_scale = 1/Sigma_window;
          raw_positive_bin_magnitude = amp * Sigma_window / 2), and
     (b) the FFTW/portable branch (same function, #else) feeds the unscaled fftwf
         forward output straight to spectral_magsq_{only,phase} — i.e. textbook —
         and that is the branch the core_contracts CTest validates (it forces
         SPECTRAL_USE_VDSP=0).
   Net: the same audio analyzed on macOS (vDSP) read 4x hotter in magsq (2x in
   magnitude, ~+6 dB) than on Linux (FFTW) and than the tested/contract path.
   Fix: immediately after vDSP_fft_zrip, scale the split by 0.5 over both realp
   and imagp (n_fft/2 elements each) — uniform across DC/Nyquist/interior, so the
   squared magnitudes drop by 0.25 to the textbook value and phase angles are
   untouched (uniform scale on re+im leaves atan2 unchanged). The FFTW branch is
   already textbook and is left unchanged.
```

## Why this is behaviourally inert on every observable output (yet still correct)

The fix changes the *raw analysis magnitudes* on the vDSP build to the textbook
value, but it is provably inert on selection, frequency and rendered audio because
every downstream consumer of magnitude is scale-relative or re-normalized:

- **Peak selection** — `spectral_tracker_derive_create_scalars`
  (`analysis/spectral_peak_track.c:688`) sets the gate to
  `threshsq = 10^(db_thresh/10) * max_magsq`, i.e. **relative to the per-analysis
  global max_magsq**. A uniform 0.25 scale multiplies both `max_magsq` and every
  bin's magsq equally, so the comparison is scale-invariant → identical peak set.
- **Detected frequency** — the parabolic / Jacobsen / Quinn interpolators all use
  magnitude *ratios* between adjacent bins → scale-invariant.
- **Rendered output** — the CLI normalizes to `SPECTRAL_NORMALIZE_HEADROOM = 0.95`
  (`cmd/cli/spectral_cli_pipeline.c:413` → `spectral_normalize_float`), and a
  global amplitude factor `k` cancels exactly: `0.95 * (k*x)/max(k*x) = 0.95*x/max(x)`.

So the *value* the engine reports internally is corrected (macOS now equals Linux
and the contract path), while segment count, detected pitch, and the rendered WAV
are unchanged. This removes a genuine cross-platform divergence — any current or
future consumer of an *absolute* magnitude/dB (golden vectors, a spectrogram dump,
an absolute-dB feature, the FFT magsq-scale contract) would otherwise read 6 dB
differently on macOS vs Linux.

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is correct:
- `core/spectral_windows.c` — Hann/Hamming/Blackman are the symmetric (N-1)
  forms; `spectral_window_interp_magsq_parabolic` is Smith's
  `p = 0.5(a-b)/(a+b)` with peak height `y0 - 0.25(y_{-1}-y_{1})p`;
  `spectral_window_peak_magsq_log_parabolic` is the log-domain parabola; the
  window metrics that produce the amp/magsq scales are backend-independent (they
  depend only on Σwindow), so they need no vDSP-vs-FFTW special case — the FFT
  scale convention is the only backend variable, now unified by defect 1.
- `analysis/spectral_peak_estimator.c` + `spectral_peak_interp.c` +
  `spectral_peak_model.c` — Jacobsen `Re{(X_{-1}-X_{+1})/(2X_0-X_{-1}-X_{+1})}`,
  Candan `tan(pi/N)/(pi/N)`, magnitude-parabolic `0.5(a-g)/(a-2b+g)`, Quinn-second
  `dp=-ap/(1-ap)`, `dm=am/(1-am)`, `offset=0.5(dp+dm)+tau(dp^2)-tau(dm^2)` with the
  tau kernel `0.25 log(3x^2+6x+1) - (sqrt6/24) log((x+1-sqrt(2/3))/(x+1+sqrt(2/3)))`,
  and the phase-vocoder instantaneous-frequency relation — all re-derived correct.
- `core/spectral_fast_math.c` — atan2 / inv_sqrt / sqrt / peak-log / sin are exact
  by default (approximations are flag-gated, validated by core_guarantees_drift);
  the atanh log series coefficients are 2/(2k+1).
- `core/spectral_osc_formulas.h` — all 8 waveforms, the Hann fade
  `0.5(1-cos(pi*j/L))`, the round-to-nearest phase reduction and the order-15
  Taylor sine reproduce their references term-for-term.
- `analysis/spectral_analysis_fused.c` — the fused single-pass driver matches the
  full-matrix path (validated structurally by the pass-119 fused-parity harness).
- Allocation/pool/cache — `core/spectral_seg_cache_fs.c`,
  `core/spectral_perf_model.{c,h}`, `core/spectral_perf.c`,
  `core/spectral_perf_embedded.c`, `core/spectral_perf_accounting.h`: cache key
  derivation, the perf/WCET accounting and the cycle model are all overflow-guarded
  and finite-validated; no defect.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float (only the pre-existing benign -mavx2 /
  -mno-avx512f unused-command-line-arg notes on host).
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift. (core_contracts forces SPECTRAL_USE_VDSP=0, i.e. it
  validates the textbook path the fix now makes vDSP agree with.)
- NOT byte-identical this pass (by design — the vDSP build's code changed). The
  effect is verified empirically on resources/testing/sin_440hz.wav:
    * desktop and simulate both detect the SAME 657 segments (peak selection
      unchanged — relative threshold);
    * desktop output peak = 0.95000 (-0.45 dBFS), simulate output peak = 0.95000
      (-0.45 dBFS) — identical (output normalized to 0.95 headroom);
    * RMS desktop -5.34 dBFS vs simulate -5.42 dBFS — within 0.08 dB, the
      float-vs-Q15 quantization gap, consistent with prior passes.
  => sim-vs-desktop parity intact; rendered output unchanged; the corrected
     calibration is the internal/raw magnitude, now equal to the FFTW/contract path.
```

## Scope (Phase C increment)

DSP-math / FFT-scaling cluster + allocation/pool/cache, one defect fixed: the
vDSP forward-FFT 2× overscale, bringing macOS/vDSP analysis magnitudes onto the
textbook DFT calibration that the FFTW/portable path and the contract CTests
already use. Behaviourally inert on selection / frequency / rendered output (all
scale-relative or re-normalized), so no regression; it closes a real
cross-backend magnitude divergence. With this increment the core/analysis/synth
CTF sweep has cleared the fixed-point (161), analysis/peak-track (162),
port/SIMD/out (163), hashing/parsing/path (164) and DSP-math/FFT-scaling +
alloc/cache (165) clusters. Remaining Phase-C surface per ULTRAPLAN: the synth
backends and the analysis orchestration layer
(`spectral_analysis.c` / `spectral_analysis_full.c`). Phase D follows the sweep.
