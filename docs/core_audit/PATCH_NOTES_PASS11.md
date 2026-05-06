# Core audit pass 11: peak estimator module

## Intuition

A detected FFT-bin maximum is not the true sinusoid frequency. The real peak
usually sits between bins, so the tracker estimates a sub-bin offset and then
emits:

```text
omega = (bin + offset) * bin_step
```

The previous code computed that offset directly inside `spectral_peak_interp.c`.
That made interpolation a hard-coded detail of segment emission.

## Estimator contract

`SpectralPeakEstimateInput` receives the current-frame magnitude-squared row,
the current-frame phase row, the detected bin, the next-frame maximum used for
amplitude slope, and the selected estimator policy. The default `AUTO` policy
resolves to log-power parabolic interpolation:

```text
p = 0.5 * (log(left) - log(right)) /
    (log(left) - 2*log(center) + log(right))
```

`left`, `center` and `right` are finite, non-negative adjacent
magnitude-squared bins from the current frame. The output `p` is a bin offset
clamped to `[-0.5, 0.5]`; `omega` is radians/sample and `df` is the
radians/sample^2 contribution used by downstream quadratic phase. Contract
tests bound Hann-windowed log-parabolic bias to `0.02` bins over synthesized
interior-bin single tones with offsets in `[-0.49, 0.49]`.

The implementation uses the exact two-log ratio form of the same expression:

```text
a = log(left / center)
b = log(right / center)
p = 0.5 * (a - b) / (a + b)
```

This removes one `logf` call in the default estimator without changing the
reference formula for finite, non-extreme triplets. Extreme finite ratios fall
back to the three-log form to avoid overflow in `left / center` or
`right / center`.

The complex estimators are available only as explicit policies. They use
reconstructed complex bins from magnitude and phase and may fall back to the
baseline when their finite-data or denominator assumptions fail. They are not
the default because the current engine signal model is Hann-windowed
magnitude-squared tracking, while Jacobsen/Kootsookos, Candan and Quinn are
defined for specific complex DFT coefficient assumptions.

Source basis:

- F. J. Harris, "On the Use of Windows for Harmonic Analysis with the Discrete
  Fourier Transform," Proc. IEEE, 1978. https://doi.org/10.1109/PROC.1978.10837
- J. O. Smith, "Quadratic Interpolation of Spectral Peaks," Spectral Audio
  Signal Processing.
  https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html
- B. G. Quinn, "Estimating Frequency by Interpolation Using Fourier
  Coefficients," IEEE Trans. Signal Processing, 1994.
  https://doi.org/10.1109/78.295186
- E. Jacobsen and P. Kootsookos, "Fast, Accurate Frequency Estimators," IEEE
  Signal Processing Magazine, 2007. https://doi.org/10.1109/MSP.2007.361611
- C. Candan, "A Method for Fine Resolution Frequency Estimation from Three DFT
  Samples," IEEE Signal Processing Letters, 2011.
  https://doi.org/10.1109/LSP.2011.2136378
- D. C. Rife and R. R. Boorstyn, "Single Tone Parameter Estimation from
  Discrete-Time Observations," IEEE Trans. Information Theory, 1974.
  https://doi.org/10.1109/TIT.1974.1055282

## Changes

- Adds `spectral_peak_estimator.h/.c`.
- Adds explicit `SpectralPeakEstimateInput` and `SpectralPeakEstimate` structs.
- Moves amplitude, amplitude slope, omega and df emission math into the
  estimator module.
- Adds estimator candidates:
  - window/log-power parabolic baseline;
  - magnitude parabolic diagnostic;
  - Jacobsen/Kootsookos complex estimator;
  - Candan-corrected complex estimator;
  - Quinn second estimator.
- Keeps `AUTO` resolving to log-power parabolic for the current Hann-windowed
  magnitude pipeline.
- Adds explicit fallback from advanced complex estimators to the baseline when
  their assumptions/data fail.
- Wires tracker configuration through `spectral_tracker_set_peak_estimator`.
- Adds `spectral_track_peaks_with_window_descriptor` so raw STFT callers can
  bind the analysis window and estimator instead of inheriting Hann/AUTO
  silently.
- Later peak-model validation resolves the window, estimator, phase policy and
  amplitude policy as one contract; invalid explicit raw-tracker profiles fail
  closed rather than falling back to Hann.
- Adds exact-mode performance cleanup:
  - Candan's `tan(pi / n_fft) / (pi / n_fft)` correction is precomputed once
    per tracker;
  - complex estimators share triplet reconstruction;
  - magnitude parabolic and complex estimators reuse the center-bin magnitude
    for emitted amplitude;
  - the tracker calls `spectral_peak_estimate_validated()` after candidate
    validation to avoid duplicate finite/non-negative triplet checks.
- Adds opt-in approximation gates. `SPECTRAL_ENABLE_APPROX_PEAK_LOG` applies
  only to peak-estimator logarithms and is implemented in the shared
  `fast_peak_log()` utility. `SPECTRAL_ENABLE_APPROX_TRIG` and
  `SPECTRAL_ENABLE_APPROX_INV_SQRT` are reused through `fast_sin()` and
  `fast_sqrt()` for phase-to-complex and magnitude extraction. Defaults remain
  exact; estimator modules must not carry local copies of those approximations.
  The `fast_peak_log()` approximation is not a library copy: it is the standard
  binary32 range reduction `x = 2^e * m`, followed by
  `ln(m) = 2*atanh((m - 1)/(m + 1))` and an odd atanh power series truncated
  after `z^11/11`. The code comment cites the IEEE-754 binary format basis and
  the NIST DLMF atanh identity/series entries.
- Adds `tools/core_audit/peak_estimator_bench.c`, reporting p50/p95 time,
  counter ticks where available, max emitted offset/amplitude error, and
  fallback rate for every estimator mode.

## Why the default stays conservative

The complex estimators are academically strong for isolated tones under their
DFT assumptions, especially rectangular/unwindowed complex DFT settings. The
current engine is Hann-windowed and tracks calibrated magnitude-squared rows.
Changing the default now would be a silent model change. This pass makes the
estimators available and auditable without pretending a new default has been
proven on the engine's actual signal model.
