# Core audit pass 10: window-aware peak interpolation

## Intuition

A spectral peak is not shaped only by the sinusoid. It is shaped by the analysis
window. Hann, Hamming, Blackman and future windows have different main-lobe
curvature and sidelobe behavior, so the rule that estimates a sub-bin offset
from three FFT bins belongs to the window contract.

Pass 8 made windows descriptor-based. Pass 10 wires that descriptor into the
peak tracker.

## Estimator contract and sources

The default callback is log-power parabolic interpolation over three adjacent
magnitude-squared STFT bins:

```text
p = 0.5 * (log(left) - log(right)) /
    (log(left) - 2*log(center) + log(right))
```

Inputs are finite, non-negative magnitude-squared bins from the current frame.
The output `p` is a dimensionless bin offset clamped to `[-0.5, 0.5]`; emitted
`omega` remains radians/sample through `(bin + p) * bin_step`. This is a local
quadratic approximation, not an exact maximum-likelihood estimator. It is used
as the conservative default because the engine currently emits Hann-windowed
magnitude-squared rows; complex estimators such as Quinn, Jacobsen/Kootsookos
and Candan have stronger assumptions about complex DFT coefficients and are not
silently substituted for this signal model.

The optimized implementation evaluates the equivalent two-log ratio form,
`a = log(left / center)`, `b = log(right / center)`,
`p = 0.5 * (a - b) / (a + b)`, with a three-log fallback for extreme finite
ratios. This preserves the estimator contract while removing one logarithm on
the normal hot path. The logarithm call is routed through the shared
`fast_peak_log()` utility so the exact default and any opt-in approximation are
tested in one place.

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

- `SpectralTracker` now stores a `SpectralWindowInterpMagsqFn`.
- The tracker initializes to the safe default parabolic interpolation callback.
- Single-shot and fused tracking bind the default Hann descriptor; raw callers
  use `spectral_track_peaks_with_window_descriptor()` to bind non-Hann STFT rows.
- Segment emission calls the active callback instead of hard-coding
  `spectral_window_interp_magsq_parabolic`.
- Candidate validation rejects non-finite or negative magnitude-squared
  neighborhoods before sqrt/interpolation.
- Interpolated offsets are finite-guarded and clamped before frequency emission.

## What this does not solve yet

This does not yet make Quinn, Jacobsen/Kootsookos, Candan, or full phase-based
frequency estimators the default. It only makes the current estimator
architecturally honest: interpolation is a window-dependent policy, not a
global constant.
