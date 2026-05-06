# Core audit pass 18: validated peak-model profile

## Why this pass exists

Window choice, frequency interpolation, amplitude peak-height estimation, phase
diagnostics and temporal slope are coupled. Exposing them as independent knobs
creates a Cartesian product of invalid DSP states.

Pass 18 introduces a validated peak-model profile so those coupled choices are
resolved together before the tracker hot path sees callbacks or policies.

## Core model

```text
SpectralPeakModel
    window descriptor
    frequency estimator
    phase policy
    amplitude policy
    capabilities
    assumptions

SpectralResolvedPeakModel
    resolved callbacks
    resolved estimator
    resolved policies
    capability/assumption masks
```

The default model remains Hann/AUTO with log-power parabolic frequency
interpolation and bounded log-parabolic peak height. Rectangular windows default
to center-bin amplitude because the rectangular Dirichlet/sinc main lobe is not
modeled as a log parabola here. Custom descriptors that provide only
`interp_magsq` are valid, but they default to center-bin amplitude; an explicit
`INTERP_BOUNDED` policy requires a `peak_magsq` callback.

## Validation and failure contract

The validator rejects:

```text
missing window descriptor
missing frequency interpolation callback
unknown estimator
unknown phase policy
unknown amplitude policy
INTERP_BOUNDED without a peak-height callback
rectangular + log-parabolic peak-height callback
```

`spectral_tracker_set_peak_model()` resolves into a temporary model and only
commits on success. Invalid mutations return an error and preserve the previous
resolved profile, so a configured Hamming, Blackman, rectangular, or custom
tracker cannot silently revert to Hann. The legacy void setters keep API
compatibility and intentionally ignore the error, but they also leave the prior
valid profile untouched on failure.

`spectral_track_peaks()` remains a Hann/AUTO compatibility wrapper.
`spectral_track_peaks_with_window_descriptor()` constructs one explicit
`SpectralPeakModel`; invalid descriptors or estimator policies fail closed and
return an empty `SegmentArray` instead of tracking with a mismatched Hann model.

## Source basis

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
