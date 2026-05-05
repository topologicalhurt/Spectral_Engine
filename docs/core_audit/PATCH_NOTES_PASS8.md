# Core audit pass 8: STFT window registry and magnitude calibration

## Intuition

A window is not just a smoothing decoration before the FFT. It changes both
leakage and the amplitude measured by each frequency bin. For a real sinusoid
exactly centered on an interior positive-frequency FFT bin, an unnormalized
forward FFT produces approximately:

```text
raw_positive_bin_magnitude = sinusoid_peak_amplitude * sum(window) / 2
```

The factor of `1/2` is because an interior real sinusoid has paired positive
and negative frequency components. DC and Nyquist are not paired in the same
way, so endpoint bins use:

```text
raw_endpoint_magnitude = sinusoid_peak_amplitude * sum(window)
```

## Changes

- Window generation is routed through a descriptor table so built-in and future
  windows share the same registry shape.
- macOS vDSP Hann generation uses `vDSP_HANN_DENORM`, matching the conventional
  portable Hann implementation and the public header contract.
- Window metrics now expose sum, energy, coherent gain, RMS gain, ENBW, and
  separate endpoint/interior-bin amplitude and magnitude-squared scales.
- FFT resources carry separate `endpoint_bin_magsq_scale` and
  `positive_bin_magsq_scale` values.
- FFT frame extraction applies endpoint scaling only to DC/Nyquist and positive
  scaling to trackable interior bins.
- The returned frame max is recomputed from scaled interior trackable bins,
  avoiding endpoint-only threshold inflation.

## Why this is intentionally narrow

This does not implement full overlap-add reconstruction theory, phase-vocoder
time-scale modification, off-bin peak correction, or residual/noise modeling.
It only gives the current sinusoidal tracker consistent amplitude units and a
clear extension point for future windows.
