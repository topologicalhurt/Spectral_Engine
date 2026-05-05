# Core audit pass 8: STFT window and magnitude calibration

## Intuition

A window is not just a smoothing decoration before the FFT. It changes the
amplitude measured by each frequency bin. For a real sinusoid exactly centered
on a positive FFT bin, an unnormalized forward FFT produces a positive-bin
magnitude of approximately:

```text
raw_magnitude = sinusoid_peak_amplitude * sum(window) / 2
```

So if the tracker later interprets `sqrt(magsq)` as an oscillator amplitude,
the magnitude-squared values should first be multiplied by:

```text
(2 / sum(window))^2
```

This pass makes that contract explicit.

## Changes

- macOS vDSP Hann generation now uses `vDSP_HANN_DENORM`, matching the
  conventional portable Hann implementation and the public header contract.
- Window calibration helpers now expose sum, energy, coherent gain, RMS gain,
  and positive-bin magnitude-squared scale.
- FFT resources now carry an explicit `magsq_scale`, defaulting to raw FFT
  magnitudes.
- Full and fused analysis paths set the positive-bin Hann calibration scale
  after generating the window.
- FFT frame extraction applies the scale to every magnitude-squared row and to
  the returned frame maximum.

## Why this is intentionally narrow

This does not yet implement full overlap-add reconstruction theory or phase
vocoder time-scale modification. It only makes the existing sinusoidal tracker
amplitude interpretation less arbitrary and backend-consistent.
