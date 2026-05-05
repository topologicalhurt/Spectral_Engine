# Pass 8: STFT window and magnitude calibration contract

## What this stage does

Analysis turns audio samples into a sequence of spectra:

```text
audio frame -> multiply by window -> FFT -> magnitude^2 + phase -> peak tracker
```

The window reduces spectral leakage, but it also reduces the amplitude measured
by the FFT. A Hann window is near zero at the boundaries and near one in the
middle, so the FFT sees less total signal than it would with a rectangular
window.

## The key intuition

A bin-centered real sinusoid has two symmetric complex DFT peaks: one at the
positive frequency and one at the negative frequency. Because the engine tracks
positive-frequency bins only, each tracked bin carries roughly half the
two-sided sinusoid amplitude.

For an unnormalized DFT:

```text
positive_bin_magnitude ~= A * sum(window) / 2
```

where `A` is the sinusoid peak amplitude.

Therefore:

```text
A ~= positive_bin_magnitude * 2 / sum(window)
```

and for magnitude-squared:

```text
A^2 ~= positive_bin_magsq * (2 / sum(window))^2
```

## What was risky before

The tracker creates segment amplitudes from `sqrt(magsq)`. Without calibration,
those amplitudes scale with FFT size and window sum rather than signal
amplitude. This is not just a display issue: exported segments and downstream
kernel callers inherit arbitrary amplitude units.

There was also a backend-consistency risk: the public window header says the
window generators produce conventional unnormalized sample-domain windows, but
the macOS Hann path requested `vDSP_HANN_NORM`. That means macOS and non-macOS
builds could disagree before peak tracking begins.

## What changed

The code now exposes explicit window gain helpers and applies the positive-bin
magnitude-squared scale in both full-matrix and fused analysis.

## What this does not solve yet

- off-bin amplitude bias from leakage;
- peak-amplitude correction based on interpolated bin offset;
- phase-vocoder reconstruction constraints;
- stochastic/noise residual modeling.

Those belong in the later peak-model and reconstruction passes.
