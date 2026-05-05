# Core audit pass 15: phase-advance diagnostics

## Intuition

Magnitude peak tracking estimates where the spectral peak is. Phase advance
estimates how fast the phase actually moved between adjacent frames.

For one STFT bin, the phase-vocoder relation is:

```text
residual = princarg(phase[t+1,k] - phase[t,k] - center_bin_omega * hop)
phase_bin_offset = residual / (bin_step_omega * hop)
```

This gives an independent diagnostic for instantaneous frequency.

## What changed

Pass 15 adds phase-advance diagnostics to `SpectralPeakEstimate`:

```text
phase_bin_offset
phase_omega
phase_error
```

and flags:

```text
SPECTRAL_PEAK_ESTIMATE_PHASE_ADVANCE_VALID
SPECTRAL_PEAK_ESTIMATE_PHASE_MODEL_CONSISTENT
```

It also wires `next_phase_row` through:

```text
fused analysis -> SpectralFrameContext -> tracker -> estimator
full/incremental tracker internal pairs -> estimator
```

For chunk-boundary overlap rows, there is still no overlap phase row in the
public incremental API, so phase diagnostics are skipped at that boundary.

## What does not change

This pass does **not** use phase advance to override `omega` or `df`. It only
measures whether the magnitude-derived oscillator model agrees with adjacent
frame phase motion.

That is intentional. Phase correction is powerful but can be wrong for partial
crossings, noisy bins, and windowed mixtures. We need diagnostics first.
