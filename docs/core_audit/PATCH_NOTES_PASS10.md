# Core audit pass 10: window-aware peak interpolation

## Intuition

A spectral peak is not shaped only by the sinusoid. It is shaped by the analysis
window. Hann, Hamming, Blackman and future windows have different main-lobe
curvature and sidelobe behavior, so the rule that estimates a sub-bin offset
from three FFT bins belongs to the window contract.

Pass 8 made windows descriptor-based. Pass 10 wires that descriptor into the
peak tracker.

## Changes

- `SpectralTracker` now stores a `SpectralWindowInterpMagsqFn`.
- The tracker initializes to the safe default parabolic interpolation callback.
- Single-shot and fused tracking bind the default Hann descriptor.
- Segment emission calls the active callback instead of hard-coding
  `spectral_window_interp_magsq_parabolic`.
- Candidate validation rejects non-finite or negative magnitude-squared
  neighborhoods before sqrt/interpolation.
- Interpolated offsets are finite-guarded and clamped before frequency emission.

## What this does not solve yet

This does not yet implement Quinn, Jacobsen/Kootsookos, Candan, or full
phase-based frequency estimators. It only makes the current estimator
architecturally honest: interpolation is a window-dependent policy, not a
global constant.
