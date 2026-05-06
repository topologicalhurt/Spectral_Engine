# Core audit pass 18: validated peak-model profile

## Why this pass exists

Window choice, frequency interpolation, amplitude peak-height estimation, phase
diagnostics and temporal slope are coupled. Exposing them as independent knobs
creates a Cartesian product of invalid DSP states.

Pass 18 introduces a validated peak-model profile so those coupled choices are
resolved together.

## New files

```text
spectral_engine/analysis/spectral_peak_model.h
spectral_engine/analysis/spectral_peak_model.c
```

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

## Validation examples

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

## Tracker behavior

The tracker now has one resolved peak model. Existing setters remain, but they
mutate and re-resolve the model instead of leaving independent fields in an
invalid combination.
