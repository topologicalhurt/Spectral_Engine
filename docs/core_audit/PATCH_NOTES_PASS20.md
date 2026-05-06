# Core audit pass 20: peak-model integration audit

## Summary

Pass 20 audits the new validated peak-model layer and hardens two invariants:

```text
1. phase diagnostics require both current and next phase rows
2. invalid model mutations must preserve the existing resolved profile
```

## Finding

`SpectralPeakModel` marked phase diagnostics with `NEXT_PHASE_ROW`, but not
`PHASE_ROW`. That made the model capability contract report no current-phase
row requirement for a log-parabolic + phase-observe model even though phase
diagnostics read both `phase_row` and `next_phase_row`.

## Fix

Phase diagnostic resolution now sets:

```text
SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW
SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW
SPECTRAL_PEAK_MODEL_CAP_PHASE_DIAGNOSTIC
```

and the API exposes one generic predicate:

```c
spectral_peak_model_has_capability(...)
```

Pass 20 deliberately does not add `requires_*` wrappers. Callers ask for the
capability they need directly, which avoids alias-style public functions that
only forward to another resolver.

## Transactional setter defense

The C-backed test now verifies all legacy void setters preserve the existing
custom profile on invalid mutations:

```text
spectral_tracker_set_window_descriptor
spectral_tracker_set_peak_estimator
spectral_tracker_set_phase_policy
spectral_tracker_set_amplitude_policy
```

This protects the audited invariant:

```text
valid model mutation   -> commit
invalid model mutation -> preserve existing model
```

## Why this matters

After Pass 18, `SpectralPeakModel` is the source of truth. Resolved tracker
fields exist only as hot-path aliases. They must not drift, and model helper
capabilities must accurately report the rows required by active diagnostics.
