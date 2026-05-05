# Core audit pass 16: phase-consistency policy

## Intuition

Pass 15 made phase advance measurable. Pass 16 makes it controllable.

There are now three policies:

```text
IGNORE              do not compute phase diagnostics
OBSERVE             compute diagnostics, never suppress a segment
REJECT_INCONSISTENT compute diagnostics and reject segments whose phase
                   advance disagrees with the magnitude-derived model
```

The default is `OBSERVE`.

## Why default is not reject

Phase advance is powerful, but it is not always trustworthy. A bin can be
dominated by different partials in adjacent frames, especially near crossings
or in dense/noisy spectra. Rejecting by default would silently change audio
behavior. Observing by default gives diagnostics without changing output.

## What this pass changes

- Adds `SpectralPeakPhasePolicy`.
- Adds `phase_policy` to `SpectralPeakEstimateInput`.
- Adds `spectral_tracker_set_phase_policy()`.
- Initializes trackers with `SPECTRAL_PEAK_PHASE_POLICY_DEFAULT`.
- Wires tracker phase policy into estimator input.
- Keeps default policy as `OBSERVE`.
- Adds a C-backed test proving:
  - IGNORE does not set phase flags;
  - OBSERVE reports inconsistency but accepts;
  - REJECT_INCONSISTENT rejects only inconsistent phase models.
