# Patch Notes — Interim Pass 3

## Modified files

```text
spectral_engine/analysis/spectral_analysis_fused.c
tools/core_audit/core_static_audit.py
spectral_engine/core/spectral_osc_formulas.h
spectral_engine/core/oscillator.c
```

## Added files

```text
docs/core_audit/INTERIM_PASS3_FINDINGS.md
docs/core_audit/PATCH_NOTES_INTERIM_PASS3.md
docs/core_audit/HANDOFF_INTERIM_PASS3.md
tests/core_math/test_core_interim_pass3_static.py
```

## Source-level changes

1. Replaced fused analysis chunk logic with explicit adjacent-frame-pair processing.
2. Added allocation-failure detection to fused max-discovery pass.
3. Removed ambiguous `row_prev` / `phase_prev` state from fused analysis.
4. Removed the unused Metal shader `norm` variable.
5. Updated static audit coverage for merge-regression invariants.
6. Replaced stale `make parity-test` comment with the actual test command family.

## Why this is an interim pass

This patch fixes a concrete correctness issue and adds guardrails, but does not claim that the fused implementation is final. The next optimization pass should first introduce a real C-level parity executable that runs full and fused analysis over the same synthetic inputs and compares emitted segments within explicit tolerances.
