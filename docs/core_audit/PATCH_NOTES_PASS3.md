# Core audit pass 3: fused-frame pair contract

## Summary

This patch fixes a concrete fused-analysis correctness issue and adds
guardrails. It does not claim that the fused implementation is final; the next
optimization pass should first introduce a real C-level parity executable that
runs full and fused analysis over the same synthetic inputs and compares emitted
segments within explicit tolerances.

## Modified files

```text
spectral_engine/analysis/spectral_analysis_fused.c
tools/core_audit/core_static_audit.py
spectral_engine/core/spectral_osc_formulas.h
spectral_engine/core/oscillator.c
```

## Added files

```text
docs/core_audit/PATCH_NOTES_PASS3.md
tests/core_math/test_core_pass3_static.py
```

## Changes

1. Replaced fused analysis chunk logic with explicit adjacent-frame-pair processing.
2. Added allocation-failure detection to fused max-discovery pass.
3. Removed ambiguous `row_prev` / `phase_prev` state from fused analysis.
4. Removed the unused Metal shader `norm` variable.
5. Updated static audit coverage for merge-regression invariants.
6. Replaced stale `make parity-test` comment with the actual test command family.
