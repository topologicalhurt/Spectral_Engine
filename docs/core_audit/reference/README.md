# Reference docs — canon, guidelines, and not-currently-executing plans

`docs/core_audit/` (the parent) holds **only the actively-worked plan(s)**: `../REVIEWER_HANDOFF.md`
(the forward mandate) and the live workstream `../IFFT_SYNTHESIS_PLAN.md`.
Completed/superseded campaigns are in `../archive/`. Everything that is **live but not the
current focus** lives here.

## Canon / rules (the SSOT — update these in the same patchset as a contract change, per AI_CANON §18)
- `AI_CANON.md` — the 19 correctness rules (exact-by-default, approximations gated + parity-tested,
  SSOT for constants, no doc/test load-bearing on source text, capability-not-CPU gating).
- `CORE_CONTRACTS.md` — per-function contracts + the guarantee registry/manifest.
- `ACADEMIC_SOURCES.md` — paper-backed methods + citations.
- `DISCIPLINE_FINDINGS.md` — recurring discipline findings.
- `CHANGELOG.md` — the rolling change digest (one terse line per change).

## Guidelines / reference
- `KERNEL_PATCHING_GUIDELINES.md` — how to patch the kernel safely.
- `VALIDATION_OWNERSHIP.md` — who/what validates each surface.
- `FULL_FUSED_PARITY_HARNESS.md` — the full-vs-fused parity harness spec (realized by
  `tests/core_contracts/test_full_fused_parity.c`).
- `REVIEWER_HANDOFF_2.md` — pass-249/250 status reconciliation companion to `../REVIEWER_HANDOFF.md`.
- `M7_PERF_MODEL_PLAN.md` — the M7 perf-model **fidelity contract** (AI.md cites it for the
  measured-vs-modeled provenance rule). The campaign (P0–P6) is complete; the doc stays as the
  live contract for the perf-model tooling (census/qemu/cycles/memory/wcet).

## Plans not currently in execution (move back up to `../` when picked up)
- `OPTIMISATION_PLAN.md` — optimisation track (PARTLY LANDED); companions `ULTRAPLAN.md`.
- `ULTRAPLAN.md` — Campaign-2 master plan (partly stale; the Campaign-3 forward mandate is
  `../REVIEWER_HANDOFF.md`).
- `OSCILLATOR_BACKEND_CONTRACT_PLAN.md` — oscillator backend contract + Q15 unification (design captured).
- `QTYPE_REFACTOR_PLAN.md` — Q-type refactor + SIMD-width generalization (design captured; tail x86-CI-gated).
