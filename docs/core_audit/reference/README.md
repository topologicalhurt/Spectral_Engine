# Reference docs — canon, guidelines, contracts (no plans)

This directory holds **only durable, non-planning** material: the canon, the contracts, the
guidelines. By rule, nothing here is a plan or pertains to planning.

- **Actively-worked plans + handoffs** live one level up in `../`: `../REVIEWER_HANDOFF.md`
  (the standing Campaign-3 mandate), `../REVIEWER_HANDOFF_2.md` (its status reconciliation),
  and `../IFFT_SYNTHESIS_PLAN.md` (the live synthesis-fork workstream).
- **Completed / superseded campaigns** live in `../archive/` (see its README) — including the
  former residents here (M7_PERF_MODEL_PLAN, OPTIMISATION_PLAN, OSCILLATOR_BACKEND_CONTRACT_PLAN,
  QTYPE_REFACTOR_PLAN, and the Campaign-2 master plan).

## Canon / rules (the SSOT — update these in the same patchset as a contract change, per AI_CANON §18)
- `AI_CANON.md` — the correctness rules (exact-by-default, approximations gated + parity-tested,
  SSOT for constants, no doc/test load-bearing on source text, capability-not-CPU gating).
- `CORE_CONTRACTS.md` — per-function contracts + the guarantee registry/manifest.
- `ACADEMIC_SOURCES.md` — paper-backed methods + citations.
- `DISCIPLINE_FINDINGS.md` — recurring discipline findings.
- `CHANGELOG.md` — the rolling change digest (one terse line per change).

## Guidelines / contracts
- `KERNEL_PATCHING_GUIDELINES.md` — how to patch the kernel safely.
- `VALIDATION_OWNERSHIP.md` — who/what validates each surface.
- `FULL_FUSED_PARITY_HARNESS.md` — the full-vs-fused parity harness spec (realized by
  `tests/core_contracts/test_full_fused_parity.c`).
- `BUILD_PROFILES.md` — the build-flag philosophy: `cmake/profiles.cmake` is the SSOT;
  host vs embedded/firmware profiles; AVX-512-off + arch-gating + reproducibility.
- `GENERATED_ARTIFACTS.md` — the registry of committed generated sources (generator +
  verify-on-build guard per artifact); the stamp-OUTPUT clean-safety pattern.
