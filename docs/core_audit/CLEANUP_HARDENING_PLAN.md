# Cleanup & Hardening Campaign

Inserted phase before resuming Ultraplan Campaign 2. Two macro stages, executed
iteratively (kernel/patch style). Maintainer-directed order; decisions locked below.

## Locked decisions
- Order: **Macro-1 cleanup first, then Macro-2 bug audit.** AI.md drafted alongside.
- Patch notes: **consolidate** the 218 `PATCH_NOTES_PASS*` into one digest, delete the
  per-pass files (git keeps history), **rewrite `AI_CANON §18`** to "terse notes per
  commit, no AI/prompt-referential prose."
- Static tests: **delete the 122 grep change-detectors**, keep the 9 behavioral Python +
  all C tests, backfill real behavioral suites in Macro-2.

## Test ground truth
- KEEP (CTest-wired C): arm-core, core-contracts, core-guarantees, osc-parity,
  osc-width-parity, q-domain-contract, osc-backend-contract, q15-compute-precision,
  q15-production-parity, q15-simd-parity, phase-nco-precision, full-fused-parity.
- KEEP (behavioral Python, compile+run C): pass11,12,13,14,15,16,17,18,20 in `tests/core_math/`.
- DELETE: every other `tests/core_math/test_*.py` (122 files) + `tools/core_audit/core_static_audit.py`.
- The 122 static tests + audit script are orphaned: nothing in CMake/CTest/CI runs them.
- CI (`.github/workflows/c-cpp.yml`) is a dead GitHub template (`make check`/`distcheck`
  don't exist) — repoint at real CTest.

## Macro-1 — cleanup & tech debt
| ID | Scope | Dep |
|----|-------|-----|
| C2 | Test triage: delete 122 static tests, retire `core_static_audit.py`, fix CI | first (unblocks refactors) |
| C1 | Doc consolidation: 218 pass-notes → digest; amend `AI_CANON §18`; strip AI-prose | after C2 |
| C3 | Comment hygiene: kill meta/prompt-referential & over-verbose comments | rolling |
| C4 | Dead/legacy/obfuscated removal + architectural minimality (no-op compute, alias wrappers) | after C2 |
| C5 | Dedup / single-source-of-truth: constants, macros, cross-backend formulas, host/embedded ports | after C4 |
| C6 | API maturation: resolve draft vs real surfaces; version & freeze public contract | after C5 |
| C7 | `AI.md`: tight, codebase-tailored, from `AI_CANON` + findings | finalize last |

## Macro-2 — logical-error audit (each finding that spans the program lifecycle → strong C test suite)
| ID | Area |
|----|------|
| L1 | DSP / audio correctness |
| L2 | Math-heavy code (estimators, phase, windows, scaling) |
| L3 | GPU / SIMD / CMSIS compute parity |
| L4 | Type casting / packing (Q15, endian, fixed↔float) |
| L5 | Conflicting build flags / feature-macro matrix |
| L6 | Pipeline end-to-end (analysis → track → synth → out) |
