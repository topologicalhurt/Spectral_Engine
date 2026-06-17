# Archived planning docs — completed campaigns

These plans/audits are **done**; kept for history (git also has full history). The forward
mandate and the actively-worked plan(s) live one level up in `../` (`../REVIEWER_HANDOFF.md`,
`../IFFT_SYNTHESIS_PLAN.md`); the canon, reference docs, and
not-currently-executing plans (AI_CANON, CORE_CONTRACTS, CHANGELOG, OPTIMISATION_PLAN, ULTRAPLAN,
OSCILLATOR_BACKEND_CONTRACT_PLAN, QTYPE_REFACTOR_PLAN, REVIEWER_HANDOFF_2, …) live in
`../reference/` (see its README).

- `MASTER_PLAN_CLOSURE_CRITERIA.md` — Campaign-2 stop criteria (met).
- `CLEANUP_HARDENING_PLAN.md` — Macro-1 cleanup campaign C1–C8 (done). Its Macro-2 logical
  audit (L1–L6) is carried forward in the handoff.
- `ARCHITECTURE_CLEANUP_STATUS.md` — architectural cleanup phases C–F record.
- `QTYPE_DOMAIN_PLAN.md` — Q-type domain campaign (Q0–Bv landed).
- `OSCILLATOR_UNIFICATION_PLAN.md` — oscillator unification (complete).
- `QTYPE_ISLAND_AUDIT.md` — Q-island audit (closed; remaining tail is x86-CI-gated, see QTYPE_REFACTOR_PLAN).
- `VDSP_MATH_ACCEL_AUDIT.md` — vDSP/Accelerate math-accel audit (complete; vvatan2f promoted on Apple).
- `GPU_Q15_DOUBLEPACK_AUDIT.md` — GPU Q15 double-pack audit (complete; promotion decision pending, hardware-gated).
- `KERNEL_LAYOUT_PLAN.md` — Linux-kernel-style kernel/arch/drivers refactor (S3, complete; the layout is now LAW, enforced by `tests/tools/test_layering.py`).
- `ARCH_PATH_SELECTION.md` — complete verified census of every path-selection site (CMake file-select / whole-file self-`#if` / in-body capability `#if` / width `.inc` / runtime vtable) and the arm/arch decoupling verdict. Conclusion: the tree is fully conformant, every apparent inconsistency (the iFFT self-guard) is principled. The durable rule is distilled into `../reference/AI_CANON.md` #20; this doc is the backing rationale + census.
- `MASTER_REVIEW_PLAN.md` — Major-Patchset Review instance 1 (complete; W1–W7 + KERNEL_LAYOUT).
- `MASTER_REVIEW_PLAN_2.md` — Major-Patchset Review instance 2 (complete; naming/comment/honesty sweep + the spectral_q.h Q-ladder).
- `MASTER_REVIEW_PLAN_3.md` — Major-Patchset Review instance 3 (complete; full AI.md + K&R sweep, every finding fixed or decided).
