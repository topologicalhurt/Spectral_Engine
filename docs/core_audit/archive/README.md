# Archived planning docs — completed campaigns

These plans/audits are **done** (or locally complete with a maintainer/CI-gated tail); kept for
history (git also has full history). The actively-worked plans + handoffs live one level up in
`../` (`../REVIEWER_HANDOFF.md`, `../REVIEWER_HANDOFF_2.md`, `../IFFT_SYNTHESIS_PLAN.md`); the
canon, contracts, and guidelines live in `../reference/` (which by rule holds **no** plans).

## Closed this pass (2026-06-18)
- `ANALYSIS_PHASE_AT_PEAKS_PLAN.md` — store the STFT spectrum as fp16 re/im, take `atan2f` only at
  tracked peaks (LANDED: net −2.9ms/−4.4%, memory-neutral; validation + complex-estimator gate test
  closed; FFTW/x86 runtime check is the one external-CI item).
- `DESKTOP_PERF_CAMPAIGN.md` — the 2026-06 desktop+ARM perf campaign (was `PERF_OPTIMISATION_PLAN`):
  meter audit + `isfinite` inline win landed; frontier verdict closed it for speed (kernels at their
  limits, remaining wins gated off-host).
- `M7_PERF_MODEL_PLAN.md` — the Cortex-M7 no-hardware perf-model stack (P0–P6 complete; the live
  perf-gate contract is the code/baseline it produced — census/qemu/cycles/memory/wcet). Cited by AI.md.
- `OPTIMISATION_PLAN.md` — kernel optimisation track: F1/F3 infra + O1-B/O2-A landed; the **F** fork
  (oscillator-bank vs IFFT) is resolved — osc-bank stays the default, IFFT ships gated (`../IFFT_SYNTHESIS_PLAN.md`).
- `OSCILLATOR_BACKEND_CONTRACT_PLAN.md` — backend contract + Q15 unification: Phase 0/1a/1d landed,
  1b declined-on-evidence, 1c is maintainer-gated (the LUT-scale decision).
- `QTYPE_REFACTOR_PLAN.md` — Q-type type-system + SIMD-width generalization: Threads A/B landed;
  Thread C (wider tiers) is x86/AVX-CI-gated.
- `CAMPAIGN_2_MASTER_PLAN.md` — the Campaign-2 master plan (was `ULTRAPLAN`); superseded by the
  Campaign-3 mandate `../REVIEWER_HANDOFF.md`.

## Earlier campaigns
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
