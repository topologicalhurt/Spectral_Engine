# Patch notes — Pass 160: guarantee manifest + self-report API (Phase B1+B2)

## Problem

B0 (pass 159) defined and tested one reconstruction invariant (COLA/WOLA) but the
registry still had no *machine-readable* answer to "which kernel correctness/quality
invariants actually hold in this build, and which has a flag relaxed?". The
correctness-relaxing gates were scattered across `spectral_config.h`, CMake, and a
Metal `.m`, with no single enumeration, no per-gate error budget, and no way for a
host/test to read the active set and fail closed.

## Change

Built the guarantee manifest (B1) from the gates the **C sources actually branch on**
(grep-verified, not the ULTRAPLAN's aspirational list) and exposed it as a
compile-time + runtime self-report (B2), pinned by a two-mode drift CTest.

```text
core/spectral_guarantees.h        NEW header-only manifest + self-report.
  7 guarantee bits, each SET when its invariant holds / CLEARED when a flag relaxed it:
    ieee_strict_fp            <- SPECTRAL_CUSTOM_FAST_MATH_MODE == 0
    exact_trig                <- SPECTRAL_ENABLE_APPROX_TRIG == 0
    exact_atan2               <- SPECTRAL_ENABLE_APPROX_ATAN2 == 0
    exact_inv_sqrt            <- SPECTRAL_ENABLE_APPROX_INV_SQRT == 0
    exact_peak_log            <- SPECTRAL_ENABLE_APPROX_PEAK_LOG == 0
    exact_gpu_fp              <- SPECTRAL_METAL_FAST_MATH == 0
    deterministic_reduction   <- SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS > 0 (inverse)
  SPECTRAL_ACTIVE_GUARANTEES   preprocessor-evaluable bitset (#if / _Static_assert ok).
  spectral_active_guarantees() / _guarantee_holds(bit) / _guarantees_satisfy(mask)
                               runtime query; satisfy() fails closed on a relaxed bit.
  spectral_guarantee_table(&n) host/sim-only rows {bit,name,gate,relaxes}.

tests/core_contracts/test_guarantees.c  NEW. Verifies (a) compile-time set == runtime
  query, (b) each bit is wired to the correct gate macro, (c) fail-closed semantics
  and the descriptor table covers ALL_MASK, (d) each APPROX_* approximation stays
  within its measured error budget vs libm.

spectral_engine/cmake/targets/core-guarantees-test.cmake  NEW. Compiles that source
  TWICE: core_guarantees_test (default gates) and core_guarantees_drift_test
  (SPECTRAL_ENABLE_APPROX_* = 1) so the budgets gate the real approximate code path.
  Both link only spectral_fast_math.c (dead-strip). Registered as CTests
  core_guarantees / core_guarantees_drift. Wired into spectral_engine/CMakeLists.txt
  after core-contracts-test.cmake.

docs/core_audit/CORE_CONTRACTS.md  B1/B2 sections filled in (status B0->B0/B1/B2
  landed): manifest table with gate/default/relaxes/cost/budget/test + the
  SPECTRAL_REPRO_BUILD and SPECTRAL_OPT_LEVEL notes; B2 API surface.
```

## Finding

Two ULTRAPLAN list entries were inaccurate and are corrected in the manifest rather
than copied:
- **`SPECTRAL_OPT_LEVEL` gates nothing.** It is defined (default 1) but no C source
  reads it (grep: zero hits outside `spectral_config.h`). The plan's "`>= 2` drops LUT
  interpolation to nearest" is aspirational. It gets **no guarantee bit** — documenting
  a guarantee the code does not honor would be the exact "hidden behind a branch" trap
  Phase B exists to prevent.
- **`SPECTRAL_REPRO_BUILD` is a CMake variable, not a C macro.** It drives
  `SPECTRAL_CUSTOM_FAST_MATH_MODE` (ON => 0, OFF => 1 plus the `-ffast-math` family in
  `host-config.cmake`), so it is represented *through* `ieee_strict_fp`, not as its own
  bit. The default dev build (`SPECTRAL_PRODUCTION_BUILD` off) reports
  `ieee_strict_fp` CLEARED — honest, since that build is compiled `-ffast-math`.

Measured worst-case approximation error (gate forced on), which set the budgets:
sin 1.72e-6 abs, atan2 2.03e-4 rad, inv_sqrt 4.74e-6 rel, peak_log 1.03e-6 abs.

## Verification

```text
- five production targets build clean: desktop, simulate, embedded_arm,
  embedded_arm_float, simulate_daisy (only pre-existing benign -mavx2/-mno-avx512f
  unused-arg notes). spectral_guarantees.h is not yet included by any production TU,
  so the engine binaries are byte-unaffected; the header stands alone for hosts/tests.
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- core_guarantees_test  prints SPECTRAL_ACTIVE_GUARANTEES = 0x3e (dev build:
  exact_trig/atan2/inv_sqrt/peak_log/gpu active; ieee_strict_fp + deterministic_reduction
  off) and all drift errors << budget.
- core_guarantees_drift_test prints 0x20 (only exact_gpu_fp active; the four APPROX_*
  bits cleared) with sin 1.72e-6 / atan2 2.03e-4 / inv_sqrt 4.74e-6 / peak_log 1.03e-6,
  each within budget; the cleared bits exercise spectral_guarantees_satisfy fail-closed.
```

## Scope (Phase B increment)

Closes Phase B: B0 (reconstruction invariant) + B1 (every relaxing flag in the
manifest with an error budget and a drift test) + B2 (machine-readable active-guarantee
set at compile and run time, fails closed). No production engine code changed; this
adds the registry header + tests + docs only. Next campaign phase per ULTRAPLAN: C.
