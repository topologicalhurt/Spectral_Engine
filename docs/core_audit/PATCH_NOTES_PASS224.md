# Patch notes — Pass 224: oscillator backend-contract pin-completeness gate (Oscillator-Backend-Contract Phase 3)

## Problem

Phase 3 of `docs/core_audit/OSCILLATOR_BACKEND_CONTRACT_PLAN.md` asks to
"generalize don't-drift beyond Q15" and produce a gate such that "adding a
backend or editing the contract without updating the others fails CI." The
literal plan framing was "fold the existing float parity CTest + the Phase-1
Q15 parity CTest into ONE backend×domain matrix gate."

Investigating the actual coverage first (per the minimal/decline-on-data rule)
showed the matrix is *already* gated as a coherent constellation — and that
folding the modular tests into one monolith would DESTROY information, not add
it. What the constellation genuinely lacked was one narrow thing: nothing
*enforced* that every Q15 re-implementer carries its anti-drift version pin.

```text
Existing per-cell gates (all real, all kept):
  float | Scalar : spectral_osc_formulas.h is the shared `static inline` source of truth
  float | SIMDe  : osc_parity + osc_width_parity  (RUNTIME numeric parity vs scalar)
  float | GPU-Metal: verify_metal_osc  (MSL is codegen'd from the C formulas; build
                     fails on drift -- a source-equivalence gate, no GPU needed in CI)
  Q15   | Scalar : spectral_osc_q15.h canonical evaluators (source of truth)
  Q15   | SIMDe  : q15_simd_parity (RUNTIME) + _Static_assert(SPECTRAL_OSC_Q15_VERSION==1)
  Q15   | CMSIS  : parity-by-construction (shared evaluators, PASS223) +
                   _Static_assert pin; runtime hardware-deferred

The gap: the Q15 _Static_assert pins are the load-bearing anti-drift mechanism for
the THREE Q15 re-implementations, but the pins are OPT-IN per consumer. The
compiler enforces a pin's VALUE (bump the contract -> `== 1` stops holding ->
build fails), but it cannot notice a re-implementer that never wrote a pin at all.
A future 4th Q15 backend could include the contract, drift, and ship -- silently.
```

## Change

```text
1. Source-scan completeness gate  (Phase 3)
   spectral_engine/cmake/scripts/osc_backend_contract.cmake   (NEW, cmake -P)
     - Reads spectral_osc_q15.h / spectral_osc_formulas.h, extracts the declared
       SPECTRAL_OSC_Q15_VERSION / SPECTRAL_OSC_FORMULAS_VERSION (asserts both still
       exist -- the matrix header in oscillator_dispatch.h never loses its anchor).
     - DISCOVERS (glob, not hard-coded) every file that #includes the canonical Q15
       contract spectral_osc_q15.h, and asserts each carries a
       `_Static_assert(... SPECTRAL_OSC_Q15_VERSION == ...)` pin. Today's three:
       core/oscillator.c, core/port/host/oscillator_simd.c,
       core/port/embedded/oscillator_simd.c. A new includer is auto-held to the rule.
     - Tolerates both pin forms: direct `== 1` (scalar, SIMDe) and named-constant
       `== SPECTRAL_OSC_Q15_VERSION_CMSIS_PIN` (CMSIS). The `[^;]*` keeps the match
       inside one _Static_assert so an unrelated assert cannot satisfy the rule.

   spectral_engine/cmake/targets/osc-backend-contract-test.cmake   (NEW)
     - add_test(NAME osc_backend_contract ...) running the scan via cmake -P. No
       toolchain, no GPU, no Cortex-M -- runs on every host (mirrors q_domain_contract).
   spectral_engine/CMakeLists.txt
     - include() the new target after q-domain-contract-test.cmake.
```

## Finding

```text
- The two-part split is the design, not a shortcut. VALUE-correctness belongs to
  the COMPILER (the _Static_assert); PRESENCE belongs to this SCAN (the compiler is
  blind to a pin that was never written). Neither alone closes the matrix; together
  they do: scan says "every Q15 re-implementer has a pin," compiler says "every pin's
  value is current." This is why the scan does NOT re-check the pinned value -- that
  would duplicate the compiler and rot.

- Float gets NO per-includer pin rule, deliberately. spectral_osc_formulas.h is a
  SHARED `static inline` source of truth that 9 files merely CALL -- there is nothing
  to drift from for a plain includer, so requiring a pin on each would be noise. The
  float backends that genuinely RE-IMPLEMENT (SIMDe, GPU-Metal, CUDA) are gated by
  runtime parity / codegen-verify, which are STRONGER than a version token (they catch
  actual numeric divergence, not just a forgotten bump). The scan only sanity-asserts
  the float contract still declares a version, so the matrix doc keeps its anchor.

- The "fold into ONE monolithic gate" framing was DECLINED on data. osc_parity,
  osc_width_parity, q15_simd_parity, q15_production_parity, q15_compute_precision and
  full_fused_parity each pin a distinct, independently-meaningful cell with its own
  budget and failure signature; collapsing them would lose per-cell diagnosability and
  churn six green tests for zero coverage gain. The minimal high-value Phase-3
  deliverable was the ONE missing invariant (pin completeness), added as a 13th gate
  ALONGSIDE the existing twelve -- not a rewrite of them.
```

## Verification

```text
- ctest: 13/13 PASSED (the prior 12 + new osc_backend_contract). The new gate runs
  in 0.02s, pure cmake -P, no build dependency.
- NEGATIVE tests proven on a synthetic tree (so no real source was perturbed):
    * consumer that #includes the contract but omits the pin -> FATAL, exit 1, "1 issue(s)"
    * direct-form pin (`== 1`)                                 -> OK, exit 0
    * named-constant pin (CMSIS `== ..._CMSIS_PIN`)            -> OK, exit 0
    * contract header missing its #define version             -> FATAL, exit 1
- POSITIVE on the real tree: "Q15 v1, float v6; 3 Q15 re-implementer(s) each carry a
  version-pin _Static_assert" (oscillator.c, port/host, port/embedded). exit 0.
- 5 standard production builds green: desktop, simulate, simulate_daisy, embedded_arm,
  embedded_arm_float.
- Default desktop render byte-identical BY CONSTRUCTION: this pass adds only a cmake -P
  script, a ctest target, and one include() line. ZERO compiled source (.c/.h) changed;
  no host runtime bytes move.
```

## Scope (Oscillator-Backend-Contract Phase 3, partial)

The autonomous, locally-verifiable part of Phase 3 — make the Q15 anti-drift
matrix SELF-ENFORCING (a new Q15 backend that forgets its version pin fails CI) —
is landed as the `osc_backend_contract` source-scan gate, complementing (not
replacing) the existing per-cell parity/codegen gates. Intentionally NOT done:
collapsing the modular parity CTests into one monolith (declined — it loses
per-cell diagnosability for no coverage gain). Still open / not autonomous: the
GPU-Metal *runtime* parity cell (needs an Apple GPU; today gated by codegen-verify
only) and the CMSIS *runtime* Q15 parity cell (hardware-deferred, PASS223) — both
remain hardware-gated, not gaps this machine can close. Phase 4 (vDSP math-accel
audit, measure-first) and Phase 5 (GPU Q15 double-pack, measure-first) are
independent and unstarted.
