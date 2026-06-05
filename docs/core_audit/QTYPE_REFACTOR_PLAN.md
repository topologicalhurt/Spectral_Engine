# Q-Type Refactor & SIMD-Width Generalization Plan

> **Status:** design captured 2026-06-05, awaiting maintainer-set execution order.
> **Relationship to [QTYPE_DOMAIN_PLAN.md](QTYPE_DOMAIN_PLAN.md):** that plan shipped the
> opt-in Q15 *compute kernels* (Q0–Q5 + Bv, PASS206–216) and a *starter* discipline layer.
> This plan is the deeper re-cut the incremental passes deferred: make the Q-domain a
> **clean, enforced, first-class type system** (not convention + a source-grep), **widen**
> the throughput win to every path where it actually pays, and **generalize the SIMD width**
> beyond the single kernel that currently uses it. Same north star: **float stays the default
> compute domain; Q15's primary home is embedded; desktop Q15 is opt-in.**

---

## 0. Why this phase exists (maintainer intent, 2026-06-05, verbatim)

1. Q15 must be first-class on **desktop**, not just embedded — every path that *could benefit*
   on desktop should leverage Q15 packing.
2. **Clean separation** of types — readability, maintainability, robustness — and **no mixing**
   of float/double/int and Q types unless absolutely necessary.
3. **Refactor** the current approach — it's a major tech-debt source; find the cleanest design
   pattern that achieves (1) and (2).
4. Generalize beyond the obvious example (SIMD packing 2×) — **don't be myopic**.
5. Decide the **representation**: is a `short`/`q16` enough, or do we want a custom type? What
   about portability and industry standard?
6. (Separate) Does SIMDe let us use the **max available width** always (AVX-512 / AVX2 / NEON),
   not the 128-wide default we mostly use today?

---

## 1. Grounded current state (verified 2026-06-05, not recited)

What already exists, with file:line, so this phase builds on the tree rather than over it:

- **Q-types are transparent typedefs** (`synth/math/spectral_q15.h:47-50`):
  `q15_t = int16_t`, `q31_t = int32_t`, `uq16_t = uint16_t`, `uq32_t = uint32_t`. This is the
  CMSIS-DSP convention. Packed *structs* exist too (`SpectralSegmentQ15` etc., lines 210/225/242)
  but the scalar domain types themselves carry **zero compiler-enforced safety**.
- **Boundary macros** exist (`FLOAT_TO_Q15` / `Q15_TO_FLOAT` / `PHASE_RAD_TO_Q15` /
  `OMEGA_TO_Q88` …) and are the sanctioned float↔Q conversion sites.
- **Region markers + contract test**: `SPECTRAL_Q_DOMAIN BEGIN/END` markers plus the
  `q_domain_contract` CTest (a `cmake -P` **source scan**) enforce "no float inside a Q region"
  and "no raw scale constant outside the allowlist." This is the *current* separation mechanism —
  a grep, not the language. Known gotcha: it token-matches `float`/`double` even in comments
  inside a region.
- **SIMD max-width parameterization exists in exactly ONE place**: the float L1 kernel,
  `core/port/host/oscillator_simd.c:43`, selects W=4 vs W=8 off
  `__AVX2__ && SIMDE_NATURAL_FLOAT_VECTOR_SIZE_GE(256)` (the Q2/PASS207 work). **Everything else
  is hard-`simde__m128` / 128-pinned**: the packed Q15 kernel (`osc_simd_q15_segment`), the
  templated `oscillator_simd_kernel.inc` at W=4, and `core/port/host/spectral_vector_ops.c`.
- **The build deliberately caps at 256**: `cmake/.../host-config.cmake:20-23` =
  `-march=native -mtune=native -mavx2 -mno-avx512f`. AVX-512 is explicitly off.
- **Desktop Q15 reaches end-to-end** via `--q15` (PASS215): packed 8×Q15 on saw/square/triangle/
  parabola, scalar Q15 on sine, float on asin/quantized/pwm. Phase is the integer-NCO (PASS211–214).
  PASS216 compiled the whole desktop Q15 body out of embedded firmware.

**Verdict on the maintainer's tech-debt instinct: correct.** The compute kernels shipped, but
"clean separation" is currently enforced by a CMake source-scan, not the type system, and the
*named type family* is implicit. That gap is Thread A.

---

## 2. Three independent threads (maintainer sequences them)

The six asks cluster into three threads that can be ordered independently. Each carries its
measured constraints and is **measure-first / decline-on-data** where a win isn't proven.

---

### Thread A — Type system & clean separation (asks 2, 3, 5)

**Goal:** make the Q-domain a coherent, documented, enforceable first-class system.

**The crux finding (drives everything):** in C you **cannot** get compiler-enforced type safety
*and* SIMD-packability from the same type — they are in direct tension.
- A **struct newtype** (`struct { int16_t v; }`) gives real safety but kills the hot path: no
  operator overloading in C, and you can't cleanly `load_si128` an array of wrapper structs and
  vectorize. This is precisely why **CMSIS-DSP, libfixmath (`fix16_t`), TI, ADI** all use
  transparent typedefs in their kernels.
- C++ strong-typedefs (`fpm`, P0037 `fixed_point`, Rust `fixed`) give safety **and** ergonomics
  via operator overloading — but only in C++, and still don't vectorize a wrapper-array by hand.
  Useful for desktop *tooling/tests*, never the kernels.
- **"q16 as a short"**: `q15_t` already *is* a short; a bare `short` discards the binary-point
  contract (the Q-format *is* the type). The right representation is **per-quantity**, which
  argues for a small **named family**, not one blanket short:
  - amplitude ∈ [−1, 1): **signed q15** (note +1.0 is not exactly representable — classic Q15 asymmetry).
  - phase mod 2π: **unsigned `uqN`**, where integer overflow *is* the wrap (free, branchless) —
    already the NCO idiom.
  - wider intermediates: q31 / q88 (omega) as today.

**Design decision (recommended): keep transparent typedefs for the kernel-facing types; make
separation a *layered, enforced discipline*, not a language feature.** This is F2 from the
existing plan, re-cut as a real system instead of a starter.

Work items (each refine-at-execution):

- **A1 — Document the complete Q-type family as one coherent map.** In `spectral_q15.h` (or a
  sibling doc), state for *every* domain type: its Q-format (`Qm.n`), numeric range, wrap
  semantics, intended quantity (amplitude / phase / omega / intermediate), and the *only* legal
  boundary conversions to/from float and to/from other Q types. This "domain map" is the
  design-pattern deliverable — it's what makes the separation legible and maintainable.
- **A2 — Harden the contract test.** Today it's a token-grep with a comment false-match gotcha.
  Decide (measure-first: count real leaks caught vs. false positives) how much rigor is worth it:
  keep the cheap scan, or move to a clang-based / AST check that understands comments and scopes.
  **Decision point — needs maintainer sign-off** (rigor vs. build-time/complexity cost).
- **A3 — Audit boundary-macro coverage.** Verify *every* float↔Q conversion in the tree routes
  through a named macro (extend the contract test's RULE 1 from scale-constants to all conversion
  sites). Fix any raw conversions found (byte-neutral, like the Q0 `wavetable.c:254` fix).
- **A4 — (Optional) C++ strong-typedef wrapper for desktop tooling/tests only.** Go/no-go on data:
  does it catch real domain-mixing bugs the contract test misses? **Likely decline** unless
  demonstrated value. Never enters the kernels.

**Output:** a clean, documented, CI-enforced Q-domain with no float/Q mixing outside named
boundaries — the "robustness + maintainability" the maintainer asked for, achieved by discipline
+ tooling because the C type system structurally can't deliver it in the hot path.

---

### Thread B — Widen the Q-domain islands on desktop (asks 1, 4)

**Goal:** extend the packed-Q15 throughput win beyond today's 4 algebraic timbres — but only
where it *actually pays*.

**The measured reframe (the non-myopic reading of "every path that could benefit"):** the
PASS210 result is decisive — a per-sample float→Q15→float round-trip (`fcvtzs`/`scvtf` crossing
the register file) costs **more** than the cheap float eval, so Q15-over-float-phase ran **3–5×
slower** than float-SIMD. The 2× density only materialized once the **whole island stayed
integer** (which is why the integer-NCO had to be built first). So the design target is
**maximal *unbroken* Q-domain islands** — phase + eval + accumulate with **no float round-trip
mid-kernel** — with boundary macros marking exactly where float is unavoidable (the final amp
ramp + WAV write are inherently float). "Q15 everywhere" is the wrong target and the myopic trap;
"identify and widen the contiguous islands" is the right one.

Work items:

- **B1 — Sine pack8 SIMD-Q15 re-validation** (the already-open deferred follow-up #2). With the
  Bv vec phase, sine's pack8 (0.761 ns/sample) now *edges* production float-SIMD (0.876) but stays
  excluded via `osc_simd_q15_available` pending precision re-validation of the serial-LUT SIMD-Q15
  sine. Measure dBFS vs the scalar oracle; route it in only if it clears the per-path budget.
- **B2 — Full-Q15 accumulate experiment.** Today the island breaks at the amp ramp: the kernel
  widens Q15→float and does a float FMA-accumulate. PASS213 noted the theoretical 2× needs
  "full-Q15 accumulate and/or native 256-bit-int." Measure-first: does keeping the amp ramp +
  accumulate in Q close the gap? **Risk:** Q15 accumulation overflow/precision over a segment —
  carries a measured dBFS justification. **Decline-on-data** if the float widen is already near
  the ceiling or precision regresses. **Decision point — needs sign-off** (precision risk on a
  shipping path).
- **B3 — Island audit (anti-myopia deliverable).** Enumerate every desktop hot path and classify
  each as a *contiguous Q-island candidate* vs. *inherently float* (the FFT/magsq/phase analysis
  path is float by nature; final WAV is float). Document *why* each is in or out. This is what
  stops "leverage Q15 on every path" from becoming a net-negative scattershot — it makes the
  coverage decision evidence-based and reviewable.

---

### Thread C — SIMD max-width generalization (ask 6)

**Goal:** use the max native width everywhere it pays, not just the one float L1 kernel.

**The definitive SIMDe answer:** yes, there is a standard idiom — `SIMDE_NATURAL_VECTOR_SIZE` and
the typed `SIMDE_NATURAL_FLOAT_VECTOR_SIZE` / `SIMDE_NATURAL_INT_VECTOR_SIZE`, with `_GE(n)`/
`_LE(n)` predicates. Under `-march=native` they resolve to the machine's real max width. Three
caveats that shape the work:

1. **Compile-time, not runtime.** SIMDe does *not* auto-scale one body or dispatch at runtime —
   you author one body per width tier (128/256/512) and select with `#if ..._GE(256)`. We've
   proven the pattern (the float L1, `.inc`-templated W=4/W=8); the work is **extending it**.
2. **Width is type-dependent.** On plain AVX, float=256 but int=128, so 256-bit *Q15* needs AVX2
   specifically. Q15 kernels must key off `SIMDE_NATURAL_INT_VECTOR_SIZE`, float off
   `SIMDE_NATURAL_FLOAT_VECTOR_SIZE` — they can differ on one CPU.
3. **We cap ourselves at 256** (`-mno-avx512f`), and it's a `-march=native` build. A *portable*
   binary that picks width on the target CPU at runtime needs FMV (`target_clones`) / Highway —
   **out of scope** unless we ship a portable binary.

**Hard measurement constraint (per [[avoid-assumptions]]):** the dev Mac is **NEON = 128**, which
is already maximal there. *Every* width tier above 128 is **x86-only and unmeasurable on this
machine.** We will not ship a 256/512 kernel we never ran — Thread C's wider tiers are gated on an
x86 build/CI being available to validate them.

Work items:

- **C1 — Width-parameterize the packed Q15 kernel** (8×Q15@128 → 16×Q15@256 on AVX2), keyed off
  `SIMDE_NATURAL_INT_VECTOR_SIZE`, reusing the Q2 `.inc`-templating pattern. This is the largest
  width opportunity (the Q15 kernel is currently hard-pinned 8-wide). Measure x86 speedup —
  **blocked on x86 validation** (can't run on NEON).
- **C2 — `spectral_vector_ops.c` width audit.** Measure-first whether parameterizing it pays;
  decline if the scalar fallback already auto-vectorizes to max width under `-march=native`.
- **C3 — AVX-512 decision.** Lift `-mno-avx512f` and add a 512 tier? Requires AVX-512 silicon to
  validate. **Decision point — needs sign-off**, and gated on x86/AVX-512 hardware access. FMV/
  runtime dispatch stays out of scope (caveat 3).

---

## 3. Cross-cutting constraints (measured, apply to all threads)

- **Float stays the DEFAULT** compute domain; Q15 is opt-in; Q15's primary home is embedded.
  (Maintainer north star — non-negotiable.)
- **float↔Q conversion tax** (PASS210): Q15 only wins inside an *unbroken* island.
- **Q15 SNR ceiling ≈ 92 dB** (PASS201): every Q-compute path carries a measured dBFS justification.
- **Dev Mac = NEON 128**: all width work >128 is x86-only and unmeasurable here — gate on x86 CI.
- **`-march=native` build**: no portable runtime width dispatch (FMV/Highway) in scope.
- **Desktop default must stay byte-identical** when Q15 is not opted in (as every prior pass held).

---

## 4. Decision points requiring maintainer sign-off

| ID | Decision | Why it needs you |
|----|----------|------------------|
| A2 | Contract-test rigor (keep grep vs. clang/AST) | rigor vs. build-time/complexity trade |
| A4 | C++ newtype for tooling/tests | scope creep risk; likely decline |
| B2 | Full-Q15 accumulate on a shipping path | precision risk vs. throughput gain |
| C3 | Lift `-mno-avx512f` / add 512 tier | needs AVX-512 hardware; portability stance |
| — | Acquire x86 (+AVX-512) CI to validate Thread C | otherwise wide tiers are unvalidatable |

---

## 5. Open questions to resolve at execution

- Does any **non-oscillator** desktop hot path form a real Q-island, or is the oscillator the only
  one (FFT/analysis is inherently float)? (B3 answers this.)
- Is the amp-ramp + accumulate worth moving into Q (B2), or is the float widen already at the
  ceiling on this 128-bit-NEON host (where 16×Q15@256 isn't even available to amortize it)?
- Is there an x86 (ideally AVX-512) target we actually ship to, or is `-march=native` desktop the
  only real target? This decides whether Thread C is a priority or a latent option.

---

## 6. Sequencing

Deliberately **not** chosen here — per how we've worked, the maintainer sets the order. For
reference, the natural dependencies are:

- **Thread A is foundational**: the documented type family + hardened separation makes B and C
  safer to refactor against (you're moving Q code around; the contract test is the seatbelt).
- **Thread B** depends only on the existing kernels + the island audit; independent of C.
- **Thread C** is the most self-contained mechanically but the **least validatable on this Mac**
  (x86-only wins). Lowest priority *unless* an x86 target matters.

A reasonable default order would be **A → B → C**, but that is a recommendation, not a decision.

---

*Linked context: [[QTYPE_DOMAIN_PLAN]] (the shipped Q0–Q5+Bv compute work), [[OSCILLATOR_UNIFICATION_PLAN]]
(the unified L1 kernel these threads refactor), and the standing feedback rules
[[faster-path-should-default]], [[avoid-assumptions]], [[minimal-decline-on-data]],
[[algorithm-before-microopt]].*
