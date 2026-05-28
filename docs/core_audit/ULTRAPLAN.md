# Spectral Kernel Ultraplan — Campaign 2

This is the master plan for the second core campaign. Campaign 1 (the host-kernel
audit, passes 1–137) is closed per `MASTER_PLAN_CLOSURE_CRITERIA.md`; its two
explicitly-deferred items were "ARM/embedded redesign as a separate project" and
"compiled full/fused parity harness". This campaign delivers those plus the
contract/guarantee, adversarial-correctness, and tooling-feedback workstreams.

## Governance (applies to every pass)

```text
- One change per pass, recorded in docs/core_audit/PATCH_NOTES_PASS<N>.md.
- Pass numbering is monotonic and numeric; next pass is 138. No letter suffixes.
- Every pass must justify itself as one of: real bug, real ownership cleanup,
  real API-surface reduction, real test/harness improvement, real perf-path
  simplification (KERNEL_PATCHING_GUIDELINES.md §7).
- Correctness before performance (AI_CANON §13). A faster wrong kernel is worse
  than a slower reference.
- When a contract changes, update AI_CANON.md / CORE_CONTRACTS.md /
  ACADEMIC_SOURCES.md / DISCIPLINE_FINDINGS.md as affected (AI_CANON §18).
- Validate at boundaries via spectral_contracts.h; no alias wrappers (§19).
```

## Delivery order (maintainer-directed)

```text
Phase A  ARM / embedded redesign            (immediate item: "arm code")
Phase B  Contract / guarantee registry      (immediate item: "WOLA/COLA contract")
Phase C  CTF / KISS adversarial sweep       (later item: "KISS pass")
Phase D  Compiled harness + tooling feedback (immediate item: "testing"; Phase G)
```

The two "later" structural items — split arches into separate files, and remove
ARM NEON / deprecated code — are folded into Phase A (A0) because they are the
natural first step of the ARM redesign, not independent efforts. The LUT
generator feedback loop is folded into Phase D, because the golden-vector oracle
it depends on is built there.

### Sequencing risk and mitigation

The compiled harness (Phase D) is the natural numerical oracle for the ARM
redesign (A) and the CTF sweep (C), but the maintainer has scheduled it last.
Mitigation:

```text
- Phase A uses the EXISTING ARM simulation parity path as its interim oracle
  (synth_arm32_simulation, hardened in passes 132/137) plus per-pass golden
  output vectors captured before each refactor.
- Phase C findings are fixed immediately with the fix documented in the pass
  note; formal compiled regression tests are backfilled in Phase D (D5).
- Escalation: if A's numeric churn outruns the interim oracle, pull D0+D2
  (CTest infra + golden-vector fixtures) forward ahead of the rest of D.
```

---

## Phase A — ARM / embedded redesign

Objective: replace nominal, comment-asserted "optimizations" with a design that
actually exploits the Cortex-M7 (Daisy / STM32H7: DSP extension, fpv5-d16 FPU,
DTCM/ITCM/AXI-SRAM/SDRAM hierarchy), with explicit cache-coherency and
memory-bandwidth contracts. Files: `synth/backends/arm/*`, `synth/math/spectral_q15.*`,
`core/oscillator_simd.c`, `core/spectral_vector_ops.*`, `core/spectral_config.h`.

### A0 — Structural reset (folds in the two "later" items)

```text
- Establish a per-arch file layout under synth/backends/ so arch code is
  backend-specific only (ARCHITECTURE_CLEANUP_STATUS principle). Move shared
  scalar reference math out of the arch files.
- Remove the __ARM_NEON path in spectral_q15.c: Cortex-M (the embedded target)
  has no NEON; that block only ever compiled on desktop Apple-Silicon and is dead
  for the real target. Desktop SIMD is SIMDe's job (oscillator_simd.c).
- Delete other deprecated/contrived scaffolding flagged during the sweep.
- Capture golden output vectors for representative fixtures BEFORE touching the
  hot loop, to anchor the interim oracle.
```

### A1 — Correctness defects (fix before optimizing, AI_CANON §13)

```text
- Dual LUT-reader divergence: the 4x-unrolled body calls spectral_osc_lookup()
  while the scalar tail and fade call spectral_lut_sin() (arm32.c:596-599 vs
  611/636). Prove sample-identical or unify; otherwise the 4-sample boundary is
  a discontinuity. (At SPECTRAL_OPT_LEVEL>=2, spectral_osc_lookup is
  nearest-neighbour — quantization noise relative to the tail.)
- Frequency-increment units contract: freq_inc = freq_q88 * freq_inc_scale_q24
  (arm32.c:821, scale at :481) evaluates to omega*2^32/sr, where a turns/sample
  accumulator needs (omega/2*pi)*2^32. Derive the units against the float->Q15
  converter; encode as a contract test (AI_CANON §9). Confirm or fix.
- spectral_q15.h:16-18 declares uq32_t three times (copy-paste; intended uq8_t /
  uq16_t). Clean up the fixed-point type set.
- Audit Q15/Q31 saturation on worst-case overlapping-segment accumulation
  (DISCIPLINE_FINDINGS, hardware perspective).
```

### A1b — ARM verification foundation (oracle + sim rework)

The sim (`synth_arm32_simulation`) was a parallel reimplementation, not the real
`spectral_arm32_process`, so the interim hash oracle never exercised the redesign
target. Decided rework (must precede A2/A3):

```text
1. Host-run the real M7 codepath: a build mode forcing SPECTRAL_ARM_M7 on with
   EMBEDDED=0, DMA=0, so spectral_arm32_process/synth_core_m7 run on the host via
   the portable intrinsic fallbacks (spectral_q15.h) and no-op placement/prefetch.
   Only the dsb barrier needs a host-safe fence. Q15 math is bit-identical to HW.
2. CTest correctness harness (tests/arm_core): drive the REAL
   spectral_arm32_init/load/process over fixtures; assert audio vs golden and/or
   desktop-float reference within tolerance. Wires in spectral_arm32_load +
   validate_segment_data (currently orphaned).
3. Sim = perf/resource MODEL over the SAME real code. Separate MEASURED
   (assumption-free: op counts, bytes moved, cache-line touches, active counts,
   memory high-water) from MODELED (cycles/WCET, cache misses via explicit,
   single-sourced, QEMU/hardware-calibratable cost params). Artefacts: memory
   bandwidth, cache-miss model, cycles/WCET, memory high-water + DTCM/SDRAM.
4. Delete the parallel synth_arm32_simulation functional loop (one impl only).
   Retire tests/arm_oracle/oracle.py (interim) once the CTest harness lands.
```

Closure (A1b): real spectral_arm32_process runs on host; CTest asserts its audio;
sim reports calibratable perf artefacts over the same code; no duplicate ARM synth.

### A2 — Memory-bandwidth redesign (the core of the maintainer's critique)

```text
- Invert the loop nest. Today it is segment-major (one voice across the whole
  block, arm32.c:871), so SPECTRAL_SOA_ACTIVE (on by default) buys nothing.
  Move to sample-major / voice-parallel so SoA lanes are contiguous and the
  accumulator is touched once per sample across voices.
- Wire the real dual-MAC: spectral_smlad() (q15.h:106) is defined but never
  called; the hot loop does 4 independent scalar spectral_mac_q15(). Use packed
  SMLAD / CMSIS DSP on the voice-parallel axis.
- Replace the false "batch LUT lookups minimize cache misses" claim: lookups are
  strided by freq_inc (effectively random in-table). Evaluate phase-bucketed or
  per-voice-register oscillator state instead.
- Quantify with the restricted-profile cycle counters already present
  (SPECTRAL_RESTRICTED_PROFILE); every retained optimization needs a
  benchmark-backed survival (AI_CANON §11, DISCIPLINE_FINDINGS perf).
```

### A3 — Cache coherency contracts (build on pass 136)

```text
- Formalize DTCM/ITCM/AXI-SRAM/SDRAM placement as explicit build contracts for
  segment data, active state, accumulator, and LUT (not comments — §6/§8).
- DMA double-buffering (TODOs.md "Double buffering"): extend the pass-136 DMA
  coherency path to a ping-pong prefetch with explicit invalidate/clean policy
  per buffer cacheability.
- Worst-case execution time and bounded active-segment count as first-class
  outputs, not average benchmarks (DISCIPLINE_FINDINGS, real-time boundedness).
```

### A4 — FPU path evaluation

```text
- Evaluate the fpv5-d16 FPU (embedded_arm_float target already exists) for the
  bandwidth-vs-compute tradeoff raised in TODOs.md (half-word/word mix-in for
  ~1.5x effective bandwidth; interleaving stalled FPU with integer DSP).
- Keep Q15 fixed-point as the default/reference; FPU is an explicitly-gated
  alternative with a parity test, never a silent default.
```

### Phase A closure criteria

```text
- Loop nest is voice-parallel and SoA actually reduces measured cache pressure.
- Every retained optimization has a target-relevant benchmark and a parity test
  vs the scalar reference / interim oracle.
- Placement/coherency/DMA are build contracts with tests, not comments.
- No dead arch code (NEON removed); arch files are backend-specific only.
- A1 defects fixed with regression coverage (interim now, compiled in D5).
```

---

## Phase B — Contract / guarantee registry

Objective: a single place that records which kernel guarantees currently hold,
which are relaxed by a flag or runtime path, and how a caller/test can discover
the active set. Generalizes the maintainer's WOLA/COLA concern to all invariants
"hidden behind obfuscations or branches". Home: `core/spectral_contracts.h` +
`docs/core_audit/CORE_CONTRACTS.md`.

### B0 — Establish the COLA/WOLA invariant (prerequisite)

```text
- There is no COLA/WOLA enforcement today: windows are generated unnormalized
  ("normalize explicitly", spectral_windows.h:4-5) and no overlap-add identity
  exists in the engine. A registry cannot record "we broke COLA" until COLA is
  first defined and tested.
- Add the COLA reconstruction invariant (analysis window * synthesis path over
  hop) as a contract with a tested tolerance, citing Griffin-Lim / Allen-Rabiner
  (ACADEMIC_SOURCES) for the WOLA condition.
```

### B1 — Guarantee manifest

```text
- Enumerate every correctness/quality-relaxing gate and the invariant it relaxes:
    SPECTRAL_CUSTOM_FAST_MATH_MODE     -> IEEE-754 / determinism
    SPECTRAL_ENABLE_APPROX_TRIG        -> oscillator/window exactness
    SPECTRAL_ENABLE_APPROX_ATAN2       -> phase estimate exactness
    SPECTRAL_ENABLE_APPROX_INV_SQRT    -> magnitude exactness
    SPECTRAL_ENABLE_APPROX_PEAK_LOG    -> peak-interp exactness
    SPECTRAL_METAL_FAST_MATH           -> GPU/CPU parity tolerance
    SPECTRAL_OPT_LEVEL >= 2            -> LUT interpolation (drops to nearest)
    SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS / SPECTRAL_REPRO_BUILD -> reduction determinism
  Each row: invariant id, default state, what relaxing it costs, error budget.
```

### B2 — Self-report

```text
- Compile-time: a single SPECTRAL_ACTIVE_GUARANTEES bitset derived from the gates.
- Runtime: a small query API so a host/test can read the active guarantee set and
  fail closed when an explicit API is handed a descriptor that violates an
  assumed guarantee (DISCIPLINE_FINDINGS, public API).
```

### Phase B closure criteria

```text
- COLA/WOLA invariant exists and is tested.
- Every relaxing flag is in the manifest with an error budget and a test that
  fails if the flag's documented effect drifts.
- The active-guarantee set is machine-readable at compile and run time.
```

---

## Phase C — CTF / KISS adversarial sweep

Objective: adversarially and exhaustively find the "easy" defects — overflow,
sign/type mismatches, narrowing conversions, arithmetic/precedence errors, weak
conditionals, off-by-one, UB — across every file in the kernel until a reviewer
would find nothing embarrassing. Re-runnable with minimal maintainer input.

### Method

```text
- Per-file checklist applied to core/, analysis/, synth/ (and arch files post-A):
    integer overflow / wraparound (esp. size_t<->uint32_t<->uint16_t hops)
    implicit narrowing and float->int casts (use spectral_* checked helpers)
    signedness and shift behavior on negative q15/q31
    fixed-point scaling / saturation correctness
    conditional and loop-bound correctness (off-by-one, >= vs >)
    UB: aliasing, alignment, uninitialized, sequence points
- One pass per file-cluster; each finding -> fix + a pinned regression case.
- Track coverage so the sweep is provably exhaustive and resumable.
```

### Phase C closure criteria

```text
- Every file in kernel/core, analysis, synth swept and signed off.
- Each finding has a fix and a regression case (interim list now; compiled in D5).
- No remaining raw narrowing casts or unchecked allocation arithmetic
  (AI_CANON §17 allocation boundary).
```

---

## Phase D — Compiled harness + tooling feedback (Phase G)

Objective: the abstraction the maintainer wants for testing — a proper,
extensible compiled harness for the C kernel — plus the out-of-source generator
feedback loop, plus retiring the brittle string-grep tests and fixing CI.

Decisions locked: CTest + small C harness; golden vectors as the canonical
oracle.

### D0 — Harness infrastructure

```text
- enable_testing() + a minimal C assertion/runner; add_test() targets in CMake.
  Link the kernel; one executable per component (oscillator, window, q15,
  peak-estimator, arm32-sim, analysis).
- Python keeps only static/lint duties; behavioral truth moves to C + CTest.
```

### D1 — Full/fused parity harness

```text
- Implement FULL_FUSED_PARITY_HARNESS.md against
  analyze_audio_with_path_mode(... PATH_FULL / PATH_FUSED ...): deterministic
  fixtures, sort, per-field tolerances, nonzero on failure. This is Phase G.
```

### D2 — Golden-vector oracle (canonical)

```text
- Commit golden fixtures (silence, DC, impulse, exact/fractional-bin sine, chirp,
  two-tone, dense) with per-field tolerances. This committed spec is the oracle.
```

### D3 — LUT generator feedback loop (golden vectors validate both sides)

```text
- The committed golden vectors are canonical. BOTH are validated against them,
  neither privileged:
    C runtime LUT      (spectral_lut_init_sine in core/spectral_lut.c)
    out-of-source gen  (tools/spectral_tools/generators/lut_generator.py)
- Use generators/native_bridge.py so the harness can call the C path and diff
  against the generator output and the golden set in one place.
- Generalize the pattern to other generators (resource_hashes.py) and document it
  via the tools ADR (ADR-0001) so the subtrees/ feedback loop is canonical.
```

### D4 — Retire string-grep tests + fix CI

```text
- Migrate the 131 tests/core_math/*.py string-matchers to contract/behavioral
  tests; delete those that only memorialize stale names (forbidden by
  KERNEL_PATCHING_GUIDELINES §6 yet currently pervasive, e.g. pass129/pass133).
- Fix .github/workflows/c-cpp.yml: it runs on the non-existent "debian-latest"
  and calls make check / make distcheck, neither of which the (CMake) Makefile
  defines, so the suite never runs in CI. Point CI at ctest + the pytest lint set.
```

### D5 — Backfill regression coverage

```text
- Convert every Phase A/B/C finding into a compiled regression test under the
  new harness.
```

### Phase D closure criteria

```text
- ctest is green and run by CI on a real runner.
- Full/fused parity harness compiled and passing within documented tolerances.
- Golden-vector oracle validates the C LUT and the Python generator together.
- No test asserts on source substrings except deliberate dangerous-pattern lints.
```

---

## Cross-phase done definition

```text
- ARM path exploits the architecture with benchmark + parity evidence.
- Every breakable guarantee is registered, budgeted, and self-reported.
- Kernel is swept clean of embarrassing-class defects, each pinned by a test.
- A compiled, extensible harness is the source of behavioral truth, with the
  out-of-source generators validated by canonical in-repo golden vectors.
```
