# AI.md — working in the Spectral Engine

Real-time spectral analysis + resynthesis engine in C. Analyze audio into
sinusoidal segments; resynthesize with stretch/pitch/timbre on desktop (float,
Metal/CUDA/vDSP) or Cortex-M7 (Q15 fixed-point, Daisy Seed). `AI_CANON.md` holds
the full correctness rules; this file is the orientation.
**Forward mandate (what to work on next):** `docs/core_audit/REVIEWER_HANDOFF.md`.

## Layout (Linux-kernel-style: kernel / arch / drivers; see KERNEL_LAYOUT_PLAN.md)
- `spectral_engine/core/` — the kernel: config, constants, oscillators, windows, segments,
  Q-domain (`spectral_q15.h`), hashing, contracts, dispatch
- `spectral_engine/arch/` — ISA-contingent kernels, build-selected: `ref/` (portable
  fallbacks, always compilable), `arm/` (ARMv7E-M synth + its host-sim adapter),
  `simd/` (host SIMDe bodies)
- `spectral_engine/drivers/` — device/library backends behind core contracts: `vdsp/` now;
  `metal/`, `cuda/` arrive at L4 (today still `synth/backends/gpu/`)
- `spectral_engine/analysis/` — FFT, peak estimation/tracking, processing chain
- `spectral_engine/synth/` — gpu backends only (lifted into `drivers/` at L4)
- `spectral_engine/runtime/` — console, utils (`spectral_utils.h` lives here)
- `spectral_engine/cmd/` — CLI
- `api/daisy_seed/` — Daisy Seed board support / public API

## Build & test
- `make configure && make` → desktop. Other targets: `simulate`, `embedded_arm`,
  `embedded_arm_float`, `embedded_arm_restricted`, `cuda`, `daisy` (`make help` for the matrix).
- Tests: `cmake --build build --target tests_all && ctest --test-dir build` (the C suite;
  test exes are `EXCLUDE_FROM_ALL`, so build `tests_all` first). Behavioral numeric tests:
  `pytest tests/core_math` (compile+run real C against a Python reference). Perf-harness
  tests: `pytest tests/tools`.
## Performance measurement (the rules; full rationale in spectral_tools ADR-0002)
- ONE entry point: `python -m spectral_tools.testing.benchmark_workflow` (run with
  `--help` for the verbs; `measure --list` shows the live target×instrument matrix;
  the matching CMake targets wrap the same module). Don't add parallel perf
  scripts — extend `performance/matrix.py`.
- Pipeline shape: reproducible build → run under an instrument → parse → interpret.
  The matrix (`performance/matrix.py`) is the SSOT for which instrument measures
  which build target, with runtime availability probes.
- Timing = in-process stage markers (CLOCK_MONOTONIC ns, spectral_log.h). Debuggers
  are never a timing instrument (orders of magnitude of overhead per stop — measured
  in ADR-0002); they are for state inspection only.
- Embedded numbers carry provenance: QEMU counts are `[measured]` and never cycles;
  llvm-mca numbers are `[modeled]`; the two are never blended. Fidelity contract:
  docs/core_audit/M7_PERF_MODEL_PLAN.md.
- C-truth rule: Python derives facts from C/CMake artifacts (parse options.cmake for
  flags, nm for symbols, binaries for counts) — never restates constants or logic.
  Python-originated data the C side needs is GENERATED into C with a content digest
  (fixture header). The only sanctioned duplication is a labeled, parity-tested
  independent-verification implementation.
- A change is not verified until it builds on desktop **and** the embedded host targets
  (`embedded_arm`, `embedded_arm_float`) and ctest is green — a symbol unused in desktop may
  be live under another build flag.

## Build flags that gate code
`SPECTRAL_EMBEDDED`, `SPECTRAL_IS_EMULATOR`, `SPECTRAL_USE_EMBEDDED_SYNTH`,
`SPECTRAL_USE_CUDA`, `SPECTRAL_USE_VDSP`, `__APPLE__`. Code inside any of these `#if`
branches is live. Emulator guard: `#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMULATOR`.

## Single sources of truth — use them, don't re-roll
- Math constants (`SPECTRAL_PI`, `SPECTRAL_TWO_PI`, `SPECTRAL_ATAN2_*`): `core/spectral_consts.h`
- Fast math (`fast_sqrt`, `fast_inv_sqrt`, `spectral_atan2_poly`): `core/spectral_fast_math.h`
- Float waveforms: `core/spectral_osc_formulas.h` · Q15 waveforms: `core/spectral_osc_q15.h`
- Array allocation: `spectral_size_add` / `spectral_malloc_array` / `spectral_calloc_array`
- A formula duplicated across scalar C / SIMD / CUDA / Metal-string code needs a parity test
  and (where a C function can't reach, e.g. GPU strings) a version pin.

## Hard rules (see AI_CANON.md for the full set)
- Correctness before performance. A faster wrong oscillator/phase/fade/estimator is worse
  than a slower reference.
- Approximations (fast atan2/trig/inv-sqrt) are opt-in behind a named gate; exact is the
  default and the test reference. No "fast"/"near exact" claims without a measured bound.
- Keep units explicit: phase=rad, omega=rad/sample, df=rad/sample², amp=linear.
- Hot kernels carry no policy/logging/CLI decisions — keep them small and deterministic.
- Embedded is first-class: deterministic allocation, bounded active segments, saturating
  fixed-point, DMA/cache coherency, worst-case timing.
- No alias wrappers (a function that only forwards to one other). Header functions are
  `static inline`, never plain `static` (per-TU `-Wunused-function`).

## Tests & docs discipline
- Tests assert **behavior**, never source text or doc prose. No test/audit/doc may be
  load-bearing on a string in the codebase.
- Change record = git log + `docs/core_audit/CHANGELOG.md` (one terse line per change). No
  per-change markdown files. No AI/prompt/planning-referential text in code, commits, or docs.

## Third-party dependencies
- `third_party/libs.yaml` is the SSOT (kind: submodule|subtree per entry; strict schema).
  Manage via `python -m spectral_tools.vendor` (`list`, `submodule status|verify|sync|add|remove`,
  `subtree ...`). Submodules bypass bulk sync with `sync: false`; subtrees with `track: none`.
  Never edit `.gitmodules` by hand — `vendor submodule verify` is the consistency gate.
  Rationale: spectral_tools ADR-0003.

## Public API (SemVer, 0.x = unstable)
- Desktop: `spectral_engine/core/spectral_synth.h`
- Daisy Seed: `api/daisy_seed/daisy_seed_spectral.h`
- `api/spectral_api.h` does not exist; these two are the contract.

## The Major-Patchset Review (standing checklist)

Run this review at every PR-sized unit of work — as a yardstick, roughly every
ten commits or before executing any structural plan. It is adversarial by
design: the goal is to find what is wrong, lazy, duplicated, or dishonest, not
to confirm what is fine. Every finding becomes either a fix in the same
review, a tracked item in a plan doc, or an explicit, written decision to
accept it. Style north star: K&R ("The C Programming Language", 2nd ed., in
`docs/core_audit/`) — economy of expression, structure that tells its own
story, comments that carry intuition.

1. **Duplication.** Hunt near-identical blocks (hoist into one function next
   to its domain — never a grab-bag utils file), switch/if ladders that should
   be data tables, parameter clumps that should be structs, and the same fact
   expressed in two languages (C/Python/MSL/docs) — cross-language duplicates
   are either GENERATED from one source or pinned by a parity test; nothing
   is kept in sync by hand and hope.
2. **Wiring.** Every TU is in a build target or a test; every test runs in
   CI; every `#if` branch is compiled by some configured build (a branch no
   target compiles is dead — delete it or add the build). Nothing is
   half-wired: unreferenced functions, unconsumed defines, unbuilt benches
   are deleted or wired, never left in limbo. Every fallible call's result is
   checked or explicitly voided with a reason. Prefer compute-once/
   event-shaped flows over per-call recomputation and polling — and when a
   loop must poll, the bound and the cost are written down.
3. **Constants & macros.** No constant defined twice (guard + value-equality
   note when a vendored library owns the name). Every empirical constant —
   buffer sizes, thresholds, batch counts — either derives from a stated
   budget/measurement (written at the definition) or is marked as a choice
   with its rationale; a bare `512` with no story is a defect. Macros never
   change the language: no control-flow hiding, no function renaming, no
   multi-evaluation of arguments — if it can be a `static inline`, it is.
   Generated constants come from generators with digests, not transcription.
4. **Architecture.** The declared frameworks (capability-not-CPU, contract
   headers, port-pattern TU selection, measured-vs-modeled provenance tags,
   C-truth) are either followed or formally amended — silent deviation is a
   defect even when the deviation is locally better. Hunt sand presenting as
   concrete: toy models, asserted-not-proven bounds, stub features that
   silently no-op. A stub must fail loudly or be marked unimplemented at its
   surface. Tech debt is paid before building on top of it.
5. **Bugs & escape analysis.** For every bug found, answer in writing: which
   existing harness should have caught it, and why didn't it? Then add the
   CLASS test (the input domain / structural property), never just the
   instance regression. A bug with no escape analysis is half-fixed.
6. **Comments.** Comments carry intuition, derivations, units, invariants,
   and provenance of numbers — never change history, session/pass/patch
   references, dates, author voice, or planning narrative (history lives in
   git and the CHANGELOG; the governing doc may be referenced by name).
   Highly technical code deserves generous explanation; obvious code
   deserves none. If a comment explains WHAT the next line does, delete it;
   if it explains WHY or the math, keep it.
7. **File structure.** Split files on responsibility boundaries, not line
   counts — a long, coherent hot pipeline may stay; a dispatch grab-bag may
   not. Headers hold contracts and `static inline` leaf math; TUs hold the
   rest. No file accumulates unrelated verbs.
8. **Library use.** Before implementing, check (a) the vendored libraries
   and (b) our own SSOT headers for an existing solution; after
   implementing, justify in place any reimplementation (the keep/switch
   rubric: measured edge, portability, fusion/directive control vs
   wheel-reinvention and debt). Neglecting our own helpers is the same
   defect as neglecting a library.
9. **The seven lenses.** Read the highest-risk files as: senior engineer
   (interfaces, ownership, error paths), DSP engineer (conditioning,
   headroom, phase/window conventions, aliasing), mathematician (derivations
   and bounds proven, invariants stated), software architect (dependency
   direction, coupling, contract fidelity), embedded engineer (placement,
   determinism, ISR-safety, hidden allocation, worst case), maintainer
   (docs true, churn justified, migration story), researcher
   (reproducibility, provenance, citations). Each lens has veto power.
10. **Honesty.** The reviewer lists what they may have missed and which areas
    deserve a second pass — unknown unknowns are surfaced by naming the
    places nobody looked. Declare every shortcut taken under time pressure
    (keep-alive hacks, inline import tricks, "good enough" tolerances): each
    is either fixed now or written into the plan with its risk. Meticulous
    beats fast; this is a kernel.
11. **K&R adherence, without mercy.** "The C Programming Language" (2nd
    ed., in `docs/core_audit/`) is the golden guide for C principles: economy
    of expression, idiomatic use of the language, structure over cleverness,
    well-chosen names, the standard library used as intended. Every violation
    of its principles is a finding — no seniority of the surrounding code
    excuses it.
12. **Tests are code too.** Every regression test is fail-on-bug verified
    (revert the fix, watch it fail). No tautologies, no bands so wide they
    gate nothing, no silently-skipped suites (a skip must be visible and
    justified). Fixtures are deterministic and digest-stamped; committed
    fixtures contain no machine-specific paths.
13. **Naming tells the truth.** Names state format and units (`*_q15`,
    `*_rad`, `*_bytes`); a counter named for something the code no longer
    does is a lie and gets renamed with its semantics.
14. **Lifecycle & boundaries.** Every create has a destroy on every path;
    allocation failures are handled; inputs crossing a trust boundary
    (files, CLI, vendored data) are validated at the boundary, once, with
    contracts — not re-checked ad hoc downstream.
15. **Docs in the same patchset.** A change that falsifies a doc updates the
    doc in the same unit of work; a stale claim is a bug with the same
    severity as the code defect it describes.

The review's findings ledger and execution waves for the current instance
live in `docs/core_audit/MASTER_REVIEW_PLAN.md`.

## Reference docs
`docs/core_audit/`: `AI_CANON.md` (rules), `CORE_CONTRACTS.md`, `ACADEMIC_SOURCES.md`
(paper-backed methods), `CHANGELOG.md`, and the `*_PLAN.md` campaign plans.
