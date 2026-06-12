# Master Review — instance plan (precedes KERNEL_LAYOUT_PLAN execution)

This is the execution plan for the first full run of the **Major-Patchset
Review** (the standing checklist now in `AI.md`). Scope: the whole engine +
tools + tests, prioritized by churn — (i) the perf-stack/F-stream/gate code
of the recent campaign, (ii) the synth/backends/core surface about to move in
KERNEL_LAYOUT_PLAN (review BEFORE moving: never relocate debt), (iii) the
Python tools. Budget: ~5 h across several sessions; every wave below is a
small, independently-committable unit so token/session limits never strand
half-applied changes. Every wave lands green: `ctest` + `pytest tests/...` +
the `m7-baseline` perf gate (mechanical cleanups must be codegen-neutral and
the gate proves it).

## Seed evidence ledger (already observed; file-referenced; each is a work item)

### R1 Duplication
- `segment_to_q15` in the sim backend duplicates the conversion policy of
  `cmd/convert_segments.c` (self-admitted in its own comment) → hoist one
  conversion into core, both callers consume it.
- GPU timbre cap defined ×3: `spectral_backend.c` vtables (×2) +
  `spectral_synth_internal.h:81` + `spectral_synth_internal.c:251` → one
  constant.
- Metal MSL struct mirrors (`SegmentGpu`/`SynthParams`/`TileRange` as
  strings, "no compile-time check possible") → generate from the C structs
  (the oscillator-formula codegen already proves the pattern).
- Rig helpers duplicated by recent work: `semihost_write0`/`write_hex_u32`
  and the xorshift PRNG exist in `qemu_main.c`, `fft_main.c`,
  `ifft_synth_sweep.c`, `test_ifft_synth_parity.c` → one rig-support header
  (`native/qemu/rig_support.h`) and one test-support header.
- `_build_fft_elf` (fft_probe.py) repeats the build recipe of
  `counts._build_runner_elf` → shared ELF-builder helper taking sources +
  defines.
- The `CHECK(...)` assert macro is re-declared in ~10 C test files →
  `tests/support/check.h` (tests stay standalone; one include is fine).
- Local `#define PI` in `ifft_synth_sweep.c` mirrors `SPECTRAL_PI_D` →
  dependency-free harnesses may keep local constants ONLY with a comment
  naming the SSOT they mirror; prefer including `spectral_consts.h`
  (header-only) where it doesn't drag dependencies.

### R2 Wiring
- BUG: `measure --target simulate` resolves the DESKTOP binary
  (`spectral_*_desktop` glob vs `spectral_*_native_simulation` name) — fix
  per-target binary resolution; add a matrix test that resolves the binary
  name for EVERY target profile.
- Dead and unwired: `embedded_sim_set_config/verbose` (no callers),
  `SPECTRAL_SIMULATION_TARGET_DAISY` (no consumers),
  `tests/core_contracts/bench_metal_q15pack.m` (never built) → delete or
  wire, never keep limbo.
- CI builds `simulate` but never executes it → add a smoke execution (tiny
  WAV through the sim pipeline) or document why build-only is the contract.
- GPU paths have ZERO automated tests → parity/smoke test gated on
  hardware availability (Metal present on the dev host — there is no excuse
  for zero coverage on the machine that builds it).
- `host-config.cmake` requires Metal/Foundation on every Apple configure
  though only desktop links them → scope the requirement.
- IFFT TUs are test-only (not in any production manifest list) — by design
  until the hybrid router lands, but the wiring item is tracked here so it
  cannot silently persist.
- Control-flow/build matrix: enumerate `#if` branches that NO current target
  compiles (dead configurations); each is either covered by a build in CI or
  removed. (Known suspects: `SPECTRAL_PRECISE_PHASE` is documented-dormant —
  fine; find the undocumented ones.)
- Adaptive-vs-static review: the GPU one-shot caches' "cleared on first
  try_get" handoff, per-call recomputation sites (e.g. window/motif/LUT
  rebuilds), and any polling-shaped loops — justify or convert to
  compute-once/event-shaped flows.

### R3 Constants & macros
- Multi-evaluation function-like macros with side-effect hazards:
  `SPECTRAL_SAMPLE_ADD`/`SPECTRAL_SAMPLE_MUL`/`FLOAT_TO_SPECTRAL_SAMPLE`
  evaluate arguments repeatedly; `SPECTRAL_PITCH_FACTOR` hides a `powf`
  call as a macro → convert to `static inline` (codegen-neutral; gate
  proves it).
- Language-changing macro: `#define synth_cpu synth_arm32_simulation`
  rebinds a public function name per-build — replace with dispatch
  (KERNEL_LAYOUT L3 carries this; reviewed here, executed there).
- Empirical constants needing provenance or centralization:
  `SPECTRAL_GPU_TILE_SIZE=512`, `SPECTRAL_GPU_SEG_CACHE_SIZE=256` (verify
  the 8 KB threadgroup rationale is written where they are defined),
  `SIMULATION_MAX_ACTIVE`, `SPECTRAL_TRACK_CANDIDATE_BATCH=128`,
  Daisy pool offsets/sizes in `daisy_seed_config.h` — every one either
  derives from a stated budget (like the L3-cache STFT chunk heuristic in
  CMakeLists, the model citizen) or gets its provenance comment; truly
  arbitrary values are marked `[chosen: rationale]` so arbitrariness is
  explicit, never ambient.
- Names owned by vendored libraries (`Q15_MAX` class): rule — guard +
  value-equality note at our definition site (done for Q range macros;
  sweep for others, e.g. `PI`, `MIN/MAX`-style names).
- Committed-fixture hygiene: `m7_baseline.json` stores the absolute
  toolchain path (machine-specific noise in a frozen fixture) → store
  basename + version string only; regenerate. Same wave renames the fixture id
  `campaign3-p2-9voice` (planning-referential identifier compiled into the
  generated workload header — W1 could not prove that codegen-neutral) to a
  workload-descriptive name (`stagger9-8k`).

### R4 Architecture & honesty-of-foundations
- Toy-presenting-as-real sweep: the remaining "method tokens"
  (`hybrid_render`, `serra_smith`, `johnston` no-op stubs) present as
  features — until S3 collapses them they must FAIL LOUDLY or be marked
  unimplemented at the call surface, not silently no-op.
- The WCET residual term and the fft-probe IPC band are labeled bounds —
  re-verify the labels travel with every consumer (no report drops the
  qualifier).
- Framework adherence: capability-not-CPU (grep raw `__ARM_ARCH`/`__APPLE__`
  tests outside the single capability map and the sanctioned port files),
  provenance tags on every emitted number, contract-header discipline for
  every cross-layer call.
- Layering inversions: the q15 and GPU-anchor inversions are queued in
  KERNEL_LAYOUT; this review verifies NO OTHERS exist (include-graph scan
  now, enforcement test lands with L5).

### R5 Bugs & escape analysis
For every bug found here or later: name the harness that should have caught
it, why it didn't, and add the CLASS test, not the instance test. Seed
examples: the simulate-glob bug escaped because no test resolves binaries
per matrix target (class fix above); the gate zip-truncation escaped because
comparison loops over parallel lists lacked a set-equality precondition
(class rule: assert set equality before element-compare — now in the gate;
sweep other zips: `grep -n "zip(" tools/`).

### R6 Comments — the conversational-reference scrub (cardinal sin; the standing rule is AI.md's "No AI/prompt/planning-referential
text in code", violated extensively by recent work)
- `grep -rn "pass[ -]2[0-9][0-9]" spectral_engine tools tests api` — every
  hit in CODE is rewritten: history belongs to git/CHANGELOG; code comments
  reference the governing DOC or the MEASUREMENT ("floors measured by the F1
  harness; see IFFT_SYNTHESIS_PLAN.md"), never the session/pass/date that
  produced it. Known dense sites: `daisy_seed_sdram.h`, `spectral_q15.h`,
  `spectral_perf_accounting.h`, `spectral_config.h` (PRECISE_PHASE block,
  "maintainer decision 2026-06-11"), `wcet.py`, `memory_model.py`,
  `expectations.py`, `mca_validation.py`, `fft_probe.py`, several tests.
- Same sweep for "maintainer decision/directive", "inline-audit",
  "(pass NNN)" datestamps in comments. Docs under docs/core_audit/ MAY carry
  pass history (they are the record); code may not.
- Then the positive pass: comments narrate intuition/derivation/units, not
  change history; delete narration-of-the-obvious.

### R7 File structure
- Cohesion review of the largest TUs: `spectral_cli_pipeline.c`,
  `spectral_synth_arm32.c`, `benchmark_workflow.py` (verb handlers →
  per-verb modules if the file has become a dispatch grab-bag) — split on
  responsibility boundaries only; a long coherent hot pipeline is NOT a
  smell.
- No new "utils" dumping grounds; anything hoisted in R1 lands next to its
  domain.

### R8 Library use
- Pre-implementation check is now doctrine (see AI.md prompt): before
  writing, search CMSIS-DSP/vDSP/SIMDe AND our own SSOT headers. Seed
  retro-checks: the ref radix-2 iFFT (justified: no FFTW on that path —
  verify the justification is written at the definition), harness PRNGs
  (R1 consolidates), `spectral_vector_ops` vs vDSP coverage on Apple.

### R9 The seven-lens panel (applied to the highest-risk files)
senior (interfaces/ownership/error paths) · DSP (conditioning, headroom,
phase/window conventions, aliasing) · mathematician (derivations, bounds
proven not asserted, invariants stated) · architect (dependency direction,
coupling, contract fidelity) · embedded (placement, determinism, ISR-safety,
no hidden allocation, WCET) · maintainer (docs true, churn justified,
migration story) · researcher (reproducibility, provenance, citations).
Target set: the F2 IFFT files, wcet/expectations/memory_model, the SDRAM
header, spectral_synth_arm32.c, the q15 header.

### R11 K&R adherence (golden guide; merciless)
The in-repo "C Programming Language" 2nd ed. governs C style. Sweep for:
cleverness over structure, non-idiomatic control flow, pointer arithmetic
where indexing reads better (and vice versa where K&R idiom prefers
pointers), gratuitous casts, misuse or avoidance of the standard library on
hosted paths, names that fight the book's economy. Runs inside the W6 panel
as an eighth lens with veto power, plus a dedicated sweep of the hot kernels
(spectral_synth_arm32.c, oscillator paths, q15 primitives) where idiom and
economy matter most.

### R10 Honesty ledger (declared up front, by the author of the recent code)
- `test_ifft_synth_parity.c` keep-alive hack (`if (ref[0] == 42.0f) ...`) —
  exactly the lazy-code class this review exists to kill → volatile sink.
- `test_perf_gate.py` inline `__import__` hack → proper import.
- Pass-references throughout recent comments (R6 carries it).
- `m7_baseline.json` absolute path (R3 carries it).
- IFFT render does full-spectrum memset per frame where only K-neighborhoods
  are dirty — measured irrelevant at current sizes; the comment must say so
  or the cleanup happens.
- The baseline `generate` path runs counts without the reproducibility
  double-run — re-enable for generate (slow path, run rarely, must be gold).

## Execution waves (each = one commit unit, each green)

- **W1 — comment scrub (R6)**: mechanical but careful; zero behavior change;
  gate proves codegen-neutral. Largest single debt; do first while context
  is fresh on which comments carry provenance that must be preserved as doc
  references.
- **W2 — dead code + wiring fixes (R2)**: simulate-glob bug + matrix
  resolution test; dead API/define/bench removal; CI smoke decision.
- **W3 — macro hygiene + constants (R3)**: multi-eval macros → static
  inline; provenance comments; baseline fixture path fix + regenerate.
- **W4 — duplication hoists (R1)**: rig/test support headers; conversion
  policy unification; ELF-builder share; timbre-cap constant.
- **W5 — honesty items (R10) + bug-class tests (R5)**.
- **W6 — seven-lens panel (R9) + K&R lens (R11) + architecture sweep (R4)**: fleet workflow
  (budget permitting) or inline lens-by-lens; findings feed W-fix commits.
- **W7 — re-verify**: full suites + gate + a final adversarial diff review
  of everything this review itself changed (the reviewer reviews the
  review); update CHANGELOG and this plan's status; THEN KERNEL_LAYOUT L0+L1
  begins on a clean foundation.

## Status
- Plan authored + standing prompt added to AI.md.
- **W1 DONE** (comment scrub): 33 files rewritten to timeless doc/measurement
  references; mca FACTS vocabulary de-jargonized ("corroborated"/"direct");
  fixture id rename deferred to W3 (compiled into the generated header).
  Verified ctest 20/20 + pytest 69 + m7-baseline gate PASS (codegen-neutral).
- **W2 DONE** (wiring + dead code): simulate-glob bug fixed via CMake-derived
  per-target resolution + 5-test class lock (fail-on-bug verified); dead sim
  API/define deleted; bench_metal_q15pack wired (Apple manual target);
  simulate_smoke ctest (E2E, isolated workdir); gpu_backend_parity ctest
  (first GPU coverage, measured bounds, exit-77 skip protocol); CI runs
  tests/tools + pyyaml; Metal REQUIRED justified in place. ctest 22,
  pytest 74, gate PASS. Remaining R2 sweeps (dead #if configurations,
  adaptive-vs-static review) ride W6; main.c vtable bypass rides
  KERNEL_LAYOUT L0 as planned.
- W3 next: multi-eval macros → static inline; constants provenance;
  baseline fixture absolute path + campaign3-p2-9voice id rename
  (deliberate regeneration).
