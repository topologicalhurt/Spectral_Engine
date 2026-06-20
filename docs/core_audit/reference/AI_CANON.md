# AI/Human Canon for Spectral Engine Core Work

This file records repeated mistake classes that are canonically incorrect for this repository. Treat it as a guardrail before touching `spectral_engine/core`, `spectral_engine/analysis`, `spectral_engine/arch`, `spectral_engine/drivers`, or GPU/embedded parity code.

## 1. Phase normalization must preserve phase zero

Incorrect pattern:

```c
norm = p / (2*pi);
rads = 2*pi * (norm - floor(norm) - 0.5);
```

That maps `p = 0` to `-pi`. It is not a harmless convention change. It changes saw, square, triangle, PWM and any phase-sensitive cross-backend parity. The canonical wrapped phase is:

```c
rads = p - 2*pi * floor(p/(2*pi) + 0.5);
```

Expected invariants:

- `wrap(0) == 0`
- `wrap(2*pi*k) == 0` for integer `k` within practical precision
- output interval is approximately `[-pi, pi)`
- scalar, SIMD, CUDA and Metal copies are byte-for-byte formula-equivalent or tested to tolerance

## 2. Fade-in starts at silence; fade-out ends at silence

Incorrect pattern:

```c
0.5f * (1.0f - sin((x - 0.5f) * pi))
```

At `x = 0`, this evaluates to `1.0`, not `0.0`. That is an inverted fade. The canonical sine-shifted Hann-ramp form is:

```c
0.5f * (1.0f + sin((x - 0.5f) * pi))
```

Expected invariants:

- first fade-in sample is approximately zero
- last fade-out sample is approximately zero
- fade-in is monotonic nondecreasing over the fade region
- fade-out is monotonic nonincreasing over the fade region
- scalar, SIMD-independent, CUDA and Metal envelopes agree

## 3. “Fast math” is not a default correctness policy

Approximate inverse square root, approximate `atan2`, approximate trigonometry, fast polynomial logs, or empirical rational interpolation must never become the default unless:

1. the error bound is documented;
2. the bound is tested against a reference implementation;
3. the DSP consequence is acceptable for audible output;
4. the approximation is selected by an explicit compile-time or runtime policy;
5. exact/reference mode remains available and is the default for tests.

The phrase “fast” is not a proof. The phrase “near exact” is not a proof. “~10x faster” is not acceptable without a benchmark harness and target hardware statement.

## 4. Three-bin peak interpolation must name its estimator

Incorrect pattern:

```c
p = (right - left) / (left + right + 2*center);
return 1.5f * p; /* empirical boost */
```

This is not a valid general Hann-window frequency estimator without a derivation tied to the exact spectrum representation. The default estimator should be conservative and documented, such as log-magnitude/log-power parabolic interpolation, or a named estimator from the literature with its assumptions encoded in tests.

## 5. Window generation and window normalization are different operations

A function that generates a Hann/Hamming/Blackman shape is not automatically coherent-gain normalized or RMS normalized. Do not document generated windows as “sum equals 1” unless the implementation actually normalizes the sum. Do not mix vDSP flags and portable formulas without verifying identical normalization.

## 6. Do not hide architectural decisions in comments

Incorrect pattern:

- “hardware prefetcher completely hides latency”
- “zero-cost approximation”
- “Linux overcommit guarantees...”
- “single source of truth” when Metal/CUDA strings duplicate formulas
- “placement is INERT / lands in default memory” once the BSP binding it names as
  the precondition already exists in source — a placement (or any conditional-
  optimization) claim is a build-configuration-specific, objdump-checkable fact, not a
  free-standing truth. State the exact config it holds in and the verification command;
  never cite section names absent from the active tree. When a BSP binding lands, every
  comment predicated on its absence is stale and must be re-checked against the binding.

These are claims requiring tests. Convert claims into measurable invariants or remove them.

## 7. A duplicated formula is a liability unless parity-tested

Duplicated formulas currently exist across scalar C, SIMD C, CUDA and Metal string code. Every formula duplication must have:

- a version bump;
- a static assertion where possible;
- a parity test over representative input ranges;
- a documented acceptable tolerance.

## 8. Avoid platform-specific memory contracts in core algorithms

Do not rely on Linux overcommit, unbounded virtual allocation, implicit cache-line size, or compiler-specific alignment attributes as correctness or performance contracts. Core code must be portable or explicitly gated.

## 9. Segment and synthesis math must keep units explicit

For every field, keep units visible in comments and tests:

- `phase`: radians
- `omega`: radians/sample
- `df`: radians/sample^2 contribution when used in quadratic phase
- `amp`: linear amplitude
- `da`: amplitude/sample
- `start`, `length`: samples or seconds; never ambiguous

A change to stretch/pitch/chirp equations must include a unit derivation.

## 10. Real-time kernels should separate policy from kernel math

Core hot loops should not decide logging policy, user-facing fallback messages, research-mode heuristics, or CLI behavior. Keep the kernel small and deterministic; put policy in dispatch/pipeline layers.

## 11. SIMD is not automatically faster

SIMD code must prove:

- equal or bounded numerical behavior relative to scalar reference;
- no worse throughput for small segment lengths;
- no alignment or aliasing assumptions hidden from callers;
- no scalar tail bugs;
- no architecture-specific intrinsic use without fallback.

## 12. Embedded constraints are first-class, not afterthoughts

Embedded code must account for:

- deterministic memory allocation;
- bounded active segment count;
- saturating fixed-point arithmetic;
- FPU availability;
- DMA/cache coherency;
- worst-case execution time, not average benchmark time.

## 13. Do not optimize around broken math

Fix correctness first, then profile. A faster wrong oscillator, wrong phase, wrong fade, or biased peak estimator is worse than a slower reference implementation.

## 14. Every core function needs a minimal contract

For every exported core function, define:

- input domains;
- output range;
- ownership/lifetime rules;
- alignment requirements;
- determinism requirements;
- error behavior.

## 15. Required review question before any core patch

Can this change be tested against a mathematically clear reference implementation? If not, the change is not ready for core.

## 16. Named techniques and paper-backed claims need sources

Any named estimator, DSP technique, numerical method, or paper-backed claim must include a source link or an in-depth technical explanation in code comments or adjacent docs. A bare phrase like “standard estimator,” “from the literature,” or “periodogram convention” is not enough. If a link is unavailable, document the derivation, assumptions, input domain, output units, and why those assumptions match this implementation.

## 17. Reuse existing core math utilities before adding formulas

Before adding a new helper, search the repository for an existing utility,
formula, approximation gate, or backend parity implementation. Duplicating a
bit hack or polynomial under a new name is a correctness risk, not harmless
locality. In particular:

- use `fast_sqrt` / `fast_inv_sqrt` instead of reimplementing inverse-square-root tricks;
- put any new approximation behind a single named gate in `spectral_config.h`;
- keep the exact/reference path as the default;
- add a comparison test between approximate and canonical outputs before
  accepting the approximation;
- update the audit when a helper is intentionally centralized.

Reachability is not unavoidability — extract before you copy
(EXIST → REUSE → EXTRACT → only-then-copy). When the utility already exists but is
not *reachable* from the new caller (a TU-local `static`, an un-exposed `inline`, a
body buried in a width-templated `.inc`), that is a fixable packaging problem, not a
reason to copy: hoist the one definition into a shared header/`.inc` so both sites
include it and there is still exactly ONE definition. A parity-tested duplicate is the
fallback ONLY when a single C definition genuinely cannot reach both sites — e.g. a
cross-language mirror (GPU MSL/CUDA strings). The `ifft_fast_sin4` failure is the
cautionary case: a third copy of the minimax sin was rolled because `osc_vfastsin` was
a private static inside the oscillator kernel; the right move (taken) was to extract the
SIMD sin into `arch/simd/spectral_fast_sin_simd.inc` and have both the oscillator and the
IFFT include it.

If a duplicate formula is genuinely unavoidable, document why a shared definition cannot
reach both sites and add a parity test that would fail if the copies drift.

The same rule applies to ownership and conversion plumbing. If success,
failure, and destroy paths free the same resource set, use one cleanup helper.
If an uncompiled file mirrors live kernel logic, delete it or make it canonical
through the source manifest and tests. Dead duplicate files are not
documentation.

For array allocation, use the kernel helpers by default. `spectral_size_add()`,
`spectral_malloc_array()`, `spectral_calloc_array()`, and
`spectral_realloc_array()` are the canonical arithmetic boundary. Local
`count * sizeof(T)`, `strlen(x) + 1`, or `calloc(1, bytes)` patterns need a
specific platform or ABI reason.

## 18. Change record: terse per-commit notes, not per-pass files

The change record is the git log plus one rolling digest at
`docs/core_audit/reference/CHANGELOG.md`. Do not create a markdown file per change. (The
former one-file-per-pass scheme produced ~218 `PATCH_NOTES_PASS<N>.md` files
and the static tests that policed them; both were consolidated into the
changelog and deleted.)

- Commit messages are terse and factual: what changed and why, measured not
  asserted. No prose narrating the process.
- No AI/prompt/planning-referential text anywhere — code, commits, or docs. No
  "as discussed", "per the plan", pass/phase narration. State the fact, not the
  process that produced it.
- When a change alters a contract, update the affected canon doc in the same
  commit (`AI_CANON.md`, `CORE_CONTRACTS.md`, `ACADEMIC_SOURCES.md`,
  `DISCIPLINE_FINDINGS.md`).
- No test, audit, or doc may be load-bearing on source text or doc prose. Tests
  assert behavior; deleting any doc must not break a test.

## 19. Alias wrappers are not architecture

Do not add a function whose only behavior is to rename another function or
resolve one constant into another helper. It increases API surface without
reducing complexity.

Acceptable wrappers must do at least one real job:

- validate or normalize inputs;
- bind a backend or ownership boundary;
- preserve a stable external ABI intentionally;
- record units or convert representation;
- centralize a policy that has multiple callsites.

For internal kernel code, prefer calling the canonical helper directly. If a
semantic predicate is needed for multiple capability bits, expose one generic
predicate instead of one alias per bit.

## 20. Path-selection mechanism is keyed to the fork type, not habit

"Macro-gate vs. new file" is not a global preference — pick the mechanism by what the
fork actually is:

- **Two mutually-exclusive port bodies** fulfilling one contract header, chosen by a
  **profile decision CMake owns** (host vs embedded) → new file + CMake file-selection,
  with **zero in-body profile `#if`** (e.g. `out_kernels`, `osc_simd`, `gpu_tile`).
- **Two port bodies gated by a platform capability** (resolved in C, not CMake) that must
  **co-link** in one test/parity binary → **whole-file self-`#if`** (e.g. the iFFT
  `SPECTRAL_USE_VDSP` pair). Do NOT convert this to CMake file-selection: it would duplicate
  a C-resolved fact into CMake (violates the C-truth rule), desync the `-D` override, and
  fail to link two same-symbol bodies.
- **Orthogonal hardware-capability flags** stacking within one ISA (DMA/DTCM/CMSIS/M7) →
  in-body `#if` on named predicates. Do not split into files — they co-occur, and on the
  byte-pinned arm32 hot path any restructure moves the m7 baseline.
- **Width/lane parametricity** → re-includable width-templated `.inc`.
- **Runtime user-selectable backend** (CPU/Metal/CUDA) → function-pointer vtable chosen at
  init; never on the embedded hot path (indirect calls defeat inlining).

The full verified census + the iFFT-exception rationale is in
`archive/ARCH_PATH_SELECTION.md`.

## 21. FPU and ALU are temporally owned on embedded, not simultaneously saturated

A Cortex-M-class core issues one in-order instruction stream. Even the dual-issue M7 cannot
hide a floating-point burst behind a *serial* integer recurrence: when each step depends on the
previous (the coupled oscillator), there is no cross-sample ILP to overlap with the FP work.
"Leverage the FPU and ALU simultaneously" is only meaningful *across overlapping work items*
(e.g. one voice's float prep overlapping another voice's integer render), never within one
serial kernel. The rules that follow keep the unit ownership clean and the FP cost honest:

- A per-voice float setup cost computed in floating point and then consumed by a pure-integer
  per-sample loop is an **FPU burst at the block boundary**, serialized against the integer
  loop — the opposite of overlap. Compute per-voice float **invariants once at activation** and
  carry them in the voice record; never recompute a loop-invariant (a rotation matrix from a
  fixed `omega`) every block. Bound recurrence drift with an **integer-domain renorm**, not a
  per-block float re-seed.
- Minimize float on the real-time path. It cannot hide behind integer work on a serial stream,
  so its cost is additive, not free.
- **“Rare / activation-time” is a real-time claim.** Measure it against the *smallest* real
  hardware block (the codec block, e.g. 48 samples), not the largest buffer the kernel can hold
  (e.g. 256). A fixed per-block cost dominates real-time headroom precisely at the small block.
- Bracket the fixed-point hot loop in `SPECTRAL_Q_DOMAIN BEGIN/END` markers so a stray float in
  the per-sample recurrence fails the `q_domain_contract` test — the RT loop is the highest-value
  place to enforce purity, and an FP op injected there re-creates the very interleaving this rule
  forbids.
- State the de-facto unit-ownership model at the kernel surface ("FPU only for the seed; the
  per-sample loop is pure Q31/Q15 ALU; no concurrent FPU+ALU synthesis"). A clean
  setup-burst/steady-state split is a legitimate model — but it is a contract to write down, and
  it makes minimizing the float burst mandatory, not optional.

## 22. A memory barrier is not cache maintenance

A `dsb`/`__DSB()` *orders* memory accesses; it does **not** clean or invalidate the data cache.
On a cacheable region:

- Data a DMA **writes** and the CPU reads needs an **invalidate-after-RX** — a barrier alone
  leaves the CPU reading stale cache.
- Data the CPU **writes** and a DMA reads needs a **clean-before-TX**.
- A read-only-by-CPU RX buffer needs no pre-DMA clean; a buffer the engine never hands to a DMA
  carries no engine-side flush (it belongs to whoever owns the DMA — e.g. the BSP codec buffer).

Each barrier's comment names the specific producer/consumer pair it fences. "Ensure caches are
coherent" after pure same-core CPU writes to normal memory orders nothing and overstates its
role. Dormant cache-maintenance code is only protected from rot if the dormancy test compiles
the **dangerous** configuration (cacheable + invalidate), not its safe sibling (DTCM, which
preprocesses the maintenance away). Extract the pure address arithmetic (line-round + overflow
guards) into a host-unit-testable inline so the logic that keeps a cacheable buffer coherent is
exercised even when the `SCB_*` call itself is firmware-only.

## 23. A real-time / WCET budget cap is the value the code enforces at the boundary

A constant that documents a hard real-time budget (max active voices, max block) must be the
value **checked at admission**, not a sibling note next to a larger storage bound. If a budget
cap and a storage cap differ, the smaller (the budget) gates admission and the larger only sizes
arrays — and the admission check references the budget constant *by name*. An unenforced budget
constant is fiction: the kernel will run past it and miss the deadline with no guard firing.

## 24. A host simulation compiles the firmware's capabilities; firmware purity is proven against the firmware source set

- A host "simulation" of an embedded target must force the same capability macros the firmware
  selects (the CPU/DSP gate the host cannot auto-detect). Otherwise it renders a *different
  program* than the device — a perf model can be correct on op-counts yet drive a different
  oscillator. If the divergence is intentional, say so at the simulation's surface.
- The include-direction layer law does **not** prove "no host/OS contracts in code the firmware
  compiles": a file can have legal in-repo include edges and still pull `<omp.h>`/`mmap`/`stdio`.
  That guarantee is asserted against the **firmware source set** (the build manifest) — a test
  that the firmware TUs reference no denylisted OS symbols — not against the whole repo's include
  graph. Separation enforced only by which files the manifest happens to compile is separation by
  convention, not by contract.

## 25. A runtime control that cannot act must fail loud

Never accept a parameter, void it, and return success. A stub that returns OK is worse than one
that returns an error, because the whole stack above it (knob, protocol, wrapper) then advertises
a capability the engine does not have. If a control is architecturally a build/synth-time
parameter with no runtime path, the setter rejects a non-identity value (or is marked
unimplemented at its surface), and the doc states where the parameter actually takes effect.

## 26. Build flags have one source of truth, keyed to profile not CPU

Every compiler/optimization/arch flag and its rationale lives in `cmake/profiles.cmake` (the
flag-group SSOT); platform configs (`host-config`, `daisy-config`) assemble per-target lists
from those groups and own only discovery (SDKs, libraries, MCU arch). No flag is invented in a
target file. Two profile philosophies are a contract (see `BUILD_PROFILES.md`): host =
quality-but-optimized; embedded/firmware = aggressively minimal/deterministic. ISA-specific
flags (`-mavx2`, `-mno-avx512f`, `-mcpu=cortex-m7`) are **arch-gated** — emitted only on the
ISA they name, never sprayed across all hosts. The optimization level is a profile property,
not `CMAKE_BUILD_TYPE` (the engine stays optimized under `Debug`). A widening that down-clocks
(AVX-512) is OFF by default and enabled only behind a measured net win, never speculatively.

## 27. A host harness that validates firmware code compiles with the firmware's numeric profile

A host build that runs firmware code for correctness (e.g. the arm32 correctness harness) must
use the firmware's numeric semantics — precise/`SAFE_MATH`, no host-only ThinLTO/`-ffast-math`
— not the desktop profile it happens to inherit. Two reasons: fidelity (it must test what
ships, per #24) and reproducibility. ThinLTO's parallel codegen is not bit-reproducible across
builds; combined with `-ffast-math` on a precision-sensitive fixed-point path it produced a
rare, gross, non-deterministic miscompile (a ~28 dB SINAD collapse, ASan/UBSan-clean) — a
build-determinism defect, never a flake to retry. The default build must be numerically
deterministic; a flag combination that can change *audible* output across builds is a defect.

## 28. Every committed generated artifact has a verify-on-build drift guard and a registry row

A file the build generates and commits in-source must carry an `AUTO-GENERATED by <generator>`
banner, a `--mode verify` (regenerate-and-byte-compare) drift guard wired onto the targets
that consume it, and a row in `GENERATED_ARTIFACTS.md` (enforced by the registry-completeness
test). A custom-command `OUTPUT` for a committed file is a **build-tree stamp**, never the
committed file itself — CMake adds every OUTPUT to `clean`, so the file-as-OUTPUT lets
`--target clean` delete a git-tracked source. Distinguish generated **build inputs** (committed
+ verified) from generated **measurement intermediates** (e.g. census `.s` — ephemeral, never
committed; the committed artifact is the frozen counts). Generated assembly, if it ever ships
as a build input, lives in a committed `asm/` folder under this same contract.

## 29. Every major path logs its entry, its decisions, and its error origins

A stage logs its entry once (INFO); every capability/dispatch decision and every fallback-taken
logs (INFO for the designed path, WARN when degraded); every ORIGINATING `SPECTRAL_ERR_*`/decline
return logs the specific reason at the site (the caller's generic "X failed" is not enough — log
the failing constraint and the inputs). Use the existing structured helpers
(`spectral_log_error_codef`/`_warn_codef`, `spectral_format_resolution_context`); do not invent a
third idiom. **Hot kernels carry NO logging** — the caller logs the aggregate outcome; leaf
recurrences and admissibility predicates stay silent and deterministic. Level discipline:
ERROR = originated failure aborting the op; WARN = degraded but proceeding; INFO = stage/path;
DEBUG/TRACE = loop-grain. Embedded RT paths use only the strippable `SPECTRAL_DBG`-class macros
(no always-on log symbol reachable from the M7 synth path). `log_check` guarantees the CHANNEL
(never raw printf); the presence lint + a decision-logging test guarantee COVERAGE.

## 30. SIMD width is widest-available unless that op's latency is worse, decided per-op

Vector width is a per-operation property, not a global build constant. The default is
**widest-available** (chosen from the ISA the build actually targets), with one carve-out:
an op widens only if its *latency* does not get worse at the wider width. The one real case
today is the AVX-512 down-clock — a wider register file that lowers core clocks and can
net-lose — which is why it is OFF by default (encoded crudely as the global `-mno-avx512f`,
#26) and would be lifted **per-op only behind a measured net win**, never globally or
speculatively. Width selection is made from the SIMDe natural-width predicates
(`SIMDE_NATURAL_{FLOAT,INT}_VECTOR_SIZE_GE`) the kernels already compile against — a separate
hand-maintained width oracle (`OSC_SIMD_WIDTH`) is a second source of truth and a smell to
retire. Do **not** author a wider-tier kernel on hardware that cannot measure it (the
16×Q15@256 and 512-bit float tiers are x86-CI-gated, not written speculatively); a lifted
flag with no kernel changes nothing. The rationale lives here, not only in commit history.
Honest current state: the float oscillator honors widest-available; the Q15 pack8 kernel is
still pinned 8-wide@128 and the `OSC_SIMD_WIDTH` oracle still exists — the per-op-predicate
migration is open work, gated on x86 silicon + a measured win.
