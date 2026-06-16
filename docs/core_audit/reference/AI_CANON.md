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
