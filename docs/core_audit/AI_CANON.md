# AI/Human Canon for Spectral Engine Core Work

This file records repeated mistake classes that are canonically incorrect for this repository. Treat it as a guardrail before touching `spectral_engine/core`, `spectral_engine/analysis`, `spectral_engine/synth`, or GPU/embedded parity code.

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
