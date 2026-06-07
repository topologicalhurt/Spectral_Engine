# Discipline-Specific Findings

## Mathematician / numerical analyst perspective

### Core issue: no sufficiently explicit mathematical contract layer

The source has many comments that say “canonical,” but canonical status is not enforced. A mathematical kernel needs small reference functions whose contracts can be stated and tested independently of performance code.

Required contracts:

- phase wrapping interval and edge behavior;
- waveform range, symmetry and discontinuities;
- fade boundary values and monotonicity;
- FFT scaling convention;
- window coherent gain and RMS gain;
- interpolation estimator assumptions;
- amplitude units and dB conversion policy;
- fixed-point scaling and saturation behavior.

### Phase convention

The previously observed wrap formula violated the elementary invariant `wrap(0) = 0`. That kind of error is especially dangerous because sine output can mask it while other waveform families reveal it.

### Frequency estimation

Three-bin interpolation is not interchangeable across magnitude, power, log magnitude and complex spectra. Estimators from Quinn, Jacobsen/Kootsookos and Candan have assumptions. A correct implementation must encode those assumptions in function names and tests.

The safe public estimator path must validate the selected next-frame bin before
computing temporal slope. The expression `best_next_bin - bin` is only defined
after proving that `best_next_bin` is non-negative, in range, and inside the
tracker's local `[bin - 1, bin + 1]` search contract.

### Numerical approximations

Approximate `sqrt`, `atan2`, and sine must be treated as approximations with error budgets, not drop-in replacements. For synthesis, local errors can become systematic timbral artifacts.

### Units

`omega`, `df`, `alpha`, `beta`, `stretch`, and `pitch_factor` need a documented unit derivation. Current quadratic phase appears plausible because `freq_step_df` includes `0.5/hop`, but that relationship is distributed across files and should be encoded as a contract test.

## DSP engineer perspective

### STFT analysis

The implementation must define:

- whether the analysis window is symmetric or periodic;
- whether FFT output is normalized;
- how magnitude and phase are scaled;
- whether vDSP and FFTW paths are equivalent;
- whether chunked and full-matrix paths produce the same result.

### Sinusoidal model

The code currently resembles frame-pair segment emission more than full partial tracking. That can be a valid design, but it must not be documented as full sinusoidal partial tracking unless identities are maintained across multiple frames.

### Oscillator correctness

The oscillator set includes discontinuous waveforms. These are not bandlimited and can alias heavily when pitched. If these timbres are intended as creative effects, document that. If they are intended as high-quality oscillators, implement bandlimited forms or restrict usage.

### Envelope correctness

Boundary fades are essential to avoid clicks. The corrected fade sign must be tested on every backend.

### Thresholding

`db_thresh` is converted to linear amplitude and squared before multiplying by global maximum magnitude squared. That is reasonable as a relative threshold, but it should be documented as relative-to-frame/global max, not absolute dB SPL or dBFS unless calibrated.

## Performance engineer perspective

### Fast paths need benchmark-backed survival

The code includes prefetches, SIMD scans, batch queues, overallocated arrays, and approximate math. Each should survive only if it has:

- a scalar reference comparison;
- a benchmark showing benefit on target hardware;
- a fallback for small sizes;
- a documented error budget if numerical.

Benchmark timing must exclude reference/error bookkeeping. A benchmark that
computes canonical answers inside the timed loop is measuring the harness, not
the estimator or kernel path being optimized.

### Memory layout

`Segment` being 64 bytes is cache-line friendly for random per-segment access but can waste memory bandwidth when only seven floats are used. The code already has `SegmentGpu` and `SegmentCompact`, which indicates pressure toward SoA/AoSoA layouts. The plan should evaluate whether CPU synthesis also benefits from SoA active blocks.

### Allocation

Hot-loop allocation and unbounded virtual allocation are not acceptable for a kernel. Preflight sizing or arenas are needed. The bundled patch bounds the initial tracker capacity but does not complete the arena refactor.

### Threading

OpenMP parallelism must be size-gated. For small files or short segment lists, thread overhead can dominate. Deterministic reductions must be available for reproducible mode.

## Hardware designer / embedded perspective

### Real-time boundedness

A real-time kernel requires worst-case bounds, not average-case benchmarks. Every path intended for embedded or callback use must have:

- maximum allocation count of zero inside the callback;
- maximum active segment count;
- maximum block processing time;
- bounded stack usage;
- defined behavior on overflow.

### Fixed-point arithmetic

Q15/Q31 paths need explicit saturation and scaling proofs. Multiply/accumulate behavior must be tested for worst-case overlapping segment sums.

### Memory hierarchy

STM32H7-style memory hierarchy makes DTCM/ITCM/SRAM/SDRAM placement consequential. Data that is repeatedly touched in the audio callback should not live in slow external memory unless DMA/cache coherency is explicitly handled.

### GPU/kernel parity

GPU tiling is a hardware-conscious choice, but GPU code must be formula-equivalent to CPU code. String-embedded Metal formulas are a drift hazard.

## Software architecture perspective

### Minimal kernel boundary

The current engine mixes kernel math, backend fallback policy, logging, runtime cache state and CLI concerns. A minimal reusable kernel should not own user-facing policy.

Thin alias wrappers are a separate architecture smell. They look like
abstraction but only increase the number of names reviewers must track. Kernel
helpers should either enforce a contract, cross a boundary, or disappear in
favor of the canonical helper.

Dead duplicate kernel files are worse than comments because they can drift from
the compiled path while still looking authoritative. If a file is not in the
manifest and not included by live code, it must not mirror active kernel logic.
Repeated ownership cleanup deserves the same treatment: shared resources should
have one cleanup helper instead of separate success, failure and destroy
versions.

Raw allocation arithmetic is another repeat-offender pattern. Use the existing
safe allocation helpers for arrays and scratch strings so overflow policy stays
centralized instead of reimplemented one callsite at a time.

### Public API

A future `spectral_kernel.h` should expose only:

- configuration structs;
- preflight sizing;
- analysis entry point;
- synthesis entry point;
- free/destroy functions;
- error/status codes.

Raw analysis/tracking APIs that accept already-computed magnitude/phase rows
must expose the active window/estimator profile. A compatibility wrapper may
default to Hann, but explicit APIs must fail closed when a caller-provided
descriptor is invalid.

### Audit docs

The durable review record is the git log plus `docs/core_audit/CHANGELOG.md`.
Validation matrices and contract explanations live in the relevant canon doc
(`CORE_CONTRACTS.md`, `ACADEMIC_SOURCES.md`), not in per-change sidecar files
that tests or audits could come to depend on.

### Global state

Process-global and thread-local caches may be useful, but they complicate embedding. Any global state should be optional and resettable; pure context objects are preferred.

## Test engineer perspective

### Required fixtures

- silence;
- DC;
- impulse;
- exact-bin sine;
- fractional-bin sine;
- chirp;
- two-tone resolution stress;
- high segment-density stress;
- randomized deterministic seed fixtures.

### Required metrics

- max absolute error;
- RMS error;
- phase error;
- frequency offset error;
- segment count drift;
- CPU time;
- memory high-water mark;
- backend parity error.

## Final position

The engine can become a high-performance kernel, but only if correctness is made explicit before optimization. The current highest-value work is not adding another backend or heuristic; it is stabilizing the mathematical contract and making every optimized path prove itself against that contract.
