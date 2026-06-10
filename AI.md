# AI.md — working in the Spectral Engine

Real-time spectral analysis + resynthesis engine in C. Analyze audio into
sinusoidal segments; resynthesize with stretch/pitch/timbre on desktop (float,
Metal/CUDA/vDSP) or Cortex-M7 (Q15 fixed-point, Daisy Seed). `AI_CANON.md` holds
the full correctness rules; this file is the orientation.
**Forward mandate (what to work on next):** `docs/core_audit/REVIEWER_HANDOFF.md`.

## Layout
- `spectral_engine/core/` — config, constants, oscillators, windows, segments, hashing, ports
- `spectral_engine/analysis/` — FFT, peak estimation/tracking, processing chain
- `spectral_engine/synth/` — synthesis backends (cpu/arm/sim) + `synth/api/spectral_synth.h`
- `spectral_engine/runtime/` — perf model, console, utils (`spectral_utils.h` lives here)
- `spectral_engine/cmd/` — CLI
- `api/daisy_seed/` — Daisy Seed public API · `core/port/{host,embedded}/` — per-target SIMD/GPU bodies

## Build & test
- `make configure && make` → desktop. Other targets: `simulate`, `embedded_arm`,
  `embedded_arm_float`, `embedded_arm_restricted`, `cuda`, `daisy` (`make help` for the matrix).
- Tests: `cmake --build build --target tests_all && ctest --test-dir build` (the C suite;
  test exes are `EXCLUDE_FROM_ALL`, so build `tests_all` first). Behavioral numeric tests:
  `pytest tests/core_math` (compile+run real C against a Python reference).
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

## Public API (SemVer, 0.x = unstable)
- Desktop: `spectral_engine/synth/api/spectral_synth.h`
- Daisy Seed: `api/daisy_seed/daisy_seed_spectral.h`
- `api/spectral_api.h` does not exist; these two are the contract.

## Reference docs
`docs/core_audit/`: `AI_CANON.md` (rules), `CORE_CONTRACTS.md`, `ACADEMIC_SOURCES.md`
(paper-backed methods), `CHANGELOG.md`, and the `*_PLAN.md` campaign plans.
