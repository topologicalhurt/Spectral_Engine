# Patch notes — Pass 203: U1d — codegen the Metal MSL from the C contract

## Scope

Oscillator-unification step **U1d** (`docs/core_audit/OSCILLATOR_UNIFICATION_PLAN.md`):
the Metal backend no longer hand-mirrors the oscillator + segment-math math as a
string literal in `oscillator.c`. A build-time generator now *derives* the
drift-prone Metal Shading Language (MSL) from the C synthesis contract
(`spectral_osc_formulas.h`, `spectral_segment_math.h`, `oscillator.h`) and emits
the committed header `core/spectral_osc_metal_generated.h`. A `verify` step runs
inside every build and fails the build if the committed MSL drifts from the
current C formulas. This resolves **P4** (Metal string mirror drift) and retires
the old `_Static_assert(SPECTRAL_OSC_FORMULAS_VERSION == 6)` /
`_Static_assert(SPECTRAL_SEGMENT_MATH_VERSION == 1)` version locks. Default build
(incl. Metal output) stays byte-identical.

## The generator

`tools/spectral_tools/generators/metal_osc.py` (`generate` / `verify` modes,
mirroring `gen_resource_hashes.py`). It **derives** the formulas that are pure
algebra from the C contract and **templates** the parts that are intentional
GPU policy:

- **Derived from C** (parsed out of the headers, token-rewritten to MSL):
  - 4 algebraic waveforms — saw, square, triangle, parabola.
  - `oscillator_normalize_phase` — derived from `spectral_normalize_phase`.
  - the fade ramp — validated against `fade_envelope_in`.
  - the 5 segment-math helpers (alpha / beta / d_amp / phase_at / amp_at).
- **Templated GPU policy** (the deliberate C↔MSL divergences, documented inline):
  - **native** `sin` / `asin` on the GPU (not the deg-9 minimax poly the CPU uses).
  - **quadratic** segment phase (`phase_at_f32`), where CPU scalar uses cubic.
  - oscillator switch covers the 6 GPU timbres (SINE,SAW,SQUARE,TRIANGLE,ASIN,
    PARABOLA); QUANTIZED/PWM fold to default→sine (`_EXCLUDED_WAVEFORMS`).
  - intrinsic spelling `floorf→floor`, `fabsf→abs`; constant *values* injected by
    the C preprocessor so the single source stays `spectral_consts.h`.

Token rewrites are table-driven (`_CONST_MAP`, `_FUNC_MAP`, longest-first). If the
C contract is not in the shape the generator expects (e.g. a waveform formula
changes, or `normalize_phase` / the fade ramp stops matching the derived form),
it raises `GeneratorError` loudly rather than emitting silently-wrong MSL.

## What changed

- **`tools/spectral_tools/generators/metal_osc.py`** (new): the generator.
- **`core/spectral_osc_metal_generated.h`** (new, committed): defines
  `oscillator_metal_source` and `spectral_segment_math_metal_source` under the
  Apple guard; banner `AUTO-GENERATED … DO NOT EDIT`,
  `osc_formulas_version=6 segment_math_version=1`.
- **`core/oscillator.c`**: the ~64-line hand-written Metal string + its two
  `_Static_assert` version locks are gone, replaced by
  `#include "spectral_osc_metal_generated.h"`.
- **`core/oscillator.h`**: `extern` declarations for the two generated strings
  under the Apple guard.
- **`synth/backends/gpu/metal/spectral_synth_metal.m`**: dropped both
  `_Static_assert`s; removed the 5 inline segment-math helpers from
  `metalKernelCode` (now provided by `spectral_segment_math_metal_source`,
  prepended at compile time); `newLibraryWithSource` now concatenates
  `%s%s%s%s` = osc source + segment-math source + kernel code; dropped the two
  now-unused includes (`spectral_segment_math.h`, `spectral_osc_formulas.h`).
- **CMake**: `cmake/scripts/run_metal_osc.cmake.in` (runner template) +
  `cmake/targets/metal-osc-codegen.cmake` (`generate_metal_osc` custom command,
  `verify_metal_osc` target gated onto all production targets) + the include in
  `spectral_engine/CMakeLists.txt`.

## The FMA-contraction parity lesson (the subtle one)

The first U1d cut emitted `normalize_phase` as two statements, the same shape as
the C source:

```c
float norm = p * INV_TWO_PI;
return p - TWO_PI * floor(norm + 0.5f);
```

Metal output matched the baseline for `sin_440hz` (all 8 timbres) and for
`shakespeare` t=5,6,7 — but **differed** for `shakespeare` t=0–4. The binary was
deterministic (×3), so this was a real semantic delta, not noise.

Cause: Metal compiles with `fastMathEnabled=NO` but **still contracts** `mul+add`
into an FMA. The named intermediate `float norm` forces a round-to-f32 that
*defeats* the contraction the inlined form allows. The contracted-vs-rounded
result differs ~1 ULP near half-integer phase boundaries → occasionally flips
`floor()` → perturbs the normalized phase. Invisible on a steady tone, byte-
different on speech. Parentheses do **not** force rounding — only a variable
assignment does — so the fix is to emit the single inlined return:

```c
return p - TWO_PI * floor((p * INV_TWO_PI) + 0.5f);
```

After regenerate + rebuild, all 16 Metal outputs are byte-identical to the
baseline. The output-identity guard caught what pure string review could not.

## Verification (measured, not asserted)

```text
- 5 production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 5/5 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity).

- Metal output byte-identity (SHA-256 of output/out_c.wav), 8 timbres x
  {sin_440hz, shakespeare_he_saw_the_cat} = 16 cases, vs /tmp/u1c_baseline:
    IDENTICAL (16/16)

- Drift guard bites: perturbing the saw formula in a /tmp copy of the contract
  (-SPECTRAL_INV_PI -> -0.5f * SPECTRAL_INV_PI) -> `verify` exit 1;
  the real contract -> exit 0.
- `verify_metal_osc` runs in-build and passes; generator `verify` is idempotent
  against the committed header.
```

## Status

Metal MSL is now generated from the C contract, not hand-mirrored. P4 (Metal
drift) is resolved structurally: a build-time `verify` makes drift a build
failure, replacing the version `_Static_assert`s. Default build byte-identical
(16/16 Metal outputs), 5 builds + ctest green, drift guard proven. With U1c (P2)
and U1d (P4) done, the per-sample dispatch and the Metal mirror are both single-
sourced; the remaining oscillator work is the band-limited *quality* file.

## Proposed next pass

**U3** — optimize `core/spectral_osc_bandlimited.c` (the CPU-float quality path,
opt-in, not golden): profile additive/PolyBLEP first, then oversample into a
reusable scratch buffer instead of malloc-per-call, and fold the symmetric FIR
(65 taps → 33). Measure-first; no behavior change without a measured win. Then
**U2** (adversarial audit of the optimized band-limited file).
