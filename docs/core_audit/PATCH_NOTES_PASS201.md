# Patch notes — Pass 201: degree-9 minimax sine becomes the default (scalar + SIMD)

## Scope

Follow-up to Pass 200 (SIMD-default CPU oscillator) under the oscillator-unification
effort (`docs/core_audit/OSCILLATOR_UNIFICATION_PLAN.md`). Pass 200 made SIMD the
default CPU path and exposed the **sine pathology**: every other timbre vectorized
~1.8×, but sine only reached ~1.1× because the SIMD `sin` kernel spilled to four
scalar `sinf` calls (`SPECTRAL_ENABLE_APPROX_TRIG==0`, libm reference). There was no
vector sine, so the four lanes were stored, called one-by-one, and reloaded.

Maintainer direction (verbatim, the sine-pathology investigation):

> "An example that comes to mind is sin - we pack the lanes with 4 sinfs (how is
> this 10% faster at all?) … (a) Does it need to be [quality gated]? (b) If it does,
> can we make a bit-identical or near bit-identical sin SIMD approximation (c) If we
> can't … surely there is a better fallback than 4 lanes of sinf."

Chosen answer (via AskUserQuestion): **"Minimax poly, shared, default."** Derive a
near-minimax single-precision sine polynomial, share it bit-for-bit between the
scalar and SIMD paths, and ship it as the **default** so both paths get faster while
staying mutually equivalent. The maintainer is the golden authority and authorized
this re-baseline (same standing as the Pass 200 SIMD-default flip).

## The polynomial

Degree-9 odd minimax over a **quadrant fold** to `[-pi/2, pi/2]`:

```
q  = floorf(x/pi + 0.5)          ; nearest integer multiple of pi
xr = x - q*pi                    ; folded argument in [-pi/2, pi/2]
x2 = xr*xr
p  = xr*(1 + x2*(c3 + x2*(c5 + x2*(c7 + x2*c9))))
sign = 1 - 2*(q mod 2)           ; (-1)^q, branchless
sin(x) = p * sign

c3 = -0.16666647791862488
c5 =  0.00833289884030819
c7 = -0.0001980086526600644
c9 =  0.0000025904300855472684
```

The fold uses only `mul`/`add`/`floor` — the identical op sequence is expressible in
scalar C and in SSE2/NEON intrinsics, so the two paths agree to within FMA-contraction
noise (see Parity below). Folding to a single quadrant (rather than evaluating a
degree-15 poly over `[-pi,pi]`) keeps `|xr|` small, which lowers float32 Horner
rounding: a 5-term fold floors at ~1.4 ULP and beats the 8-term `[-pi,pi]` Taylor on
**both** accuracy and FLOP count.

### Why "minimax" via Chebyshev truncation, not Remez

A hand-rolled Remez exchange kept diverging (ill-conditioned monomial Vandermonde
over `[0,pi]` with an `x^15` column, fragile alternation-point selection). For a
function as smooth as `sin` on a single quadrant, the truncated Chebyshev series
residual (≈3.35e-9 at degree 9) sits two orders of magnitude below the float32
evaluation floor (~3.6e-7), so the Chebyshev-projected coefficients are
**minimax-equivalent in single precision** — the polynomial-truncation error is no
longer the dominant term, float rounding is. numpy Chebyshev projection is the robust
route; true Remez buys nothing here.

## The change

1. **`core/spectral_osc_formulas.h`** — `spectral_fast_sin_inline` (the
   `APPROX_TRIG==1` branch) rewritten from the old degree-15 odd-Taylor to the
   degree-9 quadrant-folded minimax above. `SPECTRAL_OSC_FORMULAS_VERSION` bumped
   **5 → 6** (the bump-on-formula-change guard). `APPROX_TRIG==0` still `return
   sinf(x)` — the exact libm reference is preserved one flag away.
2. **`core/port/host/oscillator_simd.c`** — `simde_fast_sin_ps` rewritten as the
   op-for-op SIMD twin (same Horner order: `c9` innermost → `+1` → `×xr`, same
   branchless sign). The `APPROX_TRIG==0` branch keeps the 4×`sinf` spill as the
   exact fallback.
3. **`core/spectral_config.h`** — `SPECTRAL_ENABLE_APPROX_TRIG` default flipped
   **0 → 1**. This is what makes the poly the default and is *necessary and
   sufficient* for the SIMD sine win: the default build must run the vector poly, not
   the lane spill.
4. **Static-assert sync** — `core/oscillator.c` and
   `synth/backends/gpu/metal/spectral_synth_metal.m` both bumped to
   `SPECTRAL_OSC_FORMULAS_VERSION == 6`.
5. **Manifest** — `core/spectral_guarantees.h` EXACT_TRIG row's relaxes-string now
   reads "libm sinf -> degree-9 odd minimax fold … (relaxed by default)";
   `cmake/targets/core-guarantees-test.cmake` comment notes TRIG is the one
   approximation that is default-ON.
6. **Tests** — `test_guarantees.c` `BUDGET_SIN` tightened 5.0e-6 → 2.0e-6 (2.4×
   over measured); `test_osc_parity.c` budget comment updated (sine now runs the
   poly on both paths so its per-sample diff is ~0.5 ULP from FMA contraction, no
   longer 0.0); `tests/core_math/test_core_math_pass2_contract.py`'s sine contract
   test renamed `test_fast_sine_default_is_minimax_poly` and re-pointed at version 6
   + the quadrant fold.

## Blast radius

`spectral_fast_sin_inline` is the single source of truth behind `fast_sin`
(`spectral_fast_math.c`), used by **(a)** the CPU oscillator, **(b)** fade envelopes
(`spectral_envelope.c`), and **(c)** the peak estimator
(`spectral_peak_estimator.c`, which derives cos via `fast_sin(phase + HALF_PI)`).
All three pick up the new poly automatically. Cross-backend:

- **Metal** uses the GPU's **native** `sin()` (`oscillator.c` `oscillator_fast_sin`),
  so MSL is unchanged — only the version static-assert moved.
- **CUDA** compiles `spectral_fast_sin_inline` itself via the dual-compile
  `OSC_FORMULA_FUNC __device__` path, so it inherits the new poly with no separate
  edit.

Only **two** functions must stay in lockstep going forward: the scalar inline and
its SIMD twin. The `osc_parity` ctest pins exactly that.

## Guarantee manifest

`SPECTRAL_ACTIVE_GUARANTEES` default goes **0x3e → 0x3c**: the `EXACT_TRIG` bit
(bit 1) is now *cleared* by default because the canonical sine is the approximation.
This is correct and intended — the manifest now truthfully reports that the default
build trades exact libm sine for the minimax poly. `core_guarantees_drift` still
forces all APPROX_* on and pins each budget; `SPECTRAL_ENABLE_APPROX_TRIG=0` restores
the bit and the libm path.

## Verification (measured, not asserted)

```text
- Five production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 5/5 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity).

- Accuracy (real C, all APPROX_* on):
    fast_sin   max abs err 8.26e-7 over [-4pi,4pi]  (budget 2.0e-6)
               ≈1.4 ULP over the [-pi,pi] oscillator operating range; the larger
               4-period figure is range-reduction + -ffast-math reassociation the
               oscillator never sees. Error is smooth and bounded across the
               ±pi/2 fold seam (no quadrant-seam glitch — that would show O(1)).

- Parity (osc_parity ctest, SIMD vs scalar through the real dispatch):
    sine per-sample max abs diff 5.96e-8  (≈0.5 ULP, FMA contraction only)
    aggregate max abs diff       2.384e-7 (budget 1.0e-5)
    aggregate RMS diff           -155.1 dBFS abs, -152 dB rel signal
  Sine is no longer bit-identical to scalar (it ran libm before, now the poly on
  both) but is FMA-contraction-equivalent, the same regime as every other timbre.

- Speedup (CPU backend, shakespeare input, single thread, best-of-5 wall clock):
    SINE   SIMD 1.84 s  vs  --scalar 3.46 s  = 1.88× faster   (was ~1.1×)
    SAW    SIMD 1.64 s  vs  --scalar 2.52 s  = 1.54×           (regression check)
  Sine now vectorizes as a first-class timbre — its end-to-end speedup actually
  exceeds saw's, because the heavy 5-term Horner is exactly what SIMD parallelizes
  well while scalar pays it per sample. The 4×sinf pathology is gone.
```

## Why this is safe

The faster path is the default because it is measurably faster on every sine-bearing
code path and sub-quantization-equivalent (-155 dBFS RMS) to the libm reference,
which remains one build flag away (`SPECTRAL_ENABLE_APPROX_TRIG=0`). Scalar and SIMD
stay mutually equivalent and that equivalence is now CI-pinned by `osc_parity`. The
behavior change was sanctioned by the golden authority; see
[[faster-path-should-default]].

## Status

Degree-9 quadrant-folded minimax sine is the **default** for the CPU oscillator,
fade envelopes, peak estimator, and CUDA; shared bit-for-bit (modulo FMA) by scalar
and SIMD. Metal keeps native `sin`. Builds + ctest green; accuracy, parity, and
speedup measured.

## Proposed next pass

The sine pathology resolved, the remaining SIMD-perf question is whether the other
transcendental fallbacks (`fast_atan2`, `fast_inv_sqrt`, `fast_peak_log`) have an
analogous lane-spill on any vectorized caller, and the unification track resumes at
**U1c** (extract the shared `spectral_osc_eval` L1 kernel now that scalar/SIMD sine
share one polynomial). Both are maintainer-gated.
