# Patch notes — Pass 198: SIMD oscillator — cubic MQ phase, intuitive comments, and a parity/seam audit

## Scope

User-requested, outside the numbered optimisation track: *"The SIMD oscillators
only evaluate the quadratic phase model — implement the cubic in SIMD as well;
the SIMD code needs to be intuitively commented (what exists as well) and we
should probably do another audit on it."* Plus a separate **investigate-only**
question about harsh-sounding timbres (asin/quantized/square) — diagnosed below,
no code change for that part by request.

**Outcome:** the host SIMDe oscillator now evaluates the full cubic
McAulay-Quatieri phase polynomial (gated under `SPECTRAL_PRECISE_PHASE`), the
now-redundant cubic→scalar bypass is removed, a latent triangle parity/seam bug
is fixed, a stale "Padé" comment is corrected, and both SIMD oscillator files are
commented throughout. **Default build output is bit-identical** (proof below).

## The headline audit finding — the SIMD oscillator is dead code today

`g_osc_dispatch` is statically initialised to `OSC_DISPATCH_ALL_SCALAR`
(`oscillator.c:11`) and the only writer, `osc_set_dispatch()` (`oscillator.c:13`),
**has no caller anywhere** in `spectral_engine/`, `api/`, `examples/`, `tools/`,
or `tests/` (verified by grep). Nothing ever writes `OSC_MODE_CPU_SIMD` /
`OSC_DISPATCH_ALL_SIMD` into the dispatch word. Therefore
`OSC_GET_MODE(g_osc_dispatch, timbre)` returns `CPU_SCALAR` for every timbre, the
`OSC_MODE_CPU_SIMD` branch in `timbre_synth_segment` is unreachable, and
`osc_simd_segment_*` is **compiled but never invoked in any current build**.

Consequences:
- This whole pass changes **zero** runtime output today; it is forward-looking
  infrastructure that goes live only when dispatch is wired (the separately
  maintainer-gated **O2-A**, "default vectorised sine").
- It is the cheapest possible time to correct the SIMD↔scalar parity gaps
  (below): there is no golden to break because the path is not yet reachable.

## Changes

### 1. Host cubic phase — `core/port/host/oscillator_simd.c`

`osc_simd_fused_sustain` previously evaluated only the quadratic Horner
`phase0 + j*(alpha + beta*j)` in its vectorised sustain loop; cubic segments were
shunted to scalar by a bypass in `oscillator.c`. Now:

- The scalar **fade-in / sustain-tail / fade-out** regions always evaluate
  `spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, j)`, exactly
  mirroring the canonical scalar synth (`oscillator.c:synth_segment_scalar`).
  With the default (no-linkage) coefficients `c2==beta, c3==0` this is
  bit-identical to the quadratic helper (IEEE multiply commutes; the `+j*c3==0`
  term adds a hard `+0.0f`).
- The vectorised **sustain hot loop** is gated:
  `#if SPECTRAL_PRECISE_PHASE` → cubic Horner
  `phase0 + j*(alpha + j*(c2 + j*c3))`, written op-for-op against the scalar
  cubic helper (so SIMD-cubic == scalar-cubic bit-for-bit per lane); `#else` →
  the original quadratic Horner, so the **default build pays nothing extra in the
  inner loop** (speed-first default, quality behind the flag).

This gating is provably safe because `c2`/`c3` can only differ from `(beta, 0)`
inside `#if SPECTRAL_PRECISE_PHASE` (`spectral_synth_internal.c:332-342`), so the
quadratic fast path can never drop a chirp the scalar path would have kept.

### 2. Triangle parity + intra-segment seam fix — `wave_triangle_4`

`wave_triangle_4` computed `1 + (-2)*|rads/pi|`, while the scalar twin
`spectral_osc_triangle` (and the fade-region lane `wave_triangle_1` that calls it)
computes `(1 - |rads|/pi)*2 - 1`. These are algebraically equal but **round
differently** (up to 1 ULP). Because the SIMD function uses `wave_triangle_1` in
the fade regions and `wave_triangle_4` in the sustain, a single triangle segment
could **seam-split** at the fade/sustain boundary, and the SIMD sustain disagreed
with the scalar oscillator. `wave_triangle_4` is rewritten to the scalar op order,
closing both gaps. (Audited the other waveforms: saw/parabola/square/sine SIMD
lanes are already bit-identical to their scalar twins — see "Parity audit".)

### 3. Stale comment correction — `simde_fast_sin_ps`

The header claimed *"Vectorized Padé [5/4] sine approximation"*. The code is a
**degree-15 odd Taylor (Maclaurin) polynomial** (coeffs `1/3! … 1/15!`,
alternating sign), matching `spectral_fast_sin_inline` exactly — and the Padé
[5/4] kernel was in fact *removed* for excessive endpoint error (see that
function's header in `spectral_osc_formulas.h`). The default branch is exact
per-lane `sinf`. Comment corrected; this is documentation only.

### 4. Removed redundant bypass — `core/oscillator.c`

The `#if SPECTRAL_PRECISE_PHASE` block that forced cubic segments onto the scalar
path (because SIMD was quadratic-only) is deleted; the host SIMD oscillator now
honours cubic under the same flag, and every non-SIMD timbre (e.g. asin) already
falls to the bottom scalar path which uses the cubic helper. Replaced with a
short comment explaining why no redirect is needed.

### 5. Embedded — `core/port/embedded/oscillator_simd.c` (comments only)

The CMSIS oscillator is intentionally left **quadratic** and now documents why:
cubic coefficients live in the 64-byte desktop/GPU `Segment`'s pad words, and the
32-byte embedded `Segment` has no room (`spectral_common.h:31-36`), so
`spectral_segment_has_cubic()` is structurally false on this profile — `c2==beta,
c3==0` always, and the quadratic helper is exact and complete. A cubic Horner
would only add per-sample Cortex-M cost for coefficients that can never be
non-trivial. (The embedded SIMD has no fade/sustain seam: its batch and scalar
tail use one waveform form per timbre. Its triangle uses the `1-2|·|` form, which
differs from the canonical `spectral_osc_triangle` by ≤1 ULP — latent, dead code,
left as-is to avoid extra CMSIS ops; flagged for the eventual dispatch-wiring
golden.) Every function gained intuitive comments.

## Parity audit (SIMD sustain lane vs. scalar twin)

```text
saw       wave * -1/pi                       bit-identical
square    cmpgt(rads,0)?1:-1                  bit-identical (NaN/-0 -> -1, as scalar)
triangle  (1-|rads|/pi)*2-1                   FIXED this pass (was 1-2|·|, 1-ULP + seam)
parabola  1 + sq*(-1/pi^2)                    bit-identical (negate is exact)
sine      sinf (default) / deg-15 Taylor      bit-identical (same libm / same coeffs+order)
quantized cvtt truncation + domain guard      mirrors scalar (pre-existing careful guard)
pwm       finite/width guard + half compare   mirrors scalar (pre-existing careful guard)
phase     normalize: p-2pi*floor(p/2pi+0.5)   bit-identical to spectral_normalize_phase
```

## Why this is golden-safe (bit-identity by construction)

```text
1. The osc_simd_* path is unreachable in every current build (dead dispatch,
   proven above), so no current output — default or precise — can change.
2. The oscillator.c edit is entirely inside #if SPECTRAL_PRECISE_PHASE (default
   0) plus a comment, so the default-build codegen of oscillator.c is unchanged.
3. The scalar synth path (the one actually used) is untouched.
Therefore the default desktop output is bit-identical by construction; PASS196
already empirically cmp'd the cumulative uncommitted stack to pristine HEAD
(c7eff0f0c8) as byte-identical, and this pass adds only dead-in-default code on
top.
```

## Verification

```text
- Five production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float) — only the pre-existing benign
  -mavx2 / -mno-avx512f notes.
- ctest 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift).
- The cubic SIMD branch (gated behind SPECTRAL_PRECISE_PHASE, not compiled in the
  default build) was exercised two ways: (a) -fsyntax-only -Wall -Wextra compile
  of oscillator_simd.c and oscillator.c with -DSPECTRAL_PRECISE_PHASE=1 — clean;
  (b) a full from-scratch desktop build with CMAKE_C_FLAGS=-DSPECTRAL_PRECISE_PHASE=1
  — links clean, no warnings beyond the benign notes.
- Default CPU backend still renders sin_440hz.wav (657 segments) correctly.
```

## Harsh-timbre investigation (no code change — by request)

Root cause is **aliasing from naive, non-bandlimited per-partial waveforms**.
Resynthesis renders each detected partial as the chosen timbre *at that partial's
frequency*; the timbres with hard discontinuities or sharp features generate
overtones far above their fundamental, and any overtone past Nyquist folds back
as an inharmonic, dissonant tone:

- `square` / `pwm` — instantaneous ±1 jumps (infinite-bandwidth step).
- `saw` — the hard −π↔+π wrap is a step discontinuity.
- `quantized` — a staircase, i.e. a stack of steps (very bright).
- `asin` — cusps with near-infinite slope at ±π/2 (the sharp endpoints the
  domain-clamp protects are exactly the brightest part).
- `triangle` / `parabola` are continuous (only slope discontinuities) → milder,
  which matches perception.

None of these are bandlimited (no PolyBLEP/BLIT/BLAMP, no oversampling, no
Nyquist-limited harmonic cap), so it is a consequence of the naive shapes, **not
a deliberate timbral choice**. Options to make them "musical" (for a later,
flag-gated, golden-signed pass): PolyBLEP/BLEP step- and slope-correction at the
discontinuities; a Nyquist-limited additive harmonic count per partial; 2–4×
oversample-then-decimate; or a gentler nonlinearity for asin/quantized. Deferred
pending maintainer direction.

## Status

SIMD cubic phase **implemented** (host, gated), commented, audited; triangle
parity/seam **fixed**; stale comment **corrected**; redundant bypass **removed**;
embedded **documented** as quadratic-by-design. Default output **bit-identical**
(dead-dispatch + disabled-#if). The SIMD path goes live only with **O2-A**
(dispatch wiring), which remains maintainer-gated and will want its own signed-off
"arm32/cpu SIMD exact" golden at that time. Harsh-timbre fix scoped but
**deferred** per the investigate-only request.
