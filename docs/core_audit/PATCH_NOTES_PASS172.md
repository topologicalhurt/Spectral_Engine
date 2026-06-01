# Patch notes — Pass 172: CTF sweep increment 12 — oscillator asin NaN injection (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` (and the CLI/converter layer) and fix it in place.
This pass sweeps the **canonical oscillator math contract**
(`core/spectral_osc_formulas.h`) — the single source of truth for all 8 waveform
generators, shared by the CPU scalar path, the CUDA device path (includes the
header directly), and mirrored as MSL strings for the Metal backend
(`core/oscillator.c`).

The one defect is a **NaN injection in `spectral_osc_asin`**: it feeds the output
of `spectral_normalize_phase` straight into `asinf` with no domain clamp, but the
normalizer does **not** guarantee a strict `[-pi, pi)` result.

```c
/* core/spectral_osc_formulas.h (pre-fix) */
OSC_FORMULA_FUNC float spectral_normalize_phase(float p) {
    float norm = p * SPECTRAL_INV_TWO_PI;
    return p - SPECTRAL_TWO_PI * floorf(norm + 0.5f);   /* aims for [-pi, pi) */
}

OSC_FORMULA_FUNC float spectral_osc_asin(float rads, float width) {
    (void)width;
    return asinf(rads * SPECTRAL_INV_PI);               /* arg can be < -1 -> NaN */
}
```

`spectral_normalize_phase` reduces an arbitrary accumulated phase into `[-pi, pi)`
*mathematically*, but in IEEE-754 single precision the constants `TWO_PI` /
`INV_TWO_PI` are inexact and the `p - TWO_PI*k` subtraction rounds, so the computed
result can land a fraction of an ULP **below** `-pi` (and the overshoot grows with
`|p|` via catastrophic cancellation). The header's own waveform contract even
documents "All take rads in `[-pi, pi)` (output of `spectral_normalize_phase`)" —
a contract its normalizer cannot actually honor at the boundary.

For the saw / square / triangle / parabola generators a tiny `[-pi, pi)` overshoot
is harmless (they produce a value a hair outside `[-1, 1]`, no exception). But
`asinf` has domain `[-1, 1]`: for `rads` just below `-pi`, `rads * INV_PI` drops
just below `-1.0f` and `asinf` returns **NaN**. That NaN flows out as the segment's
`wave` and the synth accumulates it (`dst[j] += amp * wave`), poisoning the output
sample — and, because the downstream normalization stage takes a peak over the
buffer, a single NaN can propagate to the entire rendered file.

`spectral_osc_asin` is live, not dead code: it is wired into the runtime
`timbre_table[TIMBRE_ASIN]` (`oscillator.c:23`), dispatched by the public
`timbre_oscillator()` and by `synth_segment_scalar` (ASIN has no SIMD variant, so
it *always* takes the scalar `spectral_osc_asin` path), and `TIMBRE_ASIN = 4` is a
user-selectable timbre (`./spectral in.wav 4 ...`; CLI usage line
"4=asin"). The same unclamped `asin(rads * INV_PI)` is duplicated in the Metal MSL
string (`oscillator.c:182`) and inherited by CUDA via the shared header. This is
the campaign's recurring "missing domain/range guard" defect class.

## Change

```text
1. asin domain clamp on the canonical generator  (NaN injection / math contract)
   core/spectral_osc_formulas.h
   - spectral_osc_asin: clamp rads*INV_PI into [-1, 1] before asinf(). The
     endpoints map to the correct +/- pi/2 boundary value (the true limit of the
     asin waveform at phase +/- pi). Covers the CPU scalar path AND the CUDA
     device path (CUDA includes this header directly).

2. Mirror the clamp in the Metal MSL string  (cross-backend parity)
   core/oscillator.c
   - oscillator_metal_source: TIMBRE_ASIN now returns
     asin(clamp(rads * INV_PI, -1.0f, 1.0f)) (MSL clamp() is well-defined).

3. Bump the formula-parity version 4 -> 5  (enforced contract)
   core/spectral_osc_formulas.h        SPECTRAL_OSC_FORMULAS_VERSION 4 -> 5
   core/oscillator.c                   _Static_assert(... == 5)
   synth/backends/gpu/metal/spectral_synth_metal.m   _Static_assert(... == 5)
   The header mandates that any formula change bump this version; three
   compile-time guards check it so a stale MSL mirror fails the build (and in fact
   did, until all three were updated — see Verification).
```

The other 7 generators were audited and left unchanged — none can produce a
non-finite output from a finite, slightly-out-of-`[-pi, pi)` input:
saw/square/triangle/parabola are polynomials/comparisons; `quantized` and `pwm`
already guard `isfinite` and (for quantized) the `INT_MIN..INT_MAX` cast range;
`sine` uses `sinf` (defined for all finite inputs). Only `asin` (and a future
`acos`, not present) carries a bounded domain, so the clamp is applied at exactly
the one fragile site — KISS, no speculative guards elsewhere.

`spectral_normalize_phase` itself was deliberately **not** modified: tightening it
to a hard `[-pi, pi)` clamp would alter the rads fed to *every* backend and *every*
waveform (a broad, risky behavioural change against a constant that the other
generators tolerate), whereas the asin-domain clamp is local to the one function
whose math actually breaks. Fix the violated contract at the consumer that cannot
absorb the overshoot, not the shared producer the rest of the pipeline depends on.

## Why this is correct and behaviourally inert for in-domain inputs

The clamp only rewrites `rads * INV_PI` values with magnitude `> 1.0f` — exactly
the set that previously produced NaN. For every in-domain input (`|arg| <= 1`) the
code path is identical (`a = rads*INV_PI; asinf(a)`), and the non-ASIN timbres
never call this function at all. So the change is a no-op for all
previously-finite output and only *replaces NaN with the correct +/- pi/2 endpoint*.

## Finding / reachability

The defect is real and reachable, established at three levels (the fix changes
host codegen, so byte-identical-binary does not apply; verified by output/value
parity instead):

```text
- Function level (definitive). A direct repro of the pre-fix formula over a wide
  phase sweep produced 262763 NaN from osc_asin, with the exact boundary
  nextafterf(-pi) -> arg = -1.0000001 -> asinf = NaN. Against the FIXED header the
  same path yields 0 NaN over 409,600,001 swept phases, and nextafterf(-pi) now
  returns -1.57079625 (= -pi/2, the correct endpoint).
- Realistic synth trajectory (concrete reachability). Scanning the actual segment
  phase formula phase0 + j*(alpha + beta*j) with real musical frequencies
  (261.6/440/880/1318.5/2093 Hz at 44.1 kHz), varied per-segment phase0, and long
  segments — 48,400,000 calls — the pre-fix osc_asin produced 330 NaN; the fixed
  one produced 0. So a real synth phase trajectory does land in the NaN band; it is
  just rare (~7 per million samples), which is why two specific CLI fixtures did
  not happen to hit it (see below).
```

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float. The version bump correctly tripped all three
  SPECTRAL_OSC_FORMULAS_VERSION static-asserts (oscillator.c, spectral_synth_metal.m)
  on the first build; updating them to == 5 cleared it. Only the pre-existing benign
  -mavx2 / -mno-avx512f unused-command-line-arg notes on host; no new warnings.
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- integration parity (output is float32 WAV, so NaN would be byte-visible):
  desktop, sin_440hz.wav, n_fft=1024 hop=256 thresh=-70 threads=1 backend=cpu
  (ASIN always uses the scalar osc_asin path), HEAD (git-stashed) vs FIXED:
    * SINE timbre  -> out_c.wav BYTE-IDENTICAL (sha 256 f888b992...) — proves the
      change is inert for non-ASIN timbres despite the header/codegen edit.
    * ASIN timbre  -> out_c.wav BYTE-IDENTICAL — this fixture's 88200-sample phase
      trajectory never lands in the NaN band (consistent with the ~7/million rate),
      so HEAD already produced 0 NaN here; the fix changes nothing on this input.
    * ASIN at aggressive pitch/stretch (24/48/-48 semitones, stretch up to 16x):
      float-WAV NaN scan = 0 on both — the synth keeps per-segment phases bounded,
      so the band is still rarely hit through this particular pipeline.
  The function-level (409.6M, 0 NaN) and realistic-trajectory (48.4M, 330 -> 0)
  results above are the conclusive proof; the integration runs confirm the fix is
  byte-inert for the common path.
```

## Scope (Phase C increment)

Canonical oscillator math contract, one defect fixed: `spectral_osc_asin` now
clamps its argument to `asin`'s `[-1, 1]` domain (host + CUDA via the shared header,
plus the mirrored Metal MSL string), closing a NaN injection that
`spectral_normalize_phase`'s sub-ULP `[-pi, pi)` overshoot could feed into the
synthesized output for the user-selectable ASIN timbre. The formula-parity version
was bumped 4 -> 5 across all three static-assert guards. The other 7 generators,
`normalize_phase`, the fade envelopes, and the full wavetable / Q15 fixed-point /
peak-estimation / window / FFT-extraction / audio-input cluster audited this pass
were all clean. With this increment the Phase C sweep has cleared fixed-point (161),
analysis/peak-track (162), port/SIMD/out (163), hashing/parsing/path (164),
DSP-math/FFT-scaling + alloc/cache (165), synth-backends + analysis-orchestration
(166), CLI/orchestration (167), embedded fade envelope (168), core synth
dispatch/internal helpers (169), binary-deserialization/converter (170), host
GPU-tile concurrency (171), and the oscillator asin domain guard (172). Phase D
(compiled harness + LUT golden-vector loop) follows.
