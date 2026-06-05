# Patch notes — Pass 209: Q3b slice 1 — opt-in Q15 compute plumbing + production parity lock

## Scope

Q-type domain phase step **Q3b**, slice 1 (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5).
The maintainer signed off all five candidate timbres (sine/saw/square/triangle/parabola)
for the opt-in Q15 *compute* production path (2026-06-04). Q3b is the byte-moving step,
so it is built in slices that follow the standing task decomposition: this slice lands the
**per-path opt-in flag + dispatch hook (#72)** and the **production parity CTest (#74)** —
the SIMD `__smlad`/`__qadd16` packing (#73) and perf measurement (#75) are the next slice.

**Float stays the default domain. This pass moves no default bytes** — the Q15 path is
reached only when a caller explicitly sets a timbre's enable bit, which nothing in the
production CLI does yet. The added hot-path test reads a global that defaults to 0, so the
float FP op sequence is unchanged (control-flow branch only, never an FP op).

## What landed

### `spectral_osc_q15.h` — shared production Q15 evaluators (single source)

New `core/spectral_osc_q15.h`: the opt-in Q15 *compute* twin of the float L0 formulas, the
single source consumed by BOTH production (`oscillator.c`) and the CTests — so what we
*measure* is what we *ship*. Five evaluators (sine via interpolated full-scale LUT, saw via
saturating negate, square via sign, triangle via one shift, parabola via one
`spectral_mul_q15`), bracketed in `// SPECTRAL_Q_DOMAIN BEGIN/END` so `q_domain_contract`
enforces no float leak. Two boundary helpers stay OUTSIDE the region: the float→Q phase map
(`spectral_osc_q15_phase_from_rads`, rads/π→[-1,1) centered so phase 0→0 — deliberately not
the `PHASE_RAD_TO_Q15` macro, whose [0,2π) -0.5-centered convention differs) and a
gain-matched **full-scale** (peak `Q15_MAX`) sine-LUT init (not production's 32700-headroom
table, whose ~-0.02 dB interp-overflow gain would read as an amplitude error vs float).

**Scope of the Q15 path:** the *waveform* is evaluated in Q15; phase accumulation stays in
float (the cubic NCO). Integer-NCO phase resolution is an orthogonal, deferred axis
(QTYPE plan), so the precision carried is exactly the Q3a waveform-eval floor.

### Per-path opt-in flag + dispatch hook (`oscillator.h` / `oscillator.c`)

- `osc_set_q15_enable(uint16_t mask)` / `osc_get_q15_enable()` + `OSC_Q15_BIT(timbre)` +
  `osc_q15_available(timbre)` (true only for the 5 signed-off timbres). The enable mask is a
  per-timbre bitmask **orthogonal** to the float scalar/SIMD `OscDispatchWord` — Q15 is a
  *domain* choice, not an execution strategy — and defaults to 0.
- The sine LUT is built once inside `osc_set_q15_enable` (a single-threaded setup call, like
  `osc_set_dispatch`), so the per-segment OMP render loop only ever reads it — no lazy-init
  race on the hot path.
- `synth_segment_q15()`: a scalar Q15 sustain renderer, op-for-op the float
  `synth_segment_scalar` except the waveform is evaluated in Q15 (float cubic phase → Q15
  phase → Q15 eval → back to float for the unchanged float amp + fade envelope +
  accumulation). This is the **correctness/precision oracle** the SIMD Q15 kernel (#73) will
  be measured against — the same scalar-reference-then-SIMD pattern the float path used.
- Hook in `timbre_synth_segment`, placed after the band-limited early-return and before the
  float dispatch: if the timbre's Q15 bit is set and it has a Q15 path, render via Q15 and
  return; otherwise fall through to the unchanged float dispatch.

### `q15_compute_precision` (Q3a) refactored onto the shared header

The Q3a harness now `#include`s `spectral_osc_q15.h` and calls the production evaluators
(thin forwarders adapt the sine LUT-capture to its uniform function table) instead of its
own local copies. The duplicated formula math + local LUT/phase helpers are gone — the math
exists once. **Proof it preserved everything:** the refactored test reports byte-identical
PASS208 numbers (sine -85.1, saw -91.5, square -90.0, triangle -92.6, parabola -91.0 dBFS).

### `q15_production_parity` CTest (test #9)

`tests/core_contracts/test_q15_production_parity.c` + `cmake/targets/q15-production-parity-test.cmake`
(wired in `CMakeLists.txt`; same engine link set as `osc_parity`). Renders representative
segments (low/high/chirped/odd-length, with real fade + amp ramp) through the production
`timbre_synth_segment` twice — Q15 off (shipping float default) vs Q15 on — and asserts each
timbre's RMS error stays under a per-path budget pinned ~12-13 dB above the Q3a floor. This
is the CI lock on the §7 per-path sign-off: float stays default, a Q15 regression fails the
build.

**Measured (production path, with envelopes):**

```text
  sine      RMS err = 3.903e-05 (-88.2 dBFS, -82.1 dB vs sig)  [budget -72.0]
  saw       RMS err = 1.244e-05 (-98.1 dBFS, -90.4 dB vs sig)  [budget -78.0]
  square    RMS err = 1.515e-05 (-96.4 dBFS, -93.4 dB vs sig)  [budget -78.0]
  triangle  RMS err = 1.252e-05 (-98.0 dBFS, -90.3 dB vs sig)  [budget -80.0]
  parabola  RMS err = 1.145e-05 (-98.8 dBFS, -93.0 dB vs sig)  [budget -78.0]
```

The production numbers land BELOW the Q3a synthetic floor — the fade envelopes scale both
paths down in the ramp regions, dropping aggregate RMS error. The budget is pinned to the
floor (not the lucky segment shape) so it stays a stable cross-platform bound.

## Verification

```text
- ctest 9/9 PASSED (new q15_production_parity #9; q_domain_contract green confirms the new
  spectral_osc_q15.h Q-region is marker-balanced and float-free; q15_compute_precision green
  with byte-identical PASS208 numbers after the shared-header refactor).
- Production builds: desktop, simulate, simulate-daisy all rebuild clean. cuda is
  environment-blocked on this host (nvcc not installed) — unaffected by this change, which
  touches only host code in oscillator.c.
- Default byte-identity: by construction — opt-in mask defaults to 0, no CLI wires it, the
  added hot-path branch is control-flow only and never alters the float FP sequence.
```

## Known follow-ups (next slice)

- **#73 — SIMD Q15 sustain kernel** for the 5 timbres (`__smlad`/`__qadd16` packing, keyed
  off `SIMDE_NATURAL_INT_VECTOR_SIZE`): the actual throughput win. The scalar `synth_segment_q15`
  here is its oracle; a Q15 SIMD-vs-scalar parity CTest mirrors `osc_width_parity`.
- **#75 — perf measurement** (Q15 vs float speedup) + final re-verify.
- **Footprint note:** `g_osc_q15_sine_lut` (~8 KB BSS) currently lives in `oscillator.c` on
  every build that links it. Embedded already has its own Q15 LUT and does not use this
  desktop path, so a `#if !SPECTRAL_EMBEDDED` guard on the desktop Q15 additions is a clean
  refinement to fold into the SIMD slice.

## Status

Q3b slice 1 closes: the opt-in mechanism + scalar Q15 oracle are in, and the per-path dBFS
budget is locked in CI. Float remains the default domain. Next: the SIMD Q15 kernel (#73).
