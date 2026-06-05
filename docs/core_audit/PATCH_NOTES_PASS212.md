# Patch notes — Pass 212: Q5b — integer-NCO phase wired into the production Q15 path

## Scope

Q-type domain phase step **Q5b** (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5). Q5a (PASS211)
built the scalar cubic integer-NCO phase primitive (`core/spectral_phase_nco.h`) and proved,
measure-first, that it tracks the float-cubic phase at/below the Q15-eval floor (in fact
tighter). Q5b takes the GO: it replaces the float cubic phase + per-sample float→Q15
conversion in the opt-in Q15 sustain path with the integer NCO.

**Default build is byte-identical.** The change is confined to `synth_segment_q15`, reached
only when a timbre's opt-in Q15 bit is set (`osc_set_q15_enable`); the mask defaults to 0 and
no CLI wires it, so the shipping float default moves no bytes.

## What changed

### `core/oscillator.c` — `synth_segment_q15` now phases by integer NCO

Before, each of the three fade regions recomputed the phase per sample in float:

```c
float p    = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j);
float rads = phase_to_rads(p);
q15_t pq   = spectral_osc_q15_phase_from_rads(rads);   // float Horner + wrap + convert
```

Now the phase index comes straight from the integer cubic forward-difference accumulator —
3 integer adds per sample, zero float on the phase path:

```c
SpectralPhaseNco nco;
spectral_phase_nco_init(&nco, phase0, alpha, c2, c3);   // once per segment (float boundary)
...
q15_t pq = spectral_phase_nco_step(&nco);               // per sample: round top 16 bits, advance
```

The waveform eval (`osc_q15_eval` → `spectral_osc_q15_*`), the float amplitude/fade envelope,
and the accumulation are unchanged. `#include "spectral_phase_nco.h"` added.

**Step-order invariant (documented at the call site):** the NCO's forward-difference chain is
stateful, so it must be stepped exactly once per sample in sample order. The three fade loops
(fade-in / sustain / fade-out) together cover `[0, len)` contiguously and in order in *every*
case — normal, overlapping fades on a short segment, or `len` shorter than a fade region — so
stepping once per loop-body iteration reproduces the exact per-sample phase the float path
recomputed from `j`.

## Verification

```text
- ctest 10/10 PASSED. q_domain_contract green (the markers live in the header, which already
  passes; oscillator.c gains no markers). phase_nco_precision unchanged.
- q15_production_parity (the CI per-path dBFS lock) now exercises the NCO path and PASSES with
  room to spare (budgets in brackets):
    sine     -90.1 dBFS  [-72]      saw      -104.2 dBFS [-78]
    square   -96.4 dBFS  [-78]      triangle  -92.1 dBFS [-80]
    parabola -96.7 dBFS  [-78]
  vs the PASS209 float-cubic Q15 floor (sine -88.2, saw -98.1, square -96.4, triangle -98.0,
  parabola -98.8): square is bit-identical, sine/saw improved, triangle/parabola shifted a few
  dB (the NCO's round-to-nearest index vs the old truncate-toward-zero) — all far inside budget.
- Production builds: desktop, simulate, simulate-daisy rebuild clean. cuda env-blocked (no nvcc
  on this host) — unaffected.
- Default byte-identity: opt-in path, mask defaults 0, no CLI wires it — default render unchanged.
```

## Throughput re-measure (`bench_q15_throughput`, Apple Silicon, ns/sample)

The `q15-scl` column is now the integer-NCO path. Removing the per-sample float→Q15 conversion
closed the scalar gap (vs PASS210's float-phase Q15 scalar):

```text
  timbre     q15/float-scalar (PASS210 -> now)    q15/float-simd (PASS210 -> now)
  sine            0.51x -> 0.51x                      3.32x -> 2.27x
  saw             1.26x -> 0.95x                      4.69x -> 3.51x
  square          1.17x -> 0.88x                      4.47x -> 3.30x
  triangle        1.28x -> 0.97x                      4.73x -> 3.61x
  parabola        1.30x -> 0.95x                      4.88x -> 3.52x
```

The four algebraic timbres went from *losing* to scalar float (1.17–1.30x slower — the old
`fcvtzs`+`scvtf` round-trip across the float/int register file) to *tying or slightly beating*
it (0.88–0.97x). Sine stays 0.51x (LUT vs deg-9 poly, unchanged — its phase was never the cost).

The Q15 scalar path still trails 4-wide float-SIMD (2.27–3.61x), as expected: a scalar path
cannot beat a 4-wide one on throughput. That gap is exactly Q5c's job — **8-wide packed Q15**,
which the integer phase now makes feasible (8 phase indices from 8 integer adds, no per-lane
float→Q conversion to cancel the lane-density win). PASS210's decline was conditioned on
"float phase"; Q5b removes that condition.

## Status

Q5b closes: the production opt-in Q15 path phases by the integer NCO, parity-locked and within
budget, default byte-identical. The integer-phase precondition for double-lane packing is now
in the production path, not just a header. Next is **Q5c**: the width-templated packed 8×Q15
SIMD kernel over integer phases (the density win) + a SIMD-vs-scalar parity CTest (mirrors
`osc_width_parity`), **proven on `bench_q15_throughput` to beat float-SIMD before it ships** —
decline-on-data if the cubic 3×64-bit forward-difference erases the 8-wide density, exactly as
PASS210. The ~8 KB `g_osc_q15_sine_lut` `#if !SPECTRAL_EMBEDDED` footprint guard folds in there.
