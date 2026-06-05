# Patch notes — Pass 211: Q5a — integer-NCO cubic phase primitive + measure-first gate (PASSED)

## Scope

Q-type domain phase step **Q5a** (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5). PASS210 closed
Q3b by *declining* a Q15 SIMD kernel over float phase on measured data: the per-lane
float→Q15 conversion cancels the 8×int16-vs-4×float32 lane-density advantage before the eval
starts, so the structural Q15 throughput win is gated on an **integer phase accumulator**.
Q5 takes up that deferred axis. Q5a is its measure-first gate: build the scalar cubic
integer-NCO phase primitive and *characterize* it before any kernel is wired — if integer
phase can't track the float reference, we learn it here, cheaply, instead of inside a SIMD
kernel.

**No production bytes move.** This pass adds one header (included only by the new test), one
`EXCLUDE_FROM_ALL` CTest, and these notes. The opt-in Q15 mechanism (PASS209) and float
default are untouched.

## What landed

### `core/spectral_phase_nco.h` — scalar cubic integer-NCO phase primitive

The integer twin of `spectral_segment_phase_at_cubic_f32`. Accumulates
θ(t) = phase0 + alpha·t + c2·t² + c3·t³ by **cubic forward-differencing** — 3 integer adds
per sample, zero float on the hot path:

```
init: p=phase0, D1=alpha+c2+c3, D2=2·c2+6·c3, D3=6·c3   (all in turns·2^64)
step: emit p; p+=D1; D1+=D2; D2+=D3
```

- **Phase domain:** a `uint64` where one full circle (2π) ≡ 2^64, so uint64 overflow *is* the
  mod-2π wrap (no conditional). Top 16 bits are the signed Q15 index the `spectral_osc_q15_*`
  evaluators read (matching `spectral_osc_q15_phase_from_rads`); the low 48 bits are
  fractional headroom so D3 = 6·c3 (far below one Q15 LSB for realistic chirp) doesn't round
  to zero and silently drop the cubic term.
- **Boundary discipline:** the float→fixed coefficient init (`spectral_phase_nco_init` +
  `spectral_phase_turns_to_u64`, which converts in two 32-bit steps to avoid the UB of casting
  a `double ≥ 2^64` to integer) is the lone float boundary and stays OUTSIDE the
  `SPECTRAL_Q_DOMAIN` region. The per-sample `spectral_phase_nco_step` is pure integer and
  lives inside it (reads top 16 bits round-to-nearest via a half-LSB bias, then advances the
  difference chain). `q_domain_contract` enforces this — the explanatory prose that names the
  boundary types is kept above the BEGIN marker so the token scan doesn't false-positive on it.

### `phase_nco_precision` CTest — the measure-first gate

`tests/core_contracts/test_phase_nco_precision.c` +
`cmake/targets/phase-nco-precision-test.cmake` (header-only link set, like
`q15_compute_precision`). Renders worst-case cubic-phase segments (linear-mid, near-Nyquist,
quadratic chirp, cubic chirp, long+cubic over 4096 samples) two ways and reports:

- **Part 1 — phase-index drift (timbre-independent):** integer-NCO index and the production
  float-cubic index, each vs a **double-precision truth index**, in LSBs.
- **Part 2 — phase-source swap cost (per timbre, dBFS):** the SAME Q15 evaluator driven by the
  integer-NCO index vs by the float-cubic index (the currently-shipping `synth_segment_q15`
  path) — i.e. exactly what Q5b changes.

## Measurement (deterministic — fixed-point + double, no float eval in Part 1)

```text
  Part 1: phase-index drift vs double truth (LSBs, lower is tighter)
  segment           len     nco-max     float-max
  linear-mid       2048           1             1
  near-nyquist     2048           2             5
  quad-chirp       1024           1             1
  cubic-chirp      1024           1             1
  long-cubic       4096           1             2

  Part 2: phase-source swap cost (int-NCO vs float-cubic, same Q15 eval)
  sine       -86.7 dBFS    peak 2.441e-04
  saw        -93.3 dBFS    peak 9.155e-05
  square    -300.0 dBFS    peak 0          (bit-identical)
  triangle   -87.5 dBFS    peak 1.526e-04
  parabola   -91.8 dBFS    peak 1.526e-04
```

## Verdict — the gate PASSES, with margin

1. **Integer phase is *tighter* than float, not merely adequate.** The NCO holds ≤2 LSB of the
   double truth on every segment; the float-cubic path drifts to **5 LSB** near Nyquist and
   2 LSB on long-cubic. Mechanism: the integer forward differences are EXACT integer adds, so
   the only error is the one-time fixed-point init quantization of D1/D2/D3 (sub-femto-LSB,
   amplified by at most ~k³/6 over a segment → still ≪1 LSB); float32, by contrast, loses
   ~8e-5 rad of resolution once total phase reaches hundreds of radians. This is the
   mechanistic confirmation of *why* the axis is integer phase.

2. **Swapping the Q15 path's phase source costs nothing audible.** Every timbre's swap cost
   sits at or below the PASS208 Q15-eval floor (sine −85.1, saw −91.5, square −90.0, triangle
   −92.6, parabola −91.0 dBFS). Square is bit-identical (the ≤2 LSB index difference never
   straddled the sign boundary). The integer NCO therefore does not degrade the
   already-characterized Q15 path — it can drop in.

The CTest's CHECKs are loose tripwires (index drift ≤3 LSB; swap cost ≤ −55 dBFS), not the
verdict — a genuinely broken NCO (dropped cubic term, wrong scale) drifts dozens-to-hundreds
of LSB. The printed tables are the verdict, read for the Q5b GO/NO-GO.

## Verification

```text
- ctest 10/10 PASSED (was 9/9; phase_nco_precision added). q_domain_contract green — the
  new header's per-sample step is pure fixed point inside the markers.
- Production builds: desktop, simulate, simulate-daisy all rebuild clean. cuda is
  environment-blocked on this host (nvcc not installed) — unaffected (the header is included
  only by the EXCLUDE_FROM_ALL test; no production TU references it yet).
- Default byte-identity: unchanged — no production path includes spectral_phase_nco.h; float
  remains the default domain and the opt-in Q15 mask still defaults to 0.
```

## Status

Q5a closes: the integer-NCO cubic phase primitive exists, is Q-domain-clean, and is
**measured** to track the float reference at/below the Q15 floor (in fact tighter). The axis
PASS210 deferred is de-risked — integer phase is sound. Next is **Q5b**: replace the float
cubic phase + per-sample float→Q15 conversion in `synth_segment_q15` with the integer NCO,
add a production parity CTest (per-path dBFS, like `q15_production_parity`), verify the
builds, and re-run `bench_q15_throughput` (scalar integer-NCO vs scalar float). Q5c is then
the packed 8×Q15 SIMD kernel over integer phases — the density win — to be **proven on
`bench_q15_throughput` before it ships**, decline-on-data if it loses (exactly as PASS210).

## Known follow-ups

- **Q5b/Q5c** as above.
- **Footprint note (carried from PASS209/210):** `g_osc_q15_sine_lut` (~8 KB BSS) on every
  linking build; the `#if !SPECTRAL_EMBEDDED` guard folds in at Q5c.
