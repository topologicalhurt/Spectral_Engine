# Patch notes — Pass 210: Q3b slice 2 — measure-first, SIMD Q15 kernel declined on data

## Scope

Q-type domain phase step **Q3b**, slice 2 (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5).
Slice 1 (PASS209) landed the opt-in per-path Q15 flag + scalar Q15 oracle + production
parity lock. Slice 2 was scoped as **#73 (SIMD Q15 sustain kernel)** + **#75 (perf
measurement)**. This pass does #75 first — the measurement — and that measurement
**declines #73**: no Q15 SIMD kernel over the current *float* phase accumulator can beat the
shipping float-SIMD path. The throughput win is structurally gated on the deferred
integer-NCO axis, so building the kernel now would add pack/unpack machinery for a
guaranteed regression. Measure-first, then *don't* build.

**No production bytes move.** This pass adds one `EXCLUDE_FROM_ALL` manual benchmark and its
patch notes. The opt-in Q15 mechanism and the CI parity lock from PASS209 are unchanged;
float remains the default domain.

## What landed

### `bench_q15_throughput` — manual perf probe (not a ctest)

`tests/core_contracts/bench_q15_throughput.c` + `cmake/targets/q15-throughput-bench.cmake`
(`EXCLUDE_FROM_ALL`, same engine link set as `q15_production_parity_test`). Timing is
machine-dependent, so this is a standalone executable run by hand, not a pass/fail ctest. It
renders one sustain-dominated 2048-sample segment through the **real** production
`timbre_synth_segment()` under three configs and prints ns/sample:

- **float-scalar** — `osc_set_dispatch(ALL_SCALAR)`, Q15 off (apples-to-apples scalar ref)
- **q15-scalar** — `osc_set_dispatch(ALL_SCALAR)`, Q15 on (the PASS209 scalar oracle)
- **float-simd** — `osc_set_dispatch(ALL_SIMD)`, Q15 off (the shipping default baseline)

The Q15 path is scalar regardless of dispatch (dispatch steers only the float path), so
float-scalar vs q15-scalar is the honest equal-vectorization compare; float-simd is the
baseline a Q15 SIMD kernel would have to beat.

## Measurement (Apple Silicon, Release, ns/sample — lower is faster)

```text
  timbre       float-scl      q15-scl   float-simd     q15/fscl  q15/fsimd
  sine             4.425        2.279        0.686        0.51x      3.32x
  saw              1.439        1.815        0.387        1.26x      4.69x
  square           1.568        1.836        0.411        1.17x      4.47x
  triangle         1.515        1.941        0.411        1.28x      4.73x
  parabola         1.532        1.998        0.409        1.30x      4.88x
```

- **vs scalar float:** Q15 wins *only* for sine (0.51x — the LUT beats the deg-9 poly). The
  four algebraic timbres LOSE (1.17–1.30x): the per-sample float→Q15→float round-trip
  (`fcvtzs` down + `scvtf` up, both crossing the float/int register file) costs more than the
  cheap float eval (saw=negate, square=sign, triangle=shift, parabola=one mul).
- **vs the shipping float-SIMD default:** the scalar Q15 oracle is **3.3–4.9x slower** across
  the board.

## Why #73 (the SIMD Q15 kernel) is declined — three independent reasons

1. **The eval is not the bottleneck.** float-SIMD evaluates all five waveforms 4-wide in
   float with zero conversions; the floor cost is phase + accumulate, not the eval. A Q15
   SIMD kernel over float phase adds convert-down + pack + unpack + convert-up *around* the
   one step (eval) that was already nearly free. It strictly adds work vs float-SIMD.

2. **The only structural Q15 win is gated on integer phase.** A 128-bit register holds
   8×int16 vs 4×float32 — a 2× lane-density advantage. But filling 8 Q15 phase lanes from a
   *float* accumulator means computing 8 float phases first (the expensive part, already
   paid) and then converting down — the density advantage is cancelled before the eval
   starts. Realizing it requires an **integer phase accumulator** (8-wide uint add, no
   float, no conversion), which is exactly the integer-NCO axis the QTYPE plan deferred.
   Float-phase Q15 is the worst of both: float's conversion cost without integer's density
   payoff.

3. **Sine — the lone scalar winner — loses hardest vectorized.** Its Q15 form is a LUT
   lookup; 8 independent indices need an 8-way gather, which NEON has no efficient form of
   (it serializes into scalar loads + lane inserts). float-SIMD sine is a pure-register
   polynomial (0.686 ns/sample). So the one timbre that wins scalar regresses most in SIMD.

The win is not small — it is *absent* until phase is integer. #73 over float phase ships a
guaranteed regression in exchange for pack/unpack complexity. Declined; re-scoped onto the
integer-NCO axis.

## Verification

```text
- ctest 9/9 PASSED (unchanged; the bench is EXCLUDE_FROM_ALL and registers no test).
- Production builds: desktop, simulate, simulate-daisy all rebuild clean. cuda is
  environment-blocked on this host (nvcc not installed) — unaffected (host-only change plus
  an EXCLUDE_FROM_ALL target).
- Default byte-identity: unchanged from PASS209 — opt-in mask still defaults to 0, no CLI
  wires it, and this pass adds only a manual benchmark.
```

## Status

Q3b closes. Slice 1 (PASS209): the opt-in Q15 compute domain + scalar oracle + per-path dBFS
parity lock are in and green. Slice 2 (this pass): the SIMD Q15 kernel (#73) is **declined on
measured data** — it cannot beat float-SIMD over float phase, and the throughput win is
gated on the deferred integer-NCO axis. The benchmark stays in-tree as the reproducible
evidence and as the harness that re-measures if/when the integer-NCO axis is taken up. Float
remains the default domain.

## Known follow-ups

- **Integer-NCO phase axis (deferred, QTYPE plan):** the prerequisite for any real Q15
  throughput win — integer phase accumulation feeding 8-wide Q15 packing with no float↔Q
  conversions. If/when scoped, `bench_q15_throughput` is the measurement gate.
- **Footprint note (carried from PASS209):** `g_osc_q15_sine_lut` (~8 KB BSS) lives in
  `oscillator.c` on every linking build; a `#if !SPECTRAL_EMBEDDED` guard on the desktop Q15
  additions remains a clean refinement (embedded has its own Q15 LUT and does not use this
  path).
