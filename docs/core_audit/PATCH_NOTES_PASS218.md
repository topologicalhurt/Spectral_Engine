# Patch notes — Pass 218: B2 full-Q15 amp-scale experiment — DECLINED on data (refactor Thread B2)

## Scope

Thread B2 asked whether keeping the **amp ramp + scale in Q15** — instead of widening
the packed waveform to float and doing `amp * wave + accumulate` in float — closes the
gap from the shipped ~1.4–1.6× (PASS214) toward the theoretical 2× the lane density
promises. This pass **measures it and declines**, per the measure-first / decline-on-data
discipline (same shape as PASS210, which declined the Q15-over-float-phase kernel).

The shipping kernel (`osc_simd_q15_segment`, `bench_pack8_f`) does, per 8 samples:
eval 8×Q15 → **widen to float** (`cvtepi16_epi32` ×2 + `cvtepi32_ps` ×2 + `mul` ×2) →
float amp ramp → `dst[j] += amp * wave` (2 float muls + a float amp-carry).

The B2 variant (`bench_pack8_qamp`) folds the amplitude into **one 8-wide `mulhrs`** in
Q15 *before* the widen, leaving only the unavoidable float `load/add/store` accumulate
(the output buffer is float — it is the cross-segment additive sum, ultimately a float
WAV). The amp ramp becomes an int16 carry (`ampq += astep`).

## Measurement (the decline gate)

Added to the in-tree `bench_q15_pack8` probe (EXCLUDE_FROM_ALL; timing is machine-
dependent so it is **not** a ctest). Both variants use the vectorized c3==0 NCO phase,
so `Bq/Bv` is apples-to-apples. `qamp_precision()` renders one segment each way and
reports the amp-quantization error vs the float amp. Apple Silicon, 2048-sample segment:

```text
== B2: Q15-amp (mulhrs) vs float-amp pack8 [both vec phase] ==
  timbre     Bv float    Bq Q15      Bq/Bv   amp maxErr amp rmsErr
  sine          0.752     0.726      0.97x     -49.5 dB   -57.2 dB
  saw           0.265     0.273      1.03x     -49.6 dB   -58.9 dB
  square        0.276     0.300      1.09x     -49.5 dB   -54.3 dB
  triangle      0.281     0.300      1.07x     -50.0 dB   -62.3 dB
  parabola      0.305     0.316      1.03x     -49.9 dB   -57.1 dB
```

(`Bq/Bv < 1.00x` = Q15-amp faster. Stable across re-runs; the algebraic direction is
consistent, not noise.)

## Verdict: DECLINE — both decline criteria from the plan are met

1. **The float widen is already near the ceiling → no throughput win.** The four
   algebraic timbres get *slower* (Bq/Bv 1.03–1.10×); only sine ties within noise
   (0.97×). The widen Q15→float is **structurally unavoidable** (the `dst` accumulator
   and the WAV are float), so it stays in both kernels. The algebraic eval is so cheap
   that the extra serial `mulhrs` + int16 amp-carry, sitting on the critical path
   *before* the widen, costs more than it removes — the float amp-ramp it replaces
   overlapped the eval better. The 2× the lane density promises is gated on removing the
   widen entirely (a full-int accumulate or native 256-bit int — Thread C, not B2), not
   on moving the amp multiply.

2. **Precision regresses ~35 dB.** The int16 amp ramp quantizes `d_amp` to a whole Q15
   LSB per block: for `d_amp = -2e-4`, `astep = round(8·d_amp·32768) = round(-52.4) = -52`,
   so the ramp drifts ~0.4 LSB/block, ~100 LSB over the segment ≈ **−49 dB** max error.
   That is ~35 dB above the Q15-eval floor (~−84 to −90 dB) and would **blow the
   `q15_simd_parity` −84 dBFS budget** if shipped. A higher-precision amp accumulator
   (Q31 narrowed per block) would fix the drift but re-adds the per-block conversion
   ops, erasing the (already negative) throughput case. There is also an unhandled
   **amp > 1.0 overflow** risk: a single segment with gain > 1 is not Q15-representable,
   whereas the float amp carries it natively.

The deeper "full-Q15 accumulate" framing (PASS213's "needs full-Q15 accumulate and/or
native 256-bit-int") is also structurally blocked for *cross-segment* accumulation:
additive synthesis routinely sums many partials past ±1.0 before normalization, which a
Q15 accumulator saturates. So the float accumulator is not incidental — it is required
for additive headroom. The 256-bit-int density axis (16×Q15@256 on AVX2) remains open as
**Thread C**, where the win comes from doubling lane count, not from removing the widen.

## What changed

- **`tests/core_contracts/bench_q15_pack8.c`** — added `bench_pack8_qamp` (the Q15-amp
  kernel), `qamp_init8` (the int16 ramp seed), and `qamp_precision` (amp-quantization
  dBFS), plus a B2 results block in `main`. **No production source touched.** The bench
  stays in-tree as the reproducible evidence for the decline (the PASS210 pattern).

## Verification

```text
- cmake --build build --target bench_q15_pack8  -> clean
- build/bin/bench_q15_pack8                      -> table above (stable across re-runs)
- No production kernel / test / default-path change -> nothing to re-baseline.
```

## Status

**B2 — DECLINED on data.** The shipping pack8 kernel (float widen + float amp +
accumulate) is the right design on a 128-bit-NEON host; folding the amp into Q15 loses on
both throughput and precision. The remaining Thread B item is **B3** (the island audit —
anti-myopia deliverable). The density axis the 2× really needs lives in **Thread C**
(SIMD max-width, 16×Q15@256 on AVX2, x86-CI-gated). Float stays the desktop default; the
opt-in `--q15` path is unchanged.
