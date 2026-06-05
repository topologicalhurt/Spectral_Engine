# Patch notes — Pass 217: route sine into the packed 8×Q15 SIMD kernel (refactor Thread B1)

## Scope

Thread B of the Q-type refactor widens the reach of the packed Q15 islands. **B1** is
its first, smallest slice: re-validate the one timbre that the pack8 kernel had been
*excluding* — sine — and route it in only if the data clears the per-path budget.

This is the deferred follow-up #2 recorded at PASS214/216. Earlier (PASS210/214) sine
was kept on the scalar Q15 path because its waveform eval has **no vector form**: it is
an 8× serial LUT gather (`spectral_lut_sin` per lane), so a naïve "SIMD sine" gains
nothing on the eval itself. The open question B1 settles is whether the *rest* of the
pack8 pipeline — the vectorized uint32 NCO phase (Bv, landed PASS213) plus the float
widen / amp-ramp / accumulate that wraps every pack8 timbre — makes the whole segment
faster than the scalar Q15 sine it would replace, **and** than production float-SIMD,
even with a serial gather in the middle.

## Measurement (the GO gate)

`bench_q15_pack8`, sine, c3==0 (the vec-phase path):

```text
pack8 sine (Bv vec phase) : ~0.74 ns/sample
production float-SIMD     : ~0.93 ns/sample      -> pack8 ~1.26x faster
scalar Q15 sine (--q15)   : slower than both      -> pack8 strictly dominates it
```

Precision is the easy half: sine's pack8 eval is the **same** `spectral_lut_sin` gather
as scalar `spectral_osc_q15_sine`, so the eval is bit-identical. The only SIMD-vs-scalar
delta is the ≤1 LSB rounding of the vectorized phase (the c3==0 vec NCO), the same floor
the four algebraic timbres already live at. The new `q15_simd_parity` sine row measures
**−125.6 dBFS** against a **−84 dBFS** budget — a 41 dB margin.

So sine clears the per-path budget on both axes (throughput and precision) → **GO**, and
per the plan-prescribed auto-route it is wired in.

## What changed

### `core/port/host/oscillator_simd.c`

- **`osc_q15_pack8_eval`** gains a `const q15_t* lut` parameter and a `TIMBRE_SINE` case
  **before** the algebraic clamp: it stores the 8 packed phase indices, gathers
  `spectral_lut_sin` per lane (exact for every `pq`, incl. −32768), reloads, and returns.
  The algebraic clamp/abs/square path is unchanged. Call site passes `sine_lut`.
- **`osc_simd_q15_available`** now returns true for `TIMBRE_SINE` alongside
  saw/square/triangle/parabola. This is the single dispatch gate — with sine in it,
  `timbre_synth_segment` routes sine to `osc_simd_q15_segment` under `--q15` instead of
  the scalar `synth_segment_q15`, and the pipeline log correctly reads "packed 8-wide
  SIMD Q15".
- Header/inline comments updated: the kernel banner no longer says "sine LOSES … NOT
  routed here"; it documents the B1 re-measure and the serial-gather-with-vec-phase shape.

### `tests/core_contracts/test_q15_simd_parity.c`

- Adds `{ TIMBRE_SINE, "sine", −84.0 }` as the first `k_timbres[]` row, so the SIMD-vs-
  scalar Q15 parity lock now covers all five `osc_simd_q15_available()` timbres. Header
  comment updated (four → five). Sine measures −125.6 dBFS.

### `core/oscillator_dispatch.h`, `cmd/cli/spectral_cli.c`

- Doc/comment + `--q15` help text updated to list sine among the packed-SIMD timbres
  (~1.3–1.6× over float-SIMD).

## Verification

```text
- cmake --build build --target q15_simd_parity_test  -> clean
- ctest -R q15_simd_parity                            -> Passed (sine -125.6 dBFS / budget -84)
- full ctest                                          -> 11/11 PASSED
- ./build/bin/spectral_arm64_metal_desktop resources/testing/sin_440hz.wav 0 --q15
    -> "Q15 compute domain: ENABLED for sine (packed 8-wide SIMD Q15; forcing CPU backend)"
       output/out_c.wav written
```

No change to any non-`--q15` path: float stays the desktop default, the scalar Q15 path
is untouched, and the algebraic pack8 timbres are byte-for-byte identical.

## Status

**B1 — LANDED.** Sine is routed into the packed 8×Q15 SIMD kernel; deferred follow-up #2
is closed. The remaining Thread B items are maintainer-gated: **B2** (full-Q15-accumulate
experiment — a precision [DECISION], needs sign-off) and **B3** (the Q-island audit, the
anti-myopia deliverable). Thread C (SIMDe max-width, x86-CI-gated) follows per the A→B→C
order.
