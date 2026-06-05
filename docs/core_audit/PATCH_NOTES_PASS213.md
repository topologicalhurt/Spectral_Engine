# Patch notes — Pass 213: Q5c — packed 8×Q15 SIMD oscillator (the double-lane density win)

## Scope

Q-type domain phase step **Q5c** (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5), the destination
of the whole Q15-on-desktop axis: **double-lane packing.** Q5a/Q5b put an integer-NCO cubic
phase into the production Q15 path, removing the per-sample float→Q15 conversion that PASS210
measured would cancel any lane-density win. Q5c takes the GO and ships the packed kernel: eight
`int16` Q15 lanes evaluated in one 128-bit register per iteration (vs four `float32` in a 128-bit
/ eight in a 256-bit register), then widened to float for the unchanged float amp ramp +
accumulate.

**Measure-first, GO confirmed on data.** A standalone probe (`bench_q15_pack8`, PASS-prior,
EXCLUDE_FROM_ALL) drove the kernel as hard as it goes and isolated the win against a *fair*
same-width lean float baseline before any production code was written. The first untuned pass
read ~0.91× (no win) and I was about to decline; tuning (vectorized amp ramp + 1-op
`cvtepi16_epi32` widen) and a fair 8-wide baseline revealed the real win. **This reverses the
provisional decline — the maintainer's instinct that there was performance to squeeze was right.**

**Default build is byte-identical.** The packed path is reached only when a timbre's opt-in Q15
bit is set (`osc_set_q15_enable`, mask defaults 0, no CLI wires it) AND the timbre's float
dispatch resolves to SIMD. The shipping float default moves no bytes; `osc_parity` and the core
goldens confirm.

## What changed

### `core/port/host/oscillator_simd.c` — new `osc_simd_q15_segment` (host/SIMDe only)

The throughput twin of the scalar `synth_segment_q15`. Per 8-sample sustain block:

```c
simde__m128i wq = osc_q15_pack8_eval(osc_q15_nco_pack8(&nco), timbre); // 8×Q15 in one reg
// widen 8×int16 → 8 float via 1-op sign-extend, scale by 1 LSB:
wf_lo = cvtepi32_ps(cvtepi16_epi32(wq))            * inv_q15;
wf_hi = cvtepi32_ps(cvtepi16_epi32(srli(wq,8)))    * inv_q15;
// amp = amp0 + d_amp*j per lane (same expression as scalar), then fma-accumulate into dst.
```

Phase is the integer NCO stepped 8× per block (`osc_q15_nco_pack8`); the vectorized uint32 NCO
that the probe also tried (`Bv`, extra ~10%) is a **deferred** follow-up — it needs its own
precision re-validation (uint32 lanes keep only 16 fractional bits). Fade-in / sustain-tail /
fade-out stay on the scalar `spectral_osc_q15_*` evaluators (small regions, per-sample envelope),
so those samples are bit-identical to `synth_segment_q15` and only the vectorized sustain body
carries the ≤1-LSB SIMD-eval delta. **Sine is NOT routed here** (its 8× serial LUT gather has no
SIMD form and *loses* to float — 0.88 vs 0.71 ns/sample); it stays on the scalar Q15 path.

The 8-wide eval (`osc_q15_pack8_eval`, wrapped in `// SPECTRAL_Q_DOMAIN` markers — pure fixed
point, enforced by `q_domain_contract`) mirrors the scalar evaluators op-for-op, with two
correctness fixes the throughput probe had glossed over:

- **Triangle** is `subs(subs(MAX,|pq|),|pq|)` — the full-range double-subtract `MAX − 2|pq|`. The
  probe subtracted a *saturated* `2|pq|`, which clamps the entire `|pq|>0.5` half to 0. Fixed.
- **The `pq = −32768` corner** (phase exactly −π): `abs`/`mulhrs` overflow `int16` and flip
  triangle/parabola to +full-scale. `pq` is pre-clamped to `[−32767, 32767]` with one
  `max_epi16`, touching only that single phase value → ≤1 LSB. Fixed.

### `core/oscillator_dispatch.h` / `core/oscillator.c` — dispatch wiring

New host-only interface (`osc_simd_q15_available`, `osc_simd_q15_segment`, guarded by
`OSC_SIMD_GENERIC`). The Q15 dispatch hook in `timbre_synth_segment` now composes the Q15 compute
domain with the existing float scalar/SIMD axis: when the timbre's dispatch resolves to SIMD and
it is one of the four algebraic Q15 timbres, render packed-8; else fall through to scalar
`synth_segment_q15`. `--scalar` is honoured (forces scalar Q15); sine always takes scalar Q15.

### `tests/core_contracts/test_q15_simd_parity.c` (+ cmake target, CMakeLists include)

New CTest `q15_simd_parity`: renders the four algebraic timbres through the real production
dispatch with Q15 enabled, toggling only `OSC_DISPATCH_ALL_SCALAR` vs `ALL_SIMD`, and asserts the
packed kernel matches its scalar Q15 oracle within a per-path dBFS budget. The CI lock on the
kernel — a broken lane op / widen / amp ramp fails the build.

## Verification

```text
- ctest 11/11 PASSED (was 10/10; +q15_simd_parity).
- q15_simd_parity (SIMD vs scalar Q15, budget −84 dBFS):
    saw      0.000e+00  (bit-identical)      square   0.000e+00  (bit-identical)
    triangle 0.000e+00  (bit-identical)      parabola −96.9 dBFS (peak ~1 LSB)
  3 of 4 are bit-identical to scalar Q15; parabola differs ≤1 LSB (mulhrs round-to-nearest
  vs scalar >>15 truncate — the SIMD path is in fact marginally *more* accurate).
- q_domain_contract green: the new // SPECTRAL_Q_DOMAIN block in oscillator_simd.c has no
  float/double token and leaks no raw scale constant (conversion uses the Q15_TO_FLOAT macro).
- q15_production_parity (Q15-vs-float CI lock) now exercises the SIMD path by default and still
  PASSES with full headroom.
- Production builds: desktop, simulate, simulate-daisy rebuild clean. cuda env-blocked (no nvcc).
- Default byte-identity: opt-in path, mask defaults 0, no CLI wires it — osc_parity unchanged.
```

## Throughput (`bench_q15_pack8`, Apple Silicon, ns/sample — lower is faster)

Column **B** (pack8, scalar integer phase) is exactly the shipped `osc_simd_q15_segment`; **A** is
the real production float-SIMD path. **B/A < 1.00 ⇒ pack8 faster.**

```text
  timbre     A prod   B pack8   B/A      speedup
  saw         0.407    0.300    0.74x    1.36×
  square      0.431    0.311    0.72x    1.39×
  triangle    0.432    0.328    0.76x    1.32×
  parabola    0.426    0.356    0.84x    1.20×
  sine        0.714    0.884    1.24x    0.81×  (LOSES → excluded, stays scalar Q15)
```

So **~1.2–1.4× over production float-SIMD** on the four algebraic timbres. Short of the
theoretical 2× — that ceiling needs a full-Q15 accumulate chain and/or native 256-bit-int (16×Q15)
codegen, neither present on this 128-bit-NEON host — but a real, shippable desktop speedup, and
the *first realisation* of the double-lane packing that was the point of Q15-on-desktop. The
`B0` no-phase floor (~0.21 ns/sample, ~0.5× of float) shows the eval+widen+accumulate density
roughly halves the per-sample cost; the remaining gap to 2× is the serial integer phase, which
the deferred vectorized NCO (`Bv`, ~1.4–1.6×) would partly close.

## Status

Q5c closes: the production opt-in Q15 path now has a packed 8×Q15 SIMD kernel that beats float-SIMD
by ~1.2–1.4× on the algebraic timbres, parity-locked to its scalar oracle (3/4 bit-identical),
default byte-identical, host-only (embedded keeps the scalar Q15 path). The double-lane packing
destination of the Q15-on-desktop axis is reached.

**Deferred (not in this slice):** (1) the vectorized uint32 NCO phase (`Bv`, +~10%) — needs its
own precision re-validation before it can ship; (2) the ~8 KB `g_osc_q15_sine_lut`
`#if !SPECTRAL_EMBEDDED` footprint guard — an embedded-footprint concern on the *existing scalar
sine* path, independent of this host-only SIMD kernel (which never touches the LUT). Both are
clean follow-ups, decline-on-data as warranted.
