# Patch notes — Pass 208: Q3a — Q15-compute precision characterization (measure-first)

## Scope

Q-type domain phase step **Q3a** (`docs/core_audit/QTYPE_DOMAIN_PLAN.md` §5): the
measure-first input to Q3. Q3 adds an *opt-in* Q15 **compute** domain for throughput-bound
L1 oscillator paths; computing audio in Q15 is lossy (it spends ~60 dB of the float
oscillator's -155 dBFS headroom), so the plan (§6) leaves open *which* timbres clear the
15-bit bar, and §7 gates every byte-moving promotion on a measured dBFS justification +
maintainer sign-off. This pass produces that measurement. **It changes no production
output** — it adds one golden-neutral CTest and touches no production source.

## What landed

### `q15_compute_precision` CTest (test #8)

`tests/core_contracts/test_q15_compute_precision.c` +
`cmake/targets/q15-compute-precision-test.cmake` (wired in `CMakeLists.txt`). The harness
renders the five algebraically-clean / LUT L1 waveforms two ways and reports per-timbre RMS
error in dBFS:

- **float reference** — the production L0 formula `spectral_osc_<t>(rads, w)`.
- **Q15 compute** — a pure fixed-point evaluator (`q15_wave_<t>`) bracketed in
  `// SPECTRAL_Q_DOMAIN BEGIN/END` markers (so the `q_domain_contract` test enforces no
  float leaks into the Q region): sine via the interpolated Q15 LUT, saw via a saturating
  negate of the Q15 phase, square via its sign, triangle/parabola via one Q15 shift / one
  `spectral_mul_q15`.

**Scoping (deliberate):** both paths are driven from the *same* phase (float `rads` mapped
once to a Q15 phase at a single float→Q boundary), so the number isolates Q15
**waveform-evaluation** precision. Integer-NCO phase quantization is an orthogonal axis (a
frequency-resolution question) and is out of scope here — a real Q15 production oscillator
carries both, but the per-path GO/NO-GO bar Q3 asks about is dominated by waveform-eval
precision. The sine reference uses a *local full-scale* (Q15_MAX) sine LUT rather than the
production `spectral_lut_init_sine` table, whose deliberate 32700 interp-overflow headroom
is a ~-0.02 dB intentional gain (~-54 dBFS as an error) that would otherwise mask the
quantization floor we are trying to read.

## Measurement (amp 0.8, fs 48 kHz, 4 partials × 8192 samples)

```text
  sine       RMS err = 5.557e-05 (-85.1 dBFS)   peak err = 1.501e-04
  saw        RMS err = 2.656e-05 (-91.5 dBFS)   peak err = 6.101e-05
  square     RMS err = 3.146e-05 (-90.0 dBFS)   peak err = 4.274e-05
  triangle   RMS err = 2.346e-05 (-92.6 dBFS)   peak err = 6.101e-05
  parabola   RMS err = 2.823e-05 (-91.0 dBFS)   peak err = 6.104e-05
```

All five sit at the Q15 quantization floor (~-90 dBFS); sine is ~7 dB higher purely from the
12-bit-LUT + 8-bit-interp residual, as expected. **Verdict: every one clears a generous
throughput-bound bar** — the whole algebraic + LUT-sine L1 set is precision-viable for Q15
*compute*. (quantized/pwm/asin are width-/transcendental-shaped and are not Q15-compute
candidates.) The CTest's `-60 dBFS` CHECK is a gross-bug sanity tripwire, not the verdict;
the table is the verdict.

## Verification

```text
- ctest 8/8 PASSED (… osc_width_parity, q_domain_contract, q15_compute_precision).
  q_domain_contract green confirms the new Q-domain region in the test is marker-balanced
  and float-free.
- 5 production targets rebuild clean (no production source touched — golden-neutral).
```

## Status

Q3a closes: the measurement is in CI and the data says all five candidate paths are
precision-viable in Q15. **Q3b — the production opt-in Q15 path (per-path flag + SIMD
`__smlad`/`__qadd16` packing, keyed off `SIMDE_NATURAL_INT_VECTOR_SIZE`) — is the
byte-moving step and STOPS for maintainer per-path sign-off** (§7), since it is lossy by
design and moves observable output when enabled. Float stays the default domain.
