# Patch notes — Pass 197: Optimisation track O4-A — REJECTED-as-specified (the `df_q15` chirp source quantises to 0 for every realisable input)

## Scope

Fourth pass of the **optimisation track** (`docs/core_audit/OPTIMISATION_PLAN.md`),
evaluating **O4-A** (Tier 4): "ARM stores `df_q15` but rejects chirped segments at load
(`spectral_synth_arm32.c:134-140,419`). Accumulate `freq_inc += freq_delta/sample` (fixed-
point analogue of O3-A); drop the rejection. ENABLER for F1 on embedded (linked segments are
chirped). Flag `SPECTRAL_HAS_CHIRP`. Verify via the chirp-extended `arm32_process_correctness`
oracle. Risk: medium."

**Outcome: NOT VIABLE AS SPECIFIED — rejected, no code change.** The plan's premise is that
`df_q15` is a usable chirp source the ARM hot path can begin consuming. It is not: the slope
the analysis stage actually produces is 3–4 orders of magnitude finer than the plain-`Q15`
LSB, and the two conversion paths that populate `df_q15` *both* drive it to exactly 0 for
every physically realisable partial. Consuming it would therefore be a guaranteed no-op while
adding a per-sample cost to the CTF-audited inner loop and reinterpreting the persisted `.spq`
wire format — a net pessimisation for zero output change. This mirrors the **O3-B** rejection
precedent (plan lines 312-319): measured, not assumed; the residual is hollow.

## Why `df_q15` carries no chirp — three independent, airtight reasons

### 1. The runtime in-memory path forces `df_q15 = 0` unconditionally (config-independent)

`segment_to_q15()` (`synth/backends/sim/spectral_synth_simulation.c:114-150`) is the conversion
the simulator/emulator actually exercises every run (the `.spq` is regenerated in-memory, not
persisted). It zero-inits `*dst` and never writes `df_q15`:

```c
*dst = (SpectralSegmentQ15){0};
...
/* df_q15 stays 0 (zero-initialized above): chirp is intentionally dropped. */
```

So on the path the ARM oracle and the desktop emulator run, **there is no nonzero `df_q15`
to consume** — independent of FFT size, hop, or peak-matching window. `freq_inc += freq_delta`
with `freq_delta` derived from `df_q15 == 0` is identically the constant-frequency loop.

### 2. The compact segment variant has no `df_q15` field at all

Under `SPECTRAL_Q15_COMPACT` the 14-byte `SpectralSegmentQ15` (`synth/math/spectral_q15.h:
170-179`) omits `df_q15` entirely — chirp is structurally absent from the wire format. Only the
16-byte non-compact variant even has the slot.

### 3. The persisted path's `Q15` scale quantises every realisable slope to 0

The analysis stage emits the quadratic-phase slope as
`out->df = bin_delta * freq_step_df` (`analysis/spectral_peak_estimator.c:829`), where
`bin_delta` is the partial's sub-bin frequency drift across one hop
(`(best_next_bin + next_offset) − (bin + offset)`, line 796-799) and

```text
freq_step_df = 0.5 · (sr/n_fft) · (1/hop) · (2π/sr) = π / (n_fft·hop)   (sr cancels)
               (spectral_peak_track.c:706)
```

The persisted converter (`cmd/convert_segments.c:346-347`) then stores
`dst->df_q15 = FLOAT_TO_Q15(src->df / 1000)` — a *truncating* cast at plain-`Q15` scale
(LSB = 1/32768 ≈ 3.05e-5), with an additional `/SPECTRAL_MILLIS_PER_SECOND_F` (÷1000).

Measured (`python3`, exact `FLOAT_TO_Q15` truncation semantics), **without** the ÷1000, i.e.
the most generous reading — `df_q15 = trunc(bin_delta · π/(n_fft·hop) · 32768)`:

```text
config (n_fft,hop)   freq_step_df    bin_delta for df_q15≥1   df_q15 at the ±1-bin match the tracker targets
  4096,128           5.99e-06        ≥ 5.09 bins/hop          0
  2048, 64           2.40e-05        ≥ 2.05 bins/hop          0
  1024,256           1.20e-05        ≥ 5.09 bins/hop          0
  2048,512           3.00e-06        ≥ 17.0 bins/hop          0
```

A partial would have to sweep ≥5 FFT bins **per 128-sample hop** (canonical 4096/128 ≈ a
22 kHz/s glide) before the *first* `df_q15` LSB even turns on — far outside the ±1-bin
continuation interval the matcher targets, and the slope is truncated, not rounded. The actual
persisted converter divides by a further 1000, so it needs `bin_delta ≥ ~5090 bins` — more bins
than exist (`n_fft/2`). Conclusion: `FLOAT_TO_Q15(df)` (let alone `/1000`) is identically 0 for
every realisable chirp.

## Why this also breaks the "ENABLER for F1 on embedded" claim

F1's MQ linkage (PASS194) stores its **cubic** coefficients in the float `Segment`'s spare pad
(`_pad_w[0..2]` via `spectral_segment_set_cubic`), not in `df_q15`. Even the *quadratic* `df`
that does feed `df_q15` is ≤ ~6e-6 for the ±1-bin links F1 creates, so it vanishes under `Q15`
exactly as above. The Q15 segment format simply cannot carry F1's chirp at any useful
resolution; O4-A as written cannot be the embedded enabler the plan intends.

## Why not "just consume it anyway under the flag"

`SPECTRAL_HAS_CHIRP` is default-off, so a flag-gated consume would keep the default build
bit-identical — but it would deliver **nothing** (the source is 0) while:

- adding a per-sample `freq_inc += freq_delta` to the M7 inner loops and forcing the
  constant-`freq_inc` 4-sample batch (`spectral_phase_batch4`) off the fast path — a real cost
  in the CTF-audited hot loop;
- requiring the loader/validator rejection (`spectral_arm32_segment_chirp_supported`,
  `spectral_synth_arm32.c:134-144`, called at :419, with the Pass-22 overflow-guard lineage) to
  be loosened for data that is always 0 anyway.

Touching CTF-audited code to enable a provably-zero effect violates the speed-first hierarchy
(net pessimisation) and the "don't disturb audited code for unmeasured gain" rule. Skipped.

## The real prerequisite (maintainer-gated — wire-format + behaviour change + new golden)

To make embedded chirp real, the *storage*, not the consume, is the blocker. A corrected design:

```text
1. Carry the slope at a scale that survives 16 bits.  df ranges ~[6e-7, 3e-4] rad/sample² for
   bin_delta ∈ [0.1, 50]; a fixed scale S ∈ [1.7e6, 1.1e8] (e.g. S = 2^24) keeps the smallest
   realistic slope ≥1 LSB without overflowing the largest.  EITHER reinterpret the existing
   int16 df_q15 at scale 2^24 (not plain Q15), OR store the recurrence delta directly as the
   q31 freq_delta the SOA already reserves: freq_delta = 2·β_phase = df · 2^32/π per sample²
   (bin_delta=1 @4096/128 → 8192 — ample q31 resolution).
2. Stop forcing df_q15 = 0 in segment_to_q15(); populate the new field.
3. Fix convert_segments.c — drop the spurious /1000 on df (the estimator already emits a
   per-sample slope) and use the new scale.  This is a .spq SEMANTIC change.
4. Derive active_soa.freq_delta on activation (currently the q31 field at
   spectral_synth_arm32.h:54 / spectral_q15.h:208 is allocated and swap-copied at
   spectral_synth_arm32.c:751 but NEVER populated and NEVER consumed — dead state today).
5. Accumulate freq_inc += freq_delta in synth_core_m7/synth_fade_m7/generic; re-anchor at
   segment boundaries to bound drift (Q4).
6. Drop the load-time rejection.
```

Steps 1+3 change a **persisted binary format**; steps 4-6 modify **CTF-audited** code; the whole
is a **behaviour change** whose verification the plan itself designates as a new "arm32
exact+chirp" golden (Part D, line 365). All three — wire format, audited hot path, new signed-off
golden — are maintainer decisions. Per the golden-gated rule this pass does not make them
unilaterally; it hands over the precise, measured spec instead.

## Verification

```text
- No code changed (docs-only rejection); the tree remains at the PASS196 green baseline
  (5 production builds clean, ctest 4/4, default CPU output byte-identical to
  build/golden/cpu_sine_ref.wav).
- Numerical claims reproduced with python3 using the exact FLOAT_TO_Q15 truncation semantics
  (SPECTRAL_Q15_SCALE = 32768): freq_step_df = π/(n_fft·hop) for (4096,128),(2048,64),
  (1024,256),(2048,512); df_q15 = 0 at the ±1-bin match across all four, first nonzero LSB at
  bin_delta ≥ 5.09 (canonical) / 2.05 / 5.09 / 17.0 respectively.
- Source-level confirmations: segment_to_q15 forces df_q15=0 (spectral_synth_simulation.c:149);
  compact variant omits df_q15 (spectral_q15.h:170-179); df = bin_delta·freq_step_df
  (spectral_peak_estimator.c:829); persisted store FLOAT_TO_Q15(df/1000)
  (convert_segments.c:346-347); freq_delta is never populated/consumed (only swap-copied,
  spectral_synth_arm32.c:751).
```

## Status

O4-A **REJECTED-as-specified** (the `df_q15` source is identically 0 for all realisable inputs;
consuming it is a guaranteed no-op that would pessimise the CTF-audited ARM loop). The corrected
prerequisite — finer-scale chirp storage in the Q15/`.spq` segment — is a wire-format +
behaviour change requiring a new signed-off "arm32 exact+chirp" golden and is left to the
maintainer. Per exec order the next actionable item is **F2** (density-adaptive IFFT synth),
which is itself gated on the Q1/Q2 regime benchmark; absent that benchmark the next
self-contained, default-bit-identical wins are **O2-B** (32B `SPECTRAL_COMPACT_SEG` packing) and
**O5** (dead-code removal of the no-op stubs). Plan updated to mark O4-A REJECTED.
