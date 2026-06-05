# Patch notes — Pass 214: Bv — vectorized uint32 8-wide NCO phase (the deferred Q5c follow-up, scoped on c3==0)

## Scope

The **Bv** follow-up deferred by PASS213 (Q5c): replace the packed-Q15 kernel's *serial scalar
phase* — eight sequential `spectral_phase_nco_step` calls per 8-sample block — with a vectorized
**uint32 8-wide cubic NCO** that emits eight consecutive Q15 phase indices per `simde__m128i` per
step. PASS213 measured the serial phase as ~31–35% of the shipped kernel; this is the slice that
recovers most of it.

**Measure-first, then SCOPE on the data.** The deferral reason was precision: uint32 lanes keep
only **16 fractional bits** where the scalar uint64 NCO keeps 48, so the cubic third difference
loses headroom. The precision harness (`phase_nco_precision` Part 3, new this pass) measured the
vec NCO against the scalar oracle and the double truth across the worst-case segment set and found
a clean boundary:

- **c3 == 0** (linear / quadratic phase — the default model): vec holds the index to **≤2 LSB**
  of the scalar NCO (≤ −120 dBFS through the steepest timbre). The 16 fractional bits are plenty
  when there is no cubic term.
- **c3 != 0**: the narrowed third difference drifts — 44 LSB / −68 dBFS on the long aggressive
  cubic — exactly the regime the uint64's 48 fractional bits exist for.

The shipping `c3` default is **exactly `0.0f`** (`spectral_synth_internal.c`: `c2 = beta;
c3 = 0.0f;`, nonzero only under the `SPECTRAL_PRECISE_PHASE` build flag *and* track-cubic
linkage). So the win path is the common path. The decision is therefore **not** GO-everywhere and
**not** NO-GO, but **SCOPE on `c3 == 0`**: production uses the vec NCO when `lp->c3 == 0.0f` and
falls back to the shipped scalar pack8 when `c3 != 0`. `c3 == 0` is an **exact** boundary — no
calibrated threshold, no relaxed budget, zero precision regression anywhere.

**Default build is byte-identical.** Unchanged from Q5c: the packed path is opt-in (`osc_set_q15_
enable`, mask defaults 0, no CLI wires it) AND requires the timbre's float dispatch to resolve to
SIMD. The shipping float default moves no bytes; `osc_parity` confirms.

## What changed

### `core/spectral_phase_nco8.h` (new) — the vectorized 8-wide NCO

`SpectralPhaseNco8`: eight independent stride-8 cubic forward-difference chains in uint32 lanes
(full circle ≡ 2^32, top 16 bits = the signed Q15 index, low 16 = fractional headroom).

- `spectral_phase_nco8_seed(const SpectralPhaseNco*)` — seeds all 8 lanes EXACTLY from a scalar
  uint64 NCO positioned at the sustain start. Reads 25 consecutive scalar phases (cold, once per
  block) and forms each lane's stride-8 forward differences in full uint64 precision, narrowing to
  uint32 only at the end (`>> 32`). Wrap is free (mod-2^64 uint64 arithmetic). Consumes a copy, so
  the caller's NCO is untouched. Boundary helper (does the uint64→uint32 narrow) — kept OUTSIDE the
  Q-domain markers.
- `spectral_phase_nco8_step(SpectralPhaseNco8*)` — emits 8 consecutive Q15 indices (round-to-
  nearest via half-LSB bias, `packus_epi32` int32→uint16) and advances every lane by 8. Pure
  integer SIMD, wrapped in `// SPECTRAL_Q_DOMAIN` markers (no float/double token — `q_domain_
  contract` enforced). Host generic-SIMD only (`OSC_SIMD_GENERIC`); embedded keeps the scalar NCO.

### `core/port/host/oscillator_simd.c` — SCOPE-on-c3==0 wiring in `osc_simd_q15_segment`

After the (scalar) fade-in, `use_vec = (lp->c3 == 0.0f) && (j < len)`. When set, the vec NCO is
**seeded once** from the scalar NCO at the sustain start and is then the **sole** phase source for
the whole `[fade_in_end, len)` span:

- sustain SIMD blocks pull a full 8-index register from `spectral_phase_nco8_step` (replacing
  `osc_q15_nco_pack8`'s 8 serial scalar steps);
- the per-sample sustain-tail and fade-out draw one index at a time from an 8-lane buffer
  (`osc_q15_vnco_next`) fed by the same vec chain, so phase stays continuous across the
  block→tail→fade boundaries without ever re-stepping the scalar NCO.

When `c3 != 0`, `use_vec` is false and every region keeps the exact shipped scalar pack8 path —
bit-for-bit the PASS213 kernel.

### `tests/core_contracts/test_phase_nco_precision.c` — Part 3 (vec drift) + conditional gate

New `part3_vec_index_drift()` (guarded `OSC_SIMD_GENERIC`): seeds the vec NCO from the scalar NCO
at t=0 and, per worst-case segment, reports the worst lane index error vs the double truth AND vs
the scalar oracle, plus the saw-eval dBFS impact. The gate is **conditional on c3**, asserting the
shipped guarantee directly:

- `c3 == 0` segments (`[vec-shipped]`) — gated TIGHT at `NCO_INDEX_LSB_GATE` (3 LSB), the same
  floor as the scalar NCO;
- `c3 != 0` segments (`[scalar-fallback]`) — gated only by a loose 128-LSB gross-bug tripwire; the
  printed drift documents *why* production scopes vec to c3==0.

### `tests/core_contracts/test_q15_simd_parity.c` — cubic-stress row (the scope-boundary CI lock)

Added an aggressive cubic segment (c2=5e-5, **c3=3e-7**, len=2000) alongside the four c3==0 rows.
The four c3==0 rows exercise the new vec path; the cubic row exercises the scalar fallback. Both
stay within the −84 dBFS budget (saw −123, parabola −95). The cubic c3 is large enough that a
**broken scope gate** (vec wrongly applied to c3 != 0) would drift the phase ~31 LSB → **−25.9
dBFS** standalone, blowing the budget by tens of dB — so this row is the CI lock on the c3==0
boundary, not just fallback coverage (mirrored as `parity-cubic` in the precision Part 3 table to
prove the misroute magnitude).

## Verification

```text
- ctest 11/11 PASSED (q_domain_contract, q15_simd_parity, q15_production_parity, phase_nco_
  precision, osc_parity, osc_width_parity, + core suite).
- phase_nco_precision Part 3 (vec vs scalar oracle, LSB / saw-eval dBFS):
    linear-mid    1 LSB   -300.0  [vec-shipped]    near-nyquist  2 LSB  -120.4  [vec-shipped]
    quad-chirp    1 LSB   -120.4  [vec-shipped]    cubic-chirp   2 LSB   -93.1  [scalar-fallback]
    long-cubic   44 LSB    -68.0  [scalar-fallback] parity-cubic 31 LSB  -25.9  [scalar-fallback]
  c3==0 holds ≤2 LSB (TIGHT gate 3); c3!=0 drift confirms the scope boundary (tripwire 128).
- q15_simd_parity (SIMD vs scalar Q15, budget −84 dBFS), now incl. the cubic-stress fallback row:
    saw −123.3   square −300.0 (bit-identical)   triangle −116.9   parabola −95.4   all PASS.
- q_domain_contract green: spectral_phase_nco8_step's // SPECTRAL_Q_DOMAIN block has no float/
  double token; the seed (uint64→uint32 narrow) stays outside the markers.
- Production builds: desktop, simulate, simulate-daisy rebuild clean, no warnings. cuda env-
  blocked (no nvcc). Default byte-identity: opt-in path, mask defaults 0 — osc_parity unchanged.
```

## Throughput (`bench_q15_pack8`, Apple Silicon, ns/sample — lower is faster)

Column **Bv** (pack8, vectorized uint32 phase) is now exactly the shipped `osc_simd_q15_segment`
for c3==0; **A** is the real production float-SIMD path. **A/Bv = speedup.**

```text
  timbre     A prod   B scal   Bv vec    speedup(Bv/A)   was(B/A)
  saw         0.413    0.302    0.263       1.57×           1.37×
  square      0.438    0.315    0.285       1.54×           1.39×
  triangle    0.436    0.323    0.279       1.56×           1.35×
  parabola    0.438    0.350    0.309       1.42×           1.25×
  sine        0.876    0.893    0.761      (1.15×)         excluded — stays scalar Q15
```

So **~1.42–1.57× over production float-SIMD** on the four algebraic timbres, up from the PASS213
shipped 1.25–1.39×. The vectorized phase recovers most of the serial-phase gap PASS213 identified:
the `B0` no-phase floor (~0.21 ns/sample) is the eval+widen+accumulate density limit, and Bv now
sits much closer to it. The residual gap to the structural 2× ceiling (int16:float32 = 2:1 lane
density) is the unavoidable phase + widen + accumulate overhead on this 128-bit-NEON host.

**Sine note:** with the vec phase, sine's pack8 (0.761) now edges production float-SIMD (0.876) —
but sine stays excluded (`osc_simd_q15_available` gates it out): its serial-LUT Q15 path was never
re-validated for SIMD and is a separate axis. Clean future follow-up, decline-on-data as warranted.

## Status

Bv lands, scoped on `c3 == 0`: the production opt-in Q15 SIMD kernel now uses a vectorized uint32
8-wide NCO on the common (quadratic) phase model — ~1.42–1.57× over float-SIMD, up from 1.25–1.39×
— and falls back to the bit-identical scalar pack8 on cubic-linked segments where the uint64's 48
fractional bits matter. Parity-locked to its scalar oracle (3/4 bit-identical on the fallback
path), precision-gated tight on the shipped path, scope-boundary CI-locked by the cubic-stress
parity row, default byte-identical, host-only.

**Deferred (unchanged):** the ~8 KB `g_osc_q15_sine_lut` `#if !SPECTRAL_EMBEDDED` footprint guard
(scalar-sine concern, independent of this host-only kernel). The sine-pack8 SIMD-Q15 routing
question (above) is a new, separate clean follow-up.
