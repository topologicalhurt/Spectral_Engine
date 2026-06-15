# IFFT Synthesis (Rodet–Depalle) — F-stream plan

Decision basis (closed): P6/D4 priced the fork structurally; pass 255 measured
it — one CMSIS-DSP inverse RFFT-512 (Q31, Daisy flags) = **43,910 insns**
[measured: qemu-tcg] vs **6,528 insns per oscillator-bank partial per
256-sample block** [measured] ⇒ crossover ≈ **7 dense partials** (4–16 across
the IPC band). Dense material is ~8× cheaper at 64 partials and scales. The
oscillator bank stays the exact path and the default; IFFT is an
approximation and ships **only behind a gate + golden sign-off** (AI_CANON
rule 3).

## Method (one paragraph)

Per output frame of N samples (hop H = N/2): for each partial, add its
**spectral motif** — the sampled spectrum of the synthesis window, precomputed
oversampled — into the frame's complex half-spectrum at the partial's
fractional bin, scaled by amplitude and rotated by the phase at frame center;
one inverse real FFT; multiply-free overlap-add (the synthesis window is baked
into the motif). The synthesis window is **periodic Hann at 50% overlap** ⇒
exact COLA (gain 1, verified by `spectral_overlap_add_is_constant`). Window
convention note: this is the IFFT path's INTERNAL synthesis window — the
analysis-window convention decision (REVIEWER_HANDOFF_2 §4.4) stays untouched
and no analysis golden moves.

## Error budget (the gate criteria)

Approximation sources, each bounded in the F1 harness before integration:
1. **Motif truncation** (K bins per side): MEASURED (F1, pass 257, double
   precision, N=512): frame-level max error K=4 → −43 dBFS, K=8 → −55,
   K=12 → −63 (Hann decays ~1/d³; the original −80 dB@K=8 guess was wrong —
   measurement replaced it). Stream-level after OLA: −68…−76 dBFS max,
   **−83…−92 dBFS RMS** at K=8–12. K (and/or a faster-decaying motif window
   pair) is an explicit F2 operating-point decision; per-partial cost is
   only ~2K complex MACs, so K barely moves the crossover.
2. **Motif interpolation** (oversample O, linear interp): MEASURED — O=16
   is within 1 dB of O=64 at every K (truncation dominates); small tables
   are fine.
3. **Intra-frame stationarity**: frequency/amp ramps (df/da) are held to
   frame-center values; error grows with |df|·H — the hybrid sends fast
   chirps/fades to the oscillator bank (F2 routing rule, measured threshold).
Parity ladder: (a) frame-level — built spectrum iFFT'd vs time-domain
windowed cosine; (b) stream-level — OLA output vs the exact oscillator sum
over stationary partials; (c) golden — full pipeline render, maintainer signs.

## File organization (pass-256 maintainer ruling)

One contract header `synth/spectral_synth_ifft.h` in OUR types; per-backend
TUs in port dirs, never macro mono-files: `port/host/spectral_ifft_vdsp.c`
(+ FFTW variant), `port/cmsis/spectral_ifft_cmsis.c` (Q31). Third-party
headers do not leak through the contract; every port pair gets a parity test
(`window_backend_parity` pattern). The motif table is generated (SSOT
generator, committed artifact — resource-hash pattern).

## Phases

- **F1 — math harness (no engine code).** `tests/core_math/harnesses/
  ifft_synth_sweep.c`: motif build + frame builder + reference DFT + OLA;
  measures the error-budget numbers above; pytest driver asserts the ladder
  (a)+(b) within the measured-then-frozen bounds. Done-when: error floor
  numbers recorded here and the harness is green.
- **F2 — desktop reference path.** `spectral_synth_ifft.c` (float, host FFT
  backend via the contract header), opt-in flag (no default change), hybrid
  density router (oscillator bank below crossover / fast chirps; IFFT above),
  stream parity test vs osc-bank in ctest. Done-when: ctest parity green +
  measured desktop speedup on a dense fixture.
- **F3 — golden sign-off.** Maintainer listens/diffs a dense render;
  thresholds frozen into the golden set. Done-when: signed.
- **F4 — ARM Q31 port.** `port/cmsis/` TU over `arm_rfft_q31`; Q-domain map
  rows for the spectral frame (Q31 bins, motif Q15/Q31 — decide by measured
  SNR); rig-measured cycles; m7-baseline scenarios extended with a dense-frame
  STONE ceiling. Done-when: rig numbers + gate extended + oracle green.
- **F5 — capacity republish.** Re-run the capacity table with the hybrid;
  update M7_PERF_MODEL_PLAN + the published guarantees.

## Status

- F0 (this plan): committed pass 257.
- F1 CLOSED (pass 257): `tests/core_math/harnesses/ifft_synth_sweep.c` +
  `test_ifft_synth_sweep.py` — COLA machine-exact; frame/stream parity
  ladder green; measured floors frozen into the pytest gate (+3 dB
  headroom). The frame-builder construction (centered motif, (−1)^k
  twiddle, Hermitian placement) is validated and is the SSOT formulation
  for F2.
- F2 core CLOSED (pass 258): contract `synth/spectral_synth_ifft.h` +
  frame renderer `synth/spectral_synth_ifft.c` (float port of the F1
  formulation, K=8/O=16 measured operating point) + two host iFFT ports
  (`core/port/host/spectral_ifft_vdsp.c`, `spectral_ifft_ref.c` — exactly
  one live per build, selected on SPECTRAL_USE_VDSP). ctest
  `ifft_synth_parity` (#20): port-vs-reference-iDFT 5.6e-9; stream parity
  vs the exact oscillator sum **−72.8 dBFS max / −87.5 dBFS RMS** at 64
  dense partials; deterministic; **MEASURED 7.5× over the naive oscillator
  loop on desktop** (matches the embedded pricing's ~8× at 64 partials).
- Next: F2b — the hybrid density router into the engine dispatch (segment
  semantics: stationary sine interiors → IFFT, fades/chirps/non-sine →
  oscillator bank; opt-in flag, no default change), then F3 golden.

## F2b — implementation design (characterized 2026-06-15, NOT yet built)

Characterized against the current code so the build is mechanical, not exploratory.
F2b is correctness-critical (a mis-routed fade is an audible click), so it is its
own focused unit, not a tack-on.

**The two facts the router must respect (verified in code):**
1. `spectral_ifft_synth_render(s, partials, n, out, total)` (spectral_synth_ifft.c:118)
   renders a *stationary* partial set over the WHOLE `total`: every frame evaluates
   ALL `n` partials at frame center (`phi = omega*center + phase0`, :154). It has no
   notion of a partial starting/ending. Real segments are time-localized (start,
   length).
2. The fade envelope is applied at RENDER time by the oscillator path (per-segment
   fade-in/out over `fade_len`), not stored in the Segment. The IFFT renders a
   partial as steady → routing a faded span to IFFT DROPS the fade (click). So a
   segment must be split: stationary interior → IFFT, fade edges → osc bank.

**Design:**
- **Renderer extension (per-frame activity).** Add a dynamic entry beside
  `spectral_ifft_synth_render` that drives the same frame loop + OLA (:141-167) but
  pulls the active partials PER FRAME from a caller callback
  `(frame_center_sample) -> {SpectralIfftPartial[], count}`. The framing/OLA/motif
  stay owned by the renderer (the subtle part); the engine owns "which partials are
  live this frame". Keep the existing static API (its parity test #20 stays the
  oracle for the dynamic path on a stationary fixture: dynamic-with-constant-set ==
  static).
- **Segment classifier** `is_stationary_sine_segment(seg)`: sine timbre AND
  |da|·length below the F1 intra-frame-stationarity threshold (38-40) AND |df|≈0.
  Non-sine / chirped / amplitude-ramped → never IFFT.
- **Interior/edge split.** For an eligible segment, only `[start+fade_len,
  end-fade_len]` is IFFT-routed; `[start, start+fade_len]` and `[end-fade_len, end]`
  go to the osc bank (which already applies the fade ramp). The per-frame callback
  emits a partial for a segment only when the frame center lies in its interior;
  `bin = omega·n_fft/2π`, `amp = seg.amp`, `phase0 = seg.phase − omega·seg.start`
  (so the renderer's `omega·center + phase0` reproduces the segment phase model).
- **Sum.** IFFT interiors OLA into the output buffer; the osc bank renders edges +
  all non-eligible segments into the SAME buffer (additive). One shared output.
- **Crossover gate.** Only engage IFFT for frames with ≳7 simultaneous eligible
  partials (the F1/P6 measured crossover); below that the osc bank is cheaper — so
  the router counts per-frame eligible density and falls back when sparse.
- **Opt-in.** A compile gate (default 0) at the synth_cpu seam (around
  `synth_cpu_driver`, spectral_synth_cpu.c:431): default OFF → the IFFT code is not
  compiled, the default render path is byte-identical, m7 perf gate untouched.

**Test (done-when):** a tolerance stream-parity ctest — IFFT-routed engine render vs
pure osc-bank render — within the F1-frozen floor (≈ −72 dBFS max / −87 dBFS RMS at
64 dense partials, +3 dB headroom), on a dense stationary fixture AND a mixed fixture
(dense interiors + faded/chirped segments) that exercises the split seam (assert no
discontinuity at the interior↔edge handoff). Plus an informational desktop speedup.

**Risks to verify during build:** (a) phase continuity at the interior↔edge seam
(IFFT and osc must agree on the partial's phase at the boundary sample); (b) OLA
coverage at interior boundaries (the renderer needs frames overlapping the interior
edges); (c) the crossover fallback must not double-count a partial (IFFT or osc, never
both, for a given span). Default-on acceptance rides F3 (golden, maintainer).
