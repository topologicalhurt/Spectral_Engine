# Patch notes — Pass 194: Optimisation track F1+F3 — MQ track linkage + cubic-phase interpolation (double-gated, default bit-identical)

## Scope

First implementation pass of the **optimisation track** (`docs/core_audit/OPTIMISATION_PLAN.md`),
companion to the master `ULTRAPLAN.md`. Implements Foundations **F1** (make tracks
first-class via McAulay-Quatieri peak continuation) and **F3** (per-track cubic-phase
interpolation) as a single, minimal **Option 1** design: cross-frame track linkage that
feeds cubic phase coefficients into the existing per-hop segment model — *without* merging
grains into variable-length segments (Option 4), which would break the fixed-segment
invariants for zero audible gain.

Reference: McAulay & Quatieri, "Speech Analysis/Synthesis Based on a Sinusoidal
Representation," IEEE ASSP-34(4), 1986 — eqs. 33-38 (cubic phase + maximally-smooth unwrap).

## Design — double-gated, provably bit-identical when off

```text
Gate 1 (compile): SPECTRAL_PRECISE_PHASE (spectral_config.h, default 0). When 0, the
   cubic coefficients reduce to (c2 = beta, c3 = 0) and the linkage stage is a no-op.
Gate 2 (runtime): the optional `adaptive_track_density` processing mask (opt-in, default
   NONE). The cubic annotation is only written when this stage runs.
```

The synth phase evaluator was switched from the quadratic helper to a Horner cubic helper
`spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, off)`. With `c2=beta, c3=0`
this is **bitwise identical** to the old quadratic helper (IEEE multiply commutes; the
`+ off*0` term is a hard `+0.0f`). Proven by exhaustive sweep: 598,381 representative
`(phase0, alpha, beta, offset)` tuples — zero mismatches. The only divergence is the
physically-unreachable all-`-0.0` degenerate (requires `omega == -0.0`; peak estimation
emits `omega > 0`), and even then a `±0.0` sample vanishes under the `dst[] += amp*wave`
accumulator.

## Algorithm (stage body, `spectral_proc_adaptive_track_density.c`)

```text
- Segments arrive globally frame-ordered (tracker finalize). Walk maximal runs of equal
  frame index (frame = round(start/length); length == hop, constant).
- For consecutive frames f, f+1: each predecessor links to the nearest-frequency successor
  within a tolerance of ~1 FFT bin (2*pi/n_fft rad/sample; relative fallback when n_fft
  unknown). Track death (no successor in tolerance) keeps the quadratic model.
- MQ cubic coefficients in the analysis domain over T = hop:
      M*  = round( ((theta_k + w_k*T - theta_k1) + 0.5*T*(w_k1 - w_k)) / 2pi )   (eq. 38)
      dtheta = theta_k1 + 2pi*M* - theta_k - w_k*T ;   dw = w_k1 - w_k
      a2 = 3*dtheta/T^2 - dw/T ;   a3 = -2*dtheta/T^3 + dw/T^2                    (eq. 36-37)
  stored via spectral_segment_set_cubic() in the 64B Segment's spare pad (_pad_w[0..2];
  finalize already zeroes the pad so an un-annotated segment reads "absent").
- segment_loop_params_init transforms analysis coeffs to the synth (pitch/stretch) domain
  the same way alpha/beta are: c2 = a2*pitch*inv_stretch^2, c3 = a3*pitch*inv_stretch^3.
```

## Files changed

```text
- core/spectral_config.h            + SPECTRAL_PRECISE_PHASE gate (default 0)
- core/spectral_segment_math.h      + spectral_segment_phase_at_cubic_f32 (Horner cubic)
- core/spectral_common.h            + cubic pad accessors (has/set/get c2,c3) on 64B Segment
- core/spectral_synth_internal.h    + c2,c3 in SegmentLoopParams
- core/spectral_synth_internal.c    compute c2/c3 (default beta,0; override under flag);
                                    endpoint validation now via cubic helper (bit-identical)
- core/oscillator.c                 synth_segment_scalar takes c2,c3; SIMD bypass→scalar
                                    when a segment carries cubic (under flag only)
- synth/backends/cpu/spectral_synth_cpu.c   3 phase evaluators routed through cubic helper
- analysis/spectral_processing_chain.h/.c   + hop,n_fft in SpectralProcessParams + signature
- cmd/cli/spectral_cli_pipeline.c   thread opts->hop/opts->n_fft into the chain
- analysis/spectral_proc_adaptive_track_density.c   stub → MQ linkage stage (under flag)
```

## Verification

```text
- Five production targets build clean (desktop, simulate, simulate_daisy, embedded_arm,
  embedded_arm_float) — only the pre-existing benign -mavx2 / -mno-avx512f notes.
- ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift).
- Phase-helper equivalence sweep: 598,381 tuples bit-identical (degenerate all-`-0.0`
  excluded — unreachable, output-invariant).
- End-to-end (CPU backend, sin_440hz.wav):
    * SPECTRAL_PRECISE_PHASE=1 build, NO mask  == default build  → byte-identical (cmp).
    * SPECTRAL_PRECISE_PHASE=1 build, adaptive_track_density mask → differs:
      max_abs_diff = 1.6e-3 (~-56 dB), rms = 6.1e-4 (~-64 dB). A sub-perceptual
      phase-continuity refinement (Q3 satisfied), not a regression.
- Golden refs captured: build/golden/cpu_sine_ref.wav (flag-off reference),
  build/golden/cpu_sine_cubic_ref.wav (signed-off flag-on reference), compare.py restored.
```

## Status

F1 + F3 implemented and gated. Default builds are unchanged (bit-identical). The cubic
path is opt-in, behaviour-change golden-signed-off, and fully reversible via either gate.
Next optimisation-track items (per plan exec order): O4-B, O1-B, O4-A, then F2/O2-A/O3-A.
