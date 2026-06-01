# Patch notes — Pass 157: sim becomes a perf model over the REAL spectral_arm32_process (Phase A / A1b#3)

## Problem

`synth/backends/sim/spectral_synth_simulation.c` (the `make simulate` Q15 ARM
oracle) carried a **parallel reimplementation** of embedded synthesis. It defined
its own `SimSegment` / `SimActiveSegment` types, its own phase accumulator
(`spectral_sim_phase_acc_from_q15` / `_add_inc` / `_add_scaled`), its own
saturating Q31 mixer (`spectral_sim_q31_add_scaled_sat`) and its own per-block
scheduler — and produced every output sample from that second code path. The
real backend, `spectral_arm32_process()`, was never executed.

This is the AI_CANON §7 duplication defect: the "oracle" validated nothing about
the shipping code because it re-derived the answer with different math. Campaign 1
already paid for this — passes 145/146 found two real embedded bugs (frequency
~7019× too low; amplitude −6 dB) that the sim had **masked** for the entire
project because the sim's reimplementation happened to be self-consistently
wrong-in-a-different-way. The locked decision (see `project_sim_perf_model_rework`
memory) is: the sim is a *perf/resource model* layered over the SAME real code;
correctness is owned by CTest; the reimplementation is deleted.

## Change

Audio output now comes solely from `spectral_arm32_process()`. The parallel
synthesis math is gone; what remains on the sim side is a legitimate workload
**cost model** (op counts, cache pressure, cycle estimate) that re-walks the
segment schedule to *count* work without producing samples.

```text
DELETED (parallel reimplementation):
  SimSegment, segment_to_sim, SimActiveSegment,
  spectral_sim_phase_acc_from_q15 / _add_inc / _add_scaled,
  spectral_sim_q31_add_scaled_sat, find_first_segment_at.

ADDED  segment_to_q15(src, dst, amp_scale, params, out_len)
  Float Segment -> embedded SpectralSegmentQ15, the desktop-side equivalent of
  cmd/convert_segments.c but additionally folding the runtime stretch/pitch in
  at conversion time (the real backend applies neither: spectral_arm32_set_stretch
  is a no-op). start/length scaled by stretch (length clamped to the u16 field);
  freq_q88 = OMEGA_TO_Q88(alpha(omega, pitch, inv_stretch)); amp_q15/da_q15 scaled
  by the Q15 headroom factor; df_q15 forced 0 (the ARM32 hot path does not consume
  chirp and spectral_arm32_load() rejects df_q15 != 0). Returns 0 to DROP a
  segment that cannot be represented.

REWROTE synth_arm32_simulation() (signature unchanged):
  preflight + out_len overflow guard + timbre fallback  [unchanged]
  -> convert sa -> q15_src, dropping invalid, compacting to `loaded` (so the
     strict loader sees only loadable, start-ordered data)
  -> heap ctx + working buffer q15_ctx
  -> spectral_arm32_init(ctx, q15_ctx, loaded, get_simulation_lut(), sample_rate)
  -> spectral_arm32_load(ctx, q15_src, loaded, out_len)   <-- runs the REAL
     validate_segment_data (monotonic start+end, <=512 active, chirp bound);
     on reject: log warn + propagate the loader's SpectralError.
  -> WORKLOAD-ACCOUNTING MODEL: walk the same per-block schedule the real process
     follows and tally activations / scan length / peak active / per-sample
     LUT+MAC+phase ops -> EmbeddedOpCounts via spectral_perf_count_*; estimate
     block cycles via spectral_perf_model_estimate_block_cycles. NO audio here.
  -> AUDIO (real code): drive spectral_arm32_process() across out_buffer in
     <=256-sample chunks; widen each Q15 result to float (Q15_TO_FLOAT).
  -> embedded_perf_estimate / _print + embedded_memory_usage / _print  [unchanged]
```

The workload walk is intentionally retained and is **not** the forbidden
duplication: it computes no audio, only a cost estimate, which is the sim's entire
reason to exist (the embedded perf/memory report). The single source of audio
truth is the shipping `spectral_arm32_process`.

### Behavioral consequence (faithful to target)

Because the real loader is now in the path, the sim is subject to the embedded
target's actual constraints. Dense real-analysis inputs whose segments are not
monotonic in both start and end, exceed 512 simultaneously active, or carry a
non-zero chirp will be dropped at conversion or rejected by the loader. That is
the target's behavior, not a sim artifact — surfacing it is the point.

## Verification

```text
- six green targets build clean: desktop, simulate, embedded_arm,
  embedded_arm_float, simulate_daisy, arm_core_test (only the pre-existing benign
  -mavx2 / -mno-avx512f unused-arg notes; embedded_arm_restricted remains a
  pre-existing link failure, out of scope / not a green target).
- ctest -R arm32_process_correctness: PASSED (1/1) — authoritative correctness,
  exercises spectral_arm32_process directly.
- sim render of fixtures/sine_bin.wav: 82 segments accepted, peak active 2,
  exit 0 (loader did NOT reject a clean input).
- sim render of fixtures/two_tone.wav: 164 segments accepted, peak active 4.
- spectral comparison sim vs desktop CPU-float render (same fixtures/args):
      sine_bin  SIM dominant 430.7 Hz  peak 0.9500   | DESK 430.7 Hz  peak 0.9500
      two_tone  SIM {430.7,1076.7} Hz  peak 0.9500   | DESK same pair peak 0.9500
  Dominant frequencies and peak amplitude MATCH. Time-domain sample diff is large
  (sine_bin max-abs 0.63, rms 0.25) and RMS differs (two_tone 0.62 vs 0.39) — this
  is EXPECTED: freq_q88 is Q8.8 (step 1/256 rad/sample), so at audio frequencies
  the fixed-point frequency is coarse (a ~43 Hz bin-1 tone quantizes to ~27%
  error), accumulating multi-cycle phase drift over 0.5 s and altering beat
  patterns. The sim is faithful to the fixed-point TARGET, not to the float ideal;
  matching the float render sample-for-sample is neither expected nor desired.
```

## Scope (Phase A increment)

A1b#3 only: the sim no longer reimplements synthesis; all audio flows through the
real `spectral_arm32_process`, and the perf/resource model is retained as the
measured side. Net −~250 lines in `spectral_synth_simulation.c`.

Deferred to follow-on passes:
- **A1b#4 (pass 158):** retire the interim `tests/arm_oracle/oracle.py` and its
  goldens — they are now stale by design (output sourced from the real code), and
  CTest `arm32_process_correctness` is the authoritative correctness gate. Tidy
  the `synth_cpu` redirect comment.
- **A2/A3/A4:** hardware-gated ARM redesign items — documented as deferred (no
  Cortex-M hardware / QEMU calibration available in this environment).
