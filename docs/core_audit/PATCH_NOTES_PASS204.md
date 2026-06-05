# Patch notes — Pass 204: U3 — optimize the oversample band-limited renderer

## Scope

Oscillator-unification step **U3** (`docs/core_audit/OSCILLATOR_UNIFICATION_PLAN.md`):
optimize the CPU-float band-limited quality path `core/spectral_osc_bandlimited.c`.
Opt-in, **not** a golden contract (default quality is NAIVE, which never enters
this file). Measure-first: profile all three modes, then fix the one hotspot.

## Profile (measure-first; shakespeare 0–8 s window, 922 082 segments, 1 thread)

```text
mode         Synth ms   vs naive
naive          115.0     1.0x   (point-sampled reference; never enters this file)
polyblep       624.9     5.4x   branch-light per-sample, already cheap
additive      2761.5    24.0x   O(harmonics) Chebyshev recurrence, 1 sin+1 cos/sample
oversample    7607.7    66.2x   <-- the hotspot
```

`polyblep` and `additive` were profiled and left untouched: polyblep is already a
per-sample closed form, and additive is already O(N) via the Chebyshev recurrence
(one sin/cos per sample, add-only across harmonics). Their outputs are byte-
identical after this pass (verified). All actionable cost was in **oversample**.

## What oversample was doing wrong

For every one of the 922 082 segments the renderer:

1. **Rebuilt the decimation FIR from scratch** — `osc_bl_build_fir` runs 65 `sinf`
   + 65 `cosf`. The filter depends only on compile-time constants, so this was the
   *same 65 coefficients* recomputed ~922 k times (~120 M transcendental calls).
2. **`malloc`/`free`d the OS×len scratch** on every call.
3. **Ran a 65-tap decimation with a clamp branch on every tap**, even though the
   filter window is fully in-bounds for all but the first/last `OS*FIR_TAPS`
   output samples.

## The fixes

- **Build the FIR once per thread.** `osc_bl_fir()` lazily fills a thread-local
  `g_osc_bl_fir[65]` on first use. Thread-local (`SPECTRAL_THREAD_LOCAL`) keeps it
  lock-free under the OMP segment-parallel synth loop — each worker builds one
  copy instead of one-per-segment.
- **Reuse a thread-local oversample scratch.** `osc_bl_os_scratch(need)` grows a
  per-thread buffer monotonically to the largest segment that thread renders, so a
  `malloc`/`free` pair per segment becomes (after warm-up) zero allocations. The
  buffer lives for the thread's lifetime — released by the OS at process exit,
  matching the existing `gpu_tile_cache` thread-local pattern in
  `spectral_synth_internal.c`.
- **Split decimation into branch-free interior + clamped edges, with the symmetry
  fold.** The window is zero-phase symmetric (`h[n] == h[L-1-n]`), so each tap pair
  ±m shares weight `h[H-m]` → half the multiplies. Output samples whose whole
  `[center-H, center+H]` window is in-bounds (all but `OS*FIR_TAPS` edge samples)
  take a branch-free inner loop; only the edges keep the clamped form.

## Result (best of 3, shakespeare 0–8 s, 1 thread)

```text
mode         before     after     speedup
oversample saw    7607.7    1609.6     4.73x
oversample quant  9531.7    3535.7     2.70x   (quant's naive eval is heavier, so the
                                                4x render evals dominate more -> the
                                                FIR/decimate savings are a smaller share)
```

Multithreaded (8 threads) oversample saw runs in ~277 ms and is correct — confirms
the thread-local FIR + scratch are race-free under the OMP loop.

## Behavior delta (measured, not asserted)

- **Default build (NAIVE): byte-identical** — naive output hash unchanged
  (`e3ecf301…`); the optimized code is never reached at the default quality.
- **`additive` / `polyblep`: byte-identical** (untouched).
- **FIR-precompute + scratch-reuse alone: byte-identical** to the original
  malloc+per-call-FIR oversample (proven: a reference build with only the
  decimation reverted reproduced the exact pre-pass hashes `55c6ae41…` /
  `e595a555…`). So the *only* output-affecting change is the symmetry fold.
- **Symmetry fold:** vs the per-tap ascending reference, the oversample output
  shifts by **max 2.98e-7 (-130.5 dBFS), RMS 1.4e-8 (-157 dBFS), ~-135 dB below
  signal** — ~1 ULP, from the reordered tap sum + the shared symmetric coefficient
  (`h[H-m]` for both lobes; the true zero-phase filter *is* symmetric, so this is
  arguably cleaner than the original's `cosf`-rounding asymmetry). Inaudible, and
  far below the -132 dBFS the maintainer already accepted for the SIMD re-baseline.
  This is an audition/quality mode, not a golden contract.

## Verification

```text
- 5 production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float).
- ctest 5/5 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift, osc_parity).
- All 8 timbres render in oversample mode (incl. asin/quantized, which only
  oversample tames).
- Default naive build byte-identical; additive/polyblep byte-identical.
```

## Proposed next pass

**U2** — adversarial-correctness audit of the *optimized* `spectral_osc_bandlimited.c`
(NaN/Inf propagation, boundary phase, the timbre×quality fallback matrix, the
new thread-local scratch alloc-failure path, and decimation index/alignment at the
edges and at the new interior/edge split boundary).
