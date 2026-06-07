# GPU Q15 / fp16 double-pack audit (Oscillator-Backend-Contract Phase 5)

**Status:** AUDIT COMPLETE — **measured throughput WIN**, distinct from the Phase-4 / CPU
declines. **No production code is wired.** Promotion is a maintainer decision: it changes
GPU render precision (to ~−67 dBFS vs today's fp32-exact GPU path) and is a quality-floor
judgment, so it would land as an **opt-in mode**, not a silent default. Measure-first /
decline-*or-promote*-on-data, per the project rule.

## What this resolves

The maintainer's third fork was: GPU/vDSP stay float-only, **EXCEPT investigate GPU Q15
double-packing** (Metal `half` / CUDA `__half2`), *"measure-first; promote only on a proven
throughput win at an acceptable precision floor… GPU is the one float framework where 16-bit
packing might genuinely pay — so it gets a real measurement, not an assumption."*

This is that measurement. **The instinct is vindicated:** on this M1 Pro, packing two 16-bit
lanes per register for the oscillator sin is a real **~2–3× throughput win** — the opposite
of the CPU Q15 result (QTYPE Thread-B2 declined: half/Q15 was *slower* on NEON).

## Why the GPU is different from the CPU (the mechanism)

The shipping GPU oscillator uses the **hardware `sin()`** (per the codegen policy comment in
`spectral_osc_metal_generated.h:33`: *"GPU policy: native sin, NOT the CPU deg-9 minimax
poly"*). On Apple-family GPUs, the **fp16 `sin` runs ~2.5× faster than fp32 `sin`** (measured
below) — a genuine half-rate transcendental advantage that x86/NEON does **not** have. The CPU
Q15/half experiment (B2) lost precisely because NEON has no fp16 transcendental speedup and the
float-widen for accumulation dominates. On the GPU both the SFU advantage *and* the
two-samples-per-thread packing pay off.

## Constraint that fixes the design (the GPU Q-island)

Same island as the CPU path: **phase must stay fp32** (`segment_phase_at` accumulates
`phase0 + j·(α + β·j)` over a whole segment — fp16 phase detunes/aliases catastrophically) and
**accumulation must stay fp32** (`sum +=` over many partials needs fp32 range). So only the
innermost **waveform sin narrows to half**. The faithful "double-pack" is therefore **two
output samples per thread** computed as a `half2` sin, fp32 phase in, fp32 accumulate out —
exactly the int-phase→Q15-eval→widen island, on the GPU.

## Method

- Harness: `tests/core_contracts/bench_metal_q15pack.m` (host/Apple/Metal only, **not** in
  CMake — links Metal/Foundation, Apple-exclusive). It JIT-compiles MSL via
  `newLibraryWithSource` (exactly as production `spectral_synth_metal.m` does; the offline
  `metal` tool is absent on CLT-only Macs but the runtime compiler in Metal.framework is
  present). GPU time is `cb.GPUEndTime − cb.GPUStartTime`, min of 30 reps. Reproduce:

  ```
  clang -O3 -fobjc-arc -framework Metal -framework Foundation \
    tests/core_contracts/bench_metal_q15pack.m -o /tmp/bench_metal_q15pack \
    && /tmp/bench_metal_q15pack
  ```
  (Run outside a GPU-blocked sandbox — `MTLCopyAllDevices` must return a device.)

- **(A) pure sin throughput** — `K` sins/thread, fp32 vs fp16 vs half2 (isolates the SFU).
- **(B) faithful synth inner loop** — `S` segments/sample, fp32 phase + fp32 accumulate, only
  the oscillator sin narrowed; `synth_f32` vs `synth_f16x2` (2 samples/thread), plus the
  precision of the f16x2 output vs the fp32 output. Output N = 1,048,576 samples.

## Results (Apple M1 Pro, Metal 3, median of 3; stable across runs)

**(A) pure sin throughput** — speedup vs `sin_f32`:

| kernel      | K=64  | K=256 |
|-------------|------:|------:|
| sin_f32     | 1.00× | 1.00× |
| sin_f16     | 2.47× | 2.49× |
| sin_f16x2   | **3.01×** | **3.04×** |

**(B) faithful synth** (fp32 phase + accumulate, osc-sin narrowed) — speedup + precision:

| kernel        | S=64  | S=256 | RMS divergence vs fp32 |
|---------------|------:|------:|-----------------------:|
| synth_f32     | 1.00× | 1.00× | (reference)            |
| synth_f16x2   | **2.02×** | **2.71×** | **~−67.4 dBFS** (max\|diff\| ~4e-3) |

(The synth win **grows with segment density** — more sins/sample ⇒ the half-sin advantage
dominates a larger share of the inner loop.)

## Findings

1. **Throughput is a proven win, ~2–3×.** fp16 `sin` is ~2.5× the fp32 rate on M1 Pro; the
   half2 double-pack adds packing/overhead savings to reach ~3.0× pure and **2.0–2.7×** on the
   faithful synth inner loop (fp32 phase + accumulate retained). This is the single Phase-4/5
   place where 16-bit packing pays — and it is exactly the place the maintainer predicted.

2. **Precision floor is ~−67 dBFS** (≈11-bit effective), set by fp16's 10-bit mantissa near
   full-scale. That is **worse than the CPU Q15 realized floor (~−84 dBFS)** — fp16 is less
   precise than Q15 for values near ±1 — so this is a quality-tradeoff mode, not a free win.

3. **The realistic synth win is ≤ the microbench.** The shipping `synthesize_tile_parallel`
   adds threadgroup tiling, barriers, and per-segment range-check divergence (real segments are
   sparse ⇒ fewer sins/sample) that half-sin does **not** accelerate; this microbench is the
   ALU/SFU-bound ceiling (all segments active, no tiling). A real-kernel measurement is needed
   to pin the in-situ number — expect it below 2–2.7× but still positive.

## Recommendation

- **GO-candidate (surfaced, NOT wired): a half2 `synth_f16x2` GPU oscillator mode.** Unlike the
  Phase-4 vDSP declines and the CPU Q15 decline, the throughput win here is real and large. But
  it is **not autonomous** and should land as an **opt-in quality mode** (analogous to `--q15`),
  because:
    1. it moves GPU render precision to ~−67 dBFS — the GPU path is **fp32-exact today**; this
       is **not ≤1 ULP**, so it does *not* qualify for auto-default under the
       faster-path-should-default rule — it is a quality-floor decision only the maintainer can
       make;
    2. north star says *"GPU/vDSP float-only unless a measured win flips it"* — we have a
       measured win, but it carries a precision cost, so the default stays float and the half2
       kernel is an explicit opt-in, not a silent flip;
    3. wiring it means a half2 variant of `synthesize_tile_parallel`, a dispatch/quality flag,
       a GPU parity CTest budgeting ~−67 dBFS, and an **on-the-real-kernel** re-measurement
       (the microbench is the ceiling). All maintainer decisions.
  If the maintainer accepts a ~−67 dBFS GPU quality mode, this is worth building: it is the one
  proven 16-bit-packing throughput win in the whole audit.

- **Matrix note:** if promoted, the backend × domain matrix in `oscillator_dispatch.h` gains a
  **GPU-Metal | Q15/half** cell (opt-in), the first non-float GPU cell — the documentation must
  be updated per the Phase-0 rule. Until then, GPU stays documented float-only.

## Scope

Phase 5 is **measure + recommend only**, per the maintainer fork (investigate; promote only on
proven win at acceptable precision) and the measure-first rule. No production source changed;
the default GPU render is fp32-exact and byte-identical by construction. The half2 GPU mode is
a **GO-candidate surfaced as a maintainer decision** (opt-in, ~−67 dBFS floor), deliberately
left unwired. This closes the proposed Phase 0→5 order; the remaining open cells (CMSIS-Q15
wiring, GPU-Metal/CMSIS *runtime* parity, LUT-scale convergence) are hardware- or
maintainer-gated, not autonomous gaps.
