# Patch notes — Pass 226: GPU Q15 / fp16 double-pack audit (Oscillator-Backend-Contract Phase 5)

## Problem

The maintainer's third fork: GPU/vDSP stay float-only **except investigate GPU Q15
double-packing** (Metal `half` / CUDA `__half2`) — *"measure-first; promote only on a
proven throughput win at an acceptable precision floor… GPU is the one float framework
where 16-bit packing might genuinely pay, so it gets a real measurement, not an
assumption."* Phase 5 is that measurement. The CPU analog (QTYPE Thread-B2) had already
DECLINED half/Q15 on data (slower on NEON + precision regression), so the open question was
whether the GPU's different cost structure flips the verdict.

## Change

```text
NO production source changed. This pass adds an audit + its reproduction harness:

  docs/core_audit/GPU_Q15_DOUBLEPACK_AUDIT.md          (NEW)
    - Mechanism, full results, precision floor, and a GO-candidate recommendation framed
      as a maintainer decision (opt-in quality mode, not a default flip).

  tests/core_contracts/bench_metal_q15pack.m           (NEW, host/Apple/Metal only)
    - JIT-compiles MSL via newLibraryWithSource (as production spectral_synth_metal.m does;
      the offline `metal` tool is absent on CLT-only Macs, the runtime compiler is present).
    - (A) pure sin throughput: sin_f32 vs sin_f16 vs sin_f16x2 (isolates the SFU question).
    - (B) faithful synth inner loop: S segments/sample, fp32 phase + fp32 accumulate, only
      the oscillator sin narrowed to half2 (two output samples/thread) -- synth_f32 vs
      synth_f16x2 + precision (f16x2 output vs fp32 output). GPU-timed via GPUStart/EndTime.
    - DELIBERATELY NOT in CMake: links Metal/Foundation, Apple-exclusive. Build/run by hand
      (command in the file header). Swept by no glob (explicit-path bench targets; contract
      scans root at spectral_engine, not tests).
```

## Finding

```text
Apple M1 Pro, Metal 3; median of 3, stable across runs:

(A) PURE sin throughput (speedup vs sin_f32)
      sin_f16     2.47x-2.49x      -- fp16 hardware sin is ~2.5x the fp32 rate on this GPU
      sin_f16x2   3.01x-3.04x      -- + two-phase packing saves launch/loop overhead

(B) FAITHFUL synth (fp32 phase + fp32 accumulate, only osc-sin narrowed to half2)
      synth_f16x2  2.02x (S=64) -> 2.71x (S=256)   win GROWS with segment density
      precision    ~-67.4 dBFS RMS vs fp32 (max|diff| ~4e-3), every run

1. PROVEN throughput WIN, ~2-3x -- the OPPOSITE of CPU B2 (half/Q15 slower on NEON). The
   mechanism: the GPU oscillator uses the HARDWARE sin (spectral_osc_metal_generated.h:33,
   "GPU policy: native sin"), and Apple-family GPUs run fp16 sin ~2.5x faster than fp32.
   x86/NEON has no such fp16-transcendental advantage -- which is why the CPU lost and the
   GPU wins. The maintainer's instinct was right.

2. The GPU Q-island is identical to the CPU one: phase MUST stay fp32 (segment_phase_at
   accumulates over the segment; fp16 phase detunes), accumulation MUST stay fp32 (sums many
   partials). Only the innermost waveform sin narrows -> the faithful double-pack is two
   output samples/thread as half2, fp32 in, fp32 out.

3. Precision floor ~-67 dBFS (~11-bit; fp16 10-bit mantissa near full scale) is WORSE than
   the CPU Q15 realized floor (~-84 dBFS). So this is a quality-tradeoff mode, not a free win.

4. Realistic synth win is <= the microbench: the real synthesize_tile_parallel adds
   threadgroup tiling, barriers, and per-segment range-check divergence (sparse segments =>
   fewer sins/sample) that half-sin does not accelerate. This microbench is the ALU/SFU
   ceiling; an on-the-real-kernel measurement is needed to pin the in-situ number.
```

## Recommendation (surfaced, NOT wired — maintainer decision)

```text
GO-candidate: a half2 synth_f16x2 GPU oscillator mode, landed as an OPT-IN quality mode
(analogous to --q15), NOT a default flip. It is the one proven 16-bit-packing throughput win
in the entire Phase-4/5 audit. NOT autonomous because:
  (a) it moves GPU render precision to ~-67 dBFS -- the GPU path is fp32-exact today; this is
      NOT <=1 ULP, so it does NOT qualify for auto-default under faster-path-should-default --
      it is a quality-floor call only the maintainer can make;
  (b) north star = "GPU/vDSP float-only unless a measured win flips it": we have a measured
      win, but it carries a precision cost, so the default stays float and half2 is explicit
      opt-in;
  (c) wiring = a half2 variant of synthesize_tile_parallel + a dispatch/quality flag + a GPU
      parity CTest budgeting ~-67 dBFS + an on-the-real-kernel re-measurement.
If promoted, the oscillator_dispatch.h matrix gains a GPU-Metal | Q15/half (opt-in) cell --
the first non-float GPU cell -- and Phase-0 docs must be updated. Until then GPU stays float-only.
```

## Verification

```text
- Bench runs on the M1 Pro GPU (Metal 3) and is stable over 3 runs (synth_f16x2 2.02x-2.71x,
  precision -67.4 dBFS identical each run). GPU-timed, not wall-clock.
- NO production .c/.h/.m changed -> default GPU render fp32-exact / byte-identical BY
  CONSTRUCTION. ctest unaffected (still 13/13). The new bench is in no CMake target and swept
  by no glob.
- Honest scoping: the microbench is the ALU/SFU ceiling (all segments active, no tiling); the
  real-kernel win will be lower but is proven positive.
```

## Scope (Oscillator-Backend-Contract Phase 5 — closes the Phase 0->5 order)

Phase 5 is **measure + recommend only**, by maintainer fork (investigate; promote only on a
proven win at an acceptable precision floor). The measurement is **positive** — a ~2–2.7×
GPU throughput win — making this a **GO-candidate** (distinct from the Phase-4 declines),
surfaced as a maintainer decision: land it **opt-in** behind a quality flag with a ~−67 dBFS
parity budget, not as a default flip. Left unwired. This completes the proposed Phase 0→5
order; remaining open cells (CMSIS-Q15 live wiring, GPU-Metal/CMSIS *runtime* parity,
LUT-scale convergence) are hardware- or maintainer-gated, not autonomous gaps.
