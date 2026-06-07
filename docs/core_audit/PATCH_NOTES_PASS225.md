# Patch notes — Pass 225: vDSP / Accelerate math-acceleration audit (Oscillator-Backend-Contract Phase 4)

## Problem

The maintainer's first fork on the oscillator-backend-contract initiative was
explicit: **no vDSP oscillator** — instead *"audit for any math vDSP/Accelerate
can accelerate"*, measure-first. Phase 4 is that audit. The question is narrow and
data-driven: of the production host vector ops in
`spectral_engine/core/port/host/spectral_vector_ops.c`, which (if any) does
Apple's vDSP / vForce beat by enough to justify promotion — and at what cost to
bit-identity and to the build's dependency surface?

## Change

```text
NO production source changed. This pass adds an audit + its reproduction harness:

  docs/core_audit/VDSP_MATH_ACCEL_AUDIT.md            (NEW)
    - Method, full results table (median of 3 stable runs), per-op verdict, and a
      promotion recommendation framed as a maintainer decision.

  tests/core_contracts/bench_vdsp_audit.c             (NEW, host/Apple only)
    - Links the REAL spectral_vector_ops.c with the production host flags
      (-O3 -ffast-math -ffp-contract=fast -march=native) and times each op vs its
      vDSP/vForce equivalent over n = {256,513,2049,4097,65536}, reporting ns/elem,
      speedup, and max|diff|+RMS so any bit-identity loss is explicit.
    - DELIBERATELY NOT wired into CMake: it links -framework Accelerate (which the
      production host build does not) and is Apple-exclusive. Build/run by hand via
      the command in its header comment. Explicit-path bench targets + the
      spectral_engine-rooted contract scans mean this file is swept by nothing.
```

## Finding

```text
Production atan2 is EXACT (SPECTRAL_ENABLE_APPROX_ATAN2 = 0, spectral_config.h:55),
so the honest comparison for phase is scalar atan2f vs vForce vvatan2f.

speedup = SIMDe_ns / vDSP_ns  (>1 => vDSP faster);  median of 3, stable across runs:

  op            n=513  n=2049  n=65536   max|diff|        verdict
  vmul          0.21x  0.62x   3.1x      0 (exact)        DECLINE  (bandwidth-bound)
  vadd          0.21x  0.60x   2.8x      0                DECLINE
  vsq           0.19x  0.57x   2.6x      0                DECLINE
  vsmul         0.17x  0.63x   2.4x      0                DECLINE
  vmax          3.9x   5.2x    3.4x      0                marginal (tiny absolute)
  vmaxmgv       2.7x   3.5x    3.3x      0                marginal
  atan2         3.3x   3.8x    11.5x     2.4e-7 (~1 ULP)  PROMOTE-candidate
  magsq_split   1.7x   1.5x    1.3x      1.2e-7           marginal (no caller)
  magsq_only    0.94x  0.84x   0.27x     1.2e-7           DECLINE  (fused SIMDe wins)
  magsq_phase   2.7x   3.2x    6.6x      2.4e-7 (phase)   PROMOTE-candidate

1. The ONE genuine high-value win is atan2 / phase extraction. vForce vvatan2f is
   3.3x-11.5x faster than the exact scalar atan2f loop at ~1 ULP. It is the only
   compute-bound op in the file; everything else is memory-bandwidth-bound. The win
   flows into spectral_magsq_phase (2.8x-6.6x), which IS the per-frame STFT phase
   path: analysis/spectral_analysis_fft.c:365 calls it once per FFT frame at n_freqs
   (millions of times on a large analysis).

2. magsq_only correctly stays SIMDe. The vDSP route (ctoz + zvmags + maxv, 3 passes)
   LOSES to the fused single-pass SIMDe kernel. It is the no-phase sibling at
   spectral_analysis_fft.c:363; the existing choice there is right.

3. Elementwise vmul/vadd/vsq/vsmul decline: bandwidth-bound, and they LOSE at exactly
   the shipping STFT sizes (n=513/2049). vmul is the window-apply at
   spectral_analysis_fft.c:358 — in the lose band. They only "win" at n=65536 where the
   absolute op cost is already negligible; per-call vDSP overhead dominates small n.

4. Reductions (vmax/vmaxmgv): real, consistent vDSP win, bit-identical, but absolutely
   tiny (~0.1 ns/elem) and not on a hot loop. Marginal/optional.
```

## Recommendation (surfaced, NOT wired — maintainer decision)

```text
PROMOTE-candidate: vForce vvatan2f for the phase path (spectral_vatan2 + the atan2
half of spectral_magsq_phase). It is the "<=1 ULP and much faster => should default"
shape, but it is NOT autonomous because:
  (a) it moves DEFAULT DESKTOP analysis output by ~1 ULP (max 2.4e-7) -- not
      byte-identical, which crosses the default-desktop-byte-identical north star;
  (b) it adds an Accelerate dependency to the core host vector path (today only the
      FFT path links Accelerate, not spectral_vector_ops.c);
  (c) it is host/Apple-only -- embedded + non-Accelerate host builds still need the
      scalar atan2f fallback, so promotion is a #if-guarded vDSP path, not a swap.
  If accepted: wire behind a host-Accelerate guard, keep the scalar loop for all other
  targets, add a parity CTest budgeting the ~1-ULP divergence.

DECLINE (on data): vmul, vadd, vsq, vsmul (bandwidth-bound; lose at shipping sizes),
magsq_only (fused SIMDe beats 3-pass vDSP).

MARGINAL/optional: vmax, vmaxmgv (negligible absolute win, bit-identical),
magsq_split (modest, no caller).
```

## Verification

```text
- Bench compiles + runs clean with the documented command; results stable over 3 runs
  (the headline atan2 row is rock-solid: 3.3x->11.5x, max|diff| 2.384e-7 every run).
- NO production .c/.h changed -> default desktop render byte-identical BY CONSTRUCTION.
- The new bench is not in any CMake target and is swept by no glob (explicit-path bench
  targets; the osc_backend_contract / q_domain_contract scans root at spectral_engine,
  not tests). ctest unaffected (still 13/13 on the existing gates).
```

## Scope (Oscillator-Backend-Contract Phase 4)

Phase 4 is **measure + recommend only**, by maintainer fork (audit math, don't build a
vDSP oscillator) and the measure-first / decline-on-data rule. The audit is complete; the
single high-value candidate (`vvatan2f` for phase) is surfaced as a maintainer decision
and left **unwired** because it shifts default-desktop output ~1 ULP and adds an
Accelerate dependency to the core host path. Phase 5 (GPU Q15 double-pack, measure-first)
is independent and unstarted.
