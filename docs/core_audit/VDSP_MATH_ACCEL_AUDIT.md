# vDSP / Accelerate math-acceleration audit (Oscillator-Backend-Contract Phase 4)

**Status:** AUDIT COMPLETE — recommendation table below. **No production code is wired.**
Promotion of any candidate is a maintainer decision (it moves default desktop numeric
output and adds an Accelerate dependency to the core host vector path). Measure-first /
decline-on-data, per the project rule.

## What this resolves

The maintainer's first fork on this initiative was: **no vDSP *oscillator*** — instead
*"audit for any math vDSP/Accelerate can accelerate."* This is that audit. It measures
every production host vector op in
`spectral_engine/core/port/host/spectral_vector_ops.c` against its closest
vDSP / vForce equivalent, on this Apple-Silicon machine, under the exact production host
flags, and reports speedup + numeric divergence so any loss of bit-identity is explicit.

## Method

- Harness: `tests/core_contracts/bench_vdsp_audit.c` (host/Apple only, **not** wired into
  CMake — it links `-framework Accelerate`, which the production build does not). Compile +
  run command is in the file header; reproduce with:

  ```
  clang -O3 -ffast-math -ffp-contract=fast -march=native -std=c11 \
    -Ispectral_engine/core -Ispectral_engine/core/port/host -Ithird_party/simde \
    tests/core_contracts/bench_vdsp_audit.c \
    spectral_engine/core/port/host/spectral_vector_ops.c \
    -framework Accelerate -lm -o /tmp/bench_vdsp_audit && /tmp/bench_vdsp_audit
  ```

- It links the **real** `spectral_vector_ops.c` (not a re-implementation) compiled with the
  production host flags `-O3 -ffast-math -ffp-contract=fast -march=native`, so the SIMDe
  column is exactly what ships.
- Sizes `{256, 513, 2049, 4097, 65536}`. The 513 / 2049 / 4097 cases are the realistic
  STFT `n_freqs` (= `n_fft/2 + 1`) sizes; 65536 is a large-merge stress size.
- `speedup = SIMDe_ns / vDSP_ns` (>1 ⇒ vDSP faster). Divergence is max|diff| + RMS vs the
  SIMDe result. Numbers below are the median of 3 runs; they were stable across runs.
- **Production atan2 is EXACT**: `SPECTRAL_ENABLE_APPROX_ATAN2 = 0` (spectral_config.h:55),
  so the shipping phase path is the scalar `atan2f` loop, **not** the SIMDe polynomial. The
  honest comparison is therefore scalar `atan2f` vs vForce `vvatan2f` (full-precision
  vectorized atan2) — which is what the `atan2` / `magsq_phase` rows measure.

## Results (median of 3, ns/elem)

| op            |  n=256 |  n=513 | n=2049 | n=4097 | n=65536 | max\|diff\| | verdict |
|---------------|-------:|-------:|-------:|-------:|--------:|-----------:|---------|
| vmul          | 1.5x   | 0.21x  | 0.62x  | 1.00x  | 3.1x    | 0 (exact)  | DECLINE |
| vadd          | 1.4x   | 0.21x  | 0.60x  | 0.94x  | 2.8x    | 0 (exact)  | DECLINE |
| vsq           | 1.5x   | 0.19x  | 0.57x  | 0.93x  | 2.6x    | 0 (exact)  | DECLINE |
| vsmul         | 1.5x   | 0.17x  | 0.63x  | 0.91x  | 2.4x    | 0 (exact)  | DECLINE |
| vmax          | 2.7x   | 3.9x   | 5.2x   | 5.4x   | 3.4x    | 0 (exact)  | marginal |
| vmaxmgv       | 2.1x   | 2.7x   | 3.5x   | 3.9x   | 3.3x    | 0 (exact)  | marginal |
| **atan2**     | **3.4x** | **3.3x** | **3.8x** | **6.3x** | **11.5x** | 2.4e-7 (~1 ULP) | **PROMOTE-candidate** |
| magsq_split   | 1.4x   | 1.7x   | 1.5x   | 1.5x   | 1.3x    | 1.2e-7     | marginal (no caller) |
| magsq_only    | 0.90x  | 0.94x  | 0.84x  | 0.81x  | 0.27x   | 1.2e-7     | DECLINE (SIMDe wins) |
| **magsq_phase** | **2.8x** | **2.7x** | **3.2x** | **4.7x** | **6.6x** | 2.4e-7 (phase, ~1 ULP) | **PROMOTE-candidate** |

(`magsq_phase` row reports the *phase* divergence — the magnitude half matches to ~1e-6.
Absolute SIMDe cost: elementwise ops ≈ 0.08–0.16 ns/elem; atan2 ≈ 3.5–11.8 ns/elem.)

## Findings, grounded in the real callers

1. **The one genuine, high-value win is atan2 / phase extraction.** vForce `vvatan2f` is
   **3.3×–11.5×** faster than the exact scalar `atan2f` loop, consistently, growing with n,
   at **~1 ULP** divergence (max 2.4e-7 on a value range of ±π; RMS ~1e-7). This is not
   bandwidth-bound noise — atan2 is a genuine transcendental, the only compute-bound op in
   the file. `spectral_magsq_phase` inherits the win (2.8×–6.6×, the residual being the
   magsq + ctoz overhead) and **is the per-frame STFT phase path**:
   `analysis/spectral_analysis_fft.c:365` calls it once per FFT frame at `n_freqs`. For a
   large analysis this runs millions of times. This is the candidate worth a maintainer's
   attention.

2. **`magsq_only` correctly stays SIMDe.** The vDSP route for the interleaved no-phase case
   needs `ctoz` (deinterleave) + `zvmags` + `maxv` — three passes — and **loses** to the
   fused single-pass SIMDe kernel (0.27×–0.94×). It is the sibling no-phase per-frame path
   at `spectral_analysis_fft.c:363`; the existing SIMDe choice there is right. **DECLINE.**

3. **Simple elementwise ops (vmul/vadd/vsq/vsmul) decline.** They are memory-bandwidth-bound
   at ~0.1 ns/elem; the "win" is entirely size-dependent and at the noise floor. Critically
   they **lose at exactly the sizes that ship** (0.17×–0.63× at n=513/2049, the real STFT
   widths) and only "win" at n=65536 where absolute cost is already negligible. `vmul` is the
   window-apply at `spectral_analysis_fft.c:358` sized `n_fft` (1024/4096) — squarely in the
   lose-or-tie band. The per-call vDSP dispatch overhead dominates at small n. **DECLINE.**

4. **Reductions (vmax/vmaxmgv) are a consistent but absolutely tiny win.** vDSP is 2×–5×
   faster and bit-identical, but the op costs ~0.1 ns/elem and `vmaxmgv` runs once per output
   buffer normalize (`core/port/host/spectral_out_kernels.c:23`) — not a hot loop. Real win,
   negligible payoff. **Marginal; optional; low priority.**

5. **`magsq_split` (vDSP_zvmags) is a modest 1.3×–1.7×** at ~1e-7 divergence, but has **no
   production caller** (grep finds none outside the bench). Lowest priority. **Marginal.**

## Recommendation

- **Promote-candidate (surfaced, NOT wired): vForce `vvatan2f` for the phase path** —
  `spectral_vatan2` and the atan2 half of `spectral_magsq_phase`. It is the single
  high-value acceleration in the file: a hot, per-frame, compute-bound transcendental, 3–11×
  faster at ~1 ULP. This is exactly the "≤1 ULP and faster ⇒ should be default" shape — but
  it is **not autonomous** because:
    1. it moves **default desktop** analysis output by ~1 ULP (max 2.4e-7) — i.e. **not
       byte-identical**, which crosses the project's default-desktop-byte-identical north
       star and must be a maintainer call;
    2. it adds an **Accelerate dependency** to the core host vector path (today only the FFT
       path links Accelerate, not `spectral_vector_ops.c`);
    3. it is **host/Apple-only** — the embedded and any non-Accelerate host build still need
       the scalar `atan2f` fallback, so promotion means a `#if`-guarded vDSP path with the
       scalar loop retained, not a replacement.
  If the maintainer accepts a ~1-ULP shift on desktop phase, this is the wiring to do, behind
  a host-Accelerate guard, with the scalar loop kept for every other target and a parity
  CTest budgeting the ~1-ULP divergence.

- **Decline (on data):** vmul, vadd, vsq, vsmul (bandwidth-bound; lose at shipping sizes),
  `magsq_only` (fused SIMDe beats the 3-pass vDSP route).

- **Marginal / optional, maintainer's discretion:** vmax, vmaxmgv (consistent but negligible
  absolute win, bit-identical), `magsq_split` (modest, no caller).

## Scope

Phase 4 is **measure + recommend only**, per the maintainer fork (no vDSP oscillator; audit
math instead) and the measure-first / decline-on-data rule. No production source changed; the
default desktop render is byte-identical by construction. The promotion of `vvatan2f` is
surfaced as a maintainer decision and deliberately left unwired. Phase 5 (GPU Q15
double-pack, measure-first) is independent and unstarted.
