# Patch notes — Pass 195: Optimisation track O4-B — restrict hints on the CPU reduce path (bit-identical)

## Scope

Second implementation pass of the **optimisation track** (`docs/core_audit/OPTIMISATION_PLAN.md`).
Implements **O4-B** (Tier 4): "restrict / #pragma omp simd on host CPU loops … to unblock
auto-vectorisation. Quick, low-risk. ≤1 ULP."

The actual surface in this tree is smaller than the plan's generic description, because the
host vector kernels are **already hand-vectorised** and the synth inner loops are
**indirect-call-bound**. The one genuine, bit-identical win is a `restrict` qualifier on the
reduce destination. See "Findings" for why the rest is N/A here (not skipped arbitrarily).

## Change

```text
- synth/backends/cpu/spectral_synth_cpu.c
    thread_buffers_reduce_float  : out_buffer  -> float* __restrict__ out_buffer
    thread_buffers_reduce_native : out_buffer  -> spectral_sample_t* __restrict__ out_buffer
```

`out_buffer` is the final output allocation; the per-thread partial sums live in a separate
arena (`tb->arena`, distinct malloc). They provably never alias. The qualifier removes the
only aliasing barrier the compiler had against vectorising the per-`j` reduce map
(`out[j] = bufs[0][j] (+) bufs[1][j] (+) …`): the store to `out_buffer[j]` could previously
be assumed to alias a later `bufs[t][j']` read. `__restrict__` matches the spelling already
used in this codebase (e.g. `timbre_synth_segment(float* __restrict__ dst, …)`).

**Bit-identical:** `restrict` is a pure aliasing assertion — it enables optimisation but
cannot change results when the no-alias claim holds (it does). The per-element accumulation
order (`bufs[0]+bufs[1]+…`, left-to-right) is untouched, so even the saturating-`Q15`
`SPECTRAL_SAMPLE_ADD` native path is unaffected. No golden change.

## Findings — why the `#pragma omp simd` half is N/A in this tree

```text
- Host vector primitives are ALREADY explicit SIMD (SIMDe intrinsics, AVX2+SSE2+scalar
  tail) in core/port/host/spectral_vector_ops.c — spectral_vadd (used by the float reduce)
  is fully vectorised. There is no auto-vectorisation to "unblock" there, and a `restrict`
  on spectral_vadd would be WRONG: the float reduce calls it in-place
  (spectral_vadd(out, bufs[t], out, len) — a == dst aliased by design).
- The synth per-segment inner loops (oscillator.c synth_segment_scalar; the CPU
  segment_fn_* callbacks) call through a function pointer (osc_fn / timbre_oscillator) or a
  wavetable lookup. An indirect call blocks auto-vectorisation regardless of `#pragma omp
  simd`; the fix is devirtualisation, which is the separately-gated O2-A (default vectorised
  sine), a behaviour change requiring its own signed-off golden — out of O4-B scope.
- The native reduce's inner t-reduction is over a possibly-saturating op (Q15) and must NOT
  be reordered; only the OUTER per-j map is vectorisable, which the `restrict` now permits
  for the compiler without changing accumulation order.
```

## Verification

```text
- Five production targets build clean (desktop, simulate, simulate_daisy, embedded_arm,
  embedded_arm_float) — only the pre-existing benign -mavx2 / -mno-avx512f notes.
- ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift).
- End-to-end (CPU backend, sin_440hz.wav, 4096/128): output byte-identical (cmp) to
  build/golden/cpu_sine_ref.wav — the pre-optimisation reference.
```

## Status

O4-B implemented and verified bit-identical; default output unchanged (no golden change).
Per exec order, next is **O1-B** (CPU output tiling replacing the private-buffer reduce) —
note that O1-B reworks this same reduce path and carries a ≤1 ULP FP-reorder, so it will
need its own signed-off golden, unlike this pass.
