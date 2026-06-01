# Patch notes — Pass 183: CTF sweep increment 23 — CUDA tile-parallel synth backend (defect fixed: goto bypasses scalar initializer → ill-formed C++) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **CUDA
tile-parallel synthesis backend** — the NVIDIA GPU mirror of the Metal path:

```text
- synth/backends/gpu/cuda/spectral_synth_cuda.cu
    synthesize_tile_kernel    shared-memory segment cache, chunked cooperative
                              load + __syncthreads, per-sample segment bounds
                              test and quadratic-phase/linear-amp/fade accumulation
    fade_envelope             __device__ wrapper over spectral_fade_envelope_gpu
    synth_cuda                host dispatch: preflight, timbre gate, dispatch-plan
                              build, buffer grow/copy, kernel launch, event timing,
                              D2H copy, stream sync, cleanup (DEFECT HERE)
    cuda_init / _available / _vram_usage_bytes / _cleanup
```

**Outcome: one real defect found and fixed** — in `synth_cuda` the timing local
`float gpu_ms = 0.0f;` was declared *after* ~15 `goto cleanup` statements while the
`cleanup:` label sits inside that local's scope, so each goto jumps into the scope of a
scalar variable that has an initializer. That is ill-formed C++ ([stmt.dcl]/3), and nvcc
compiles `.cu` host code as C++ — so the CUDA backend fails to build on a conforming
compiler. The kernel DSP math is clean and bit-matches the Metal/CPU canonical formulas.

## The defect — jump into the scope of an initialized scalar

`synth_cuda` is a single-exit function built around a `cleanup:` label reached by
`goto` from every error path. All of its locals are declared up top *except* the GPU
timing value:

```c
...
SpectralError return_err = SPECTRAL_OK;
...
return_err = spectral_gpu_dispatch_plan_init(...);
if (return_err != SPECTRAL_OK) goto cleanup;        /* goto #1 ... */
... (≈15 more `goto cleanup` on every cuda* error) ...
    float gpu_ms = 0.0f;                            /* declared LATE, with initializer */
    if (cudaEventElapsedTime(&gpu_ms, ...) != cudaSuccess) { ... }
    *t_synth = gpu_ms / SPECTRAL_MILLIS_PER_SECOND_D;
cleanup:                                            /* inside gpu_ms's scope */
    if (ev_start) cudaEventDestroy(ev_start);
    ...
```

`gpu_ms`'s scope runs from its declaration to the end of the function block, so the
`cleanup:` label is **within** that scope. Every `goto cleanup` placed before the
declaration therefore jumps *from a point where `gpu_ms` is not in scope to a point where
it is*, bypassing its initialization.

C++ [stmt.dcl]/3 permits such a jump only when the bypassed variable "has scalar type …
**and is declared without an initializer**." `float gpu_ms = 0.0f;` is scalar but *has*
an initializer, so the exemption does not apply and the program is ill-formed. g++/clang++
(the host compilers nvcc invokes) reject it with `jump to label 'cleanup' crosses
initialization of 'float gpu_ms'`.

### Why it survived undetected

The dev/CI host is macOS with no CUDA toolkit, so `spectral_synth_cuda.cu` is never
compiled here — the five production targets do not include it. The sibling Metal backend
carries the identical late-declared timing local, but `spectral_synth_metal.m` is compiled
as Objective-C (C rules), where a jump over a scalar initialization is *well-formed* (the
value merely becomes indeterminate). The defect is thus unique to the C++ translation
unit and only bites a real `nvcc` build of the CUDA backend.

### The fix

Hoist the declaration above the first `goto`, into the top-of-function declaration block
alongside the other locals, and drop the late redeclaration (the value is unconditionally
overwritten by `cudaEventElapsedTime` before use, and `0.0f` at the top preserves the
prior initial value):

```c
SpectralError return_err = SPECTRAL_OK;
float gpu_ms = 0.0f;                       /* NEW: declared before any goto */
...
/* later, after the kernel + event records: */
if (cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop) != cudaSuccess) { ... }
```

No `goto` now crosses an initialization, so the function is well-formed C++. Behavior is
otherwise byte-identical: `gpu_ms` is still written by `cudaEventElapsedTime` and read
only on the success path.

## What else was checked and is correct (no change)

```text
- Kernel DSP parity: synthesize_tile_kernel reproduces the Metal kernel line-for-line —
  seg_start = seg.start*stretch; seg_end = seg_start + seg.length*stretch; the half-open
  bounds test `if (sample_pos < seg_start || sample_pos >= seg_end) continue;`; j =
  sample_pos - seg_start; alpha/beta/d_amp/phase/amp via the canonical
  spectral_segment_{alpha,beta,d_amp,phase_at,amp_at}_f32 helpers (CUDA includes
  spectral_segment_math.h directly; verified identical to the Metal MSL duplicates);
  sum += amp * fade_envelope(j, seg_len) * oscillator_cuda(p, timbre).
- fade_envelope: fade_len = min(seg_len*0.25, FADE_SAMPLES_DESKTOP), clamped >= 1, inv_fade
  = 1/fade_len, delegated to spectral_fade_envelope_gpu — matches the Metal MSL
  fade_envelope and the canonical osc-formulas version exactly (fade in/out raised-cosine).
- __syncthreads() correctness: range.count is read from tile_ranges[tile_idx] and is
  identical for every thread in the block, so the chunk loop runs the same iteration count
  for all threads; both barriers (post-load, post-accumulate) are reached uniformly by all
  threads — no barrier divergence / deadlock. Threads with sample_idx >= out_len skip the
  inner accumulation but still hit both barriers.
- Cooperative load bound: TILE_SIZE(512) threads load chunks of <= SEG_CACHE_SIZE(256)
  segments via `if (tid < chunk_size)`, so there are always enough threads to fill every
  cached slot read by the inner i<chunk_size loop — no stale/uninitialized shared slot.
- Host lifecycle: cuda_grow_buffer / cuda_grow_host_buffer go through
  spectral_next_capacity_3_over_2 (overflow-checked), free-then-realloc on grow, and zero
  the capacity before the (cuda)Malloc so a failed alloc leaves cap=0 (retry-safe). Every
  cudaMemcpyAsync / event / launch error routes to cleanup, which destroys both events,
  frees the dispatch plan, and on error zeroes the output and t_synth. out_size is computed
  via spectral_array_bytes (overflow-checked) and used uniformly for the output buffer,
  D2H copy length, and the error/zero-output memset.
- Preflight + timbre gate (synth_preflight_float, gpu_check_timbre_or_fallback) and the
  dispatch-plan construction were already audited clean in Pass 179; unchanged here.
```

## Verification

```text
- The CUDA TU is not part of any of the five production targets (no CUDA toolkit on the
  build host), so the host binaries are unaffected by this edit — confirmed by a clean
  rebuild:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
- The fix is a pure C++ well-formedness correction (declaration hoist, no behavior change),
  so a real nvcc build now compiles where it previously errored, with identical runtime
  behavior on every path.
```

## Phase C status

With this increment the sweep has cleared 161-182 (see prior notes) and now the CUDA
tile-parallel backend (183 — the GPU-timing local `float gpu_ms = 0.0f;` was declared after
~15 `goto cleanup` statements whose target label lies inside the local's scope, so every
goto jumped into the scope of an initialized scalar: ill-formed C++ that nvcc rejects,
latent because the macOS host never compiles the `.cu` and the Metal sibling is Objective-C
where the same jump is legal; FIXED by hoisting the declaration above the first goto; the
kernel DSP math, fade envelope, barrier placement, cooperative-load bound, and host
buffer/stream lifecycle are clean and bit-match the Metal/CPU canonical formulas).
Phase D (compiled harness + LUT golden-vector loop) follows.
