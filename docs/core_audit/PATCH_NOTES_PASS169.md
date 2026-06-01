# Patch notes — Pass 169: CTF sweep increment 9 — core synth dispatch / internal helpers (Phase C)

## Problem

Phase C is the CTF/KISS adversarial defect sweep: capture every latent defect in
`core/`, `analysis/`, `synth/` and fix it in place. This pass sweeps the **core
synthesis dispatch / shared-helper cluster** — `core/spectral_synth_internal.c`
(shared synthesis helpers, param derivation, the preflight validator, the GPU
tile/segment caches, and the GPU dispatch-plan builder) and
`core/spectral_backend.c` (the vtable-driven backend dispatch with CPU fallback).

The one defect is **dead duplicate code**: a `static` helper that is defined but
never called in any build.

```text
core/spectral_synth_internal.c:84
    static void synth_zero_output_if_valid(void* out_buffer, size_t out_len,
                                           size_t elem_size) { ... memset ... }
```

`synth_zero_output_if_valid` has **zero call sites anywhere** in the tree (`grep`
confirms only the definition). It re-implements an output-zeroing routine
(`spectral_size_mul` overflow guard + `memset`) that the actual preflight validator
`synth_preflight_common` already performs inline — and does so *better*, using the
`preflight_out_bytes` it computed once at the top (also overflow-checked via
`spectral_size_mul`) rather than recomputing the product on every early return.

It surfaced as a `-Wunused-function` warning on the `simulate_daisy`,
`embedded_arm`, and `embedded_arm_float` targets (the host targets dead-strip it
silently). This is the same defect class as pass 164's dead divergent validator: a
never-called duplicate of canonical logic is a maintenance/drift hazard — a future
edit could "fix" the dead copy and believe output is zeroed when it is not.

## Change

```text
1. Removed the dead, never-called output-zeroing helper  (dead duplicate code)
   core/spectral_synth_internal.c  (synth_zero_output_if_valid)
   Deleted the 6-line static function. The behaviour it nominally provided (zero
   the output buffer when the synth produces nothing) is already correctly handled
   inline by synth_preflight_common for every reachable early-return path:
     - elem_size == 0            -> error, no zero (cannot size the buffer)
     - out_len*elem_size overflow-> SPECTRAL_ERR_OVERFLOW, no zero (cannot size it)
     - !out_buffer || out_len==0 -> nothing to zero
     - sa.count == 0             -> memset(out_buffer, 0, preflight_out_bytes)
     - invalid params            -> memset(out_buffer, 0, preflight_out_bytes)
     - sa.count > UINT32_MAX     -> memset(out_buffer, 0, preflight_out_bytes)
     - invalid segment array     -> memset(out_buffer, 0, preflight_out_bytes)
   preflight_out_bytes is computed once (line 131) behind a spectral_size_mul
   overflow guard, so the inline zeroing is both correct and strictly cheaper than
   the deleted helper's per-call recomputation. Removing the helper clears the
   -Wunused-function warning on the embedded/daisy targets with no behavioural
   change on any target.
```

## Why this is correct and behaviourally inert

The deleted function was `static` and had no callers, so it contributed no
reachable code on any target (the compiler already excluded it — that is exactly
why it warned). Removing it cannot change any synthesis result, control flow, or
data layout of a reachable symbol. The authoritative output-zeroing contract lives
in `synth_preflight_common` and is unchanged.

## Finding

Audited and left unchanged (no defect) — the rest of the cluster is solid:
- `synth_derive_param_scalars` / `synth_validate_params` / `make_synth_params` —
  validate the full stretch/pitch domain (finite, positive, `<= SPECTRAL_MAX_STRETCH`,
  pitch in `[MIN,MAX]`) and reject the tiny-stretch `stretch*stretch <= 0` underflow
  before any backend consumes the derived inverse scalars; `make_synth_params` also
  caps `num_segs <= UINT32_MAX`.
- `synth_preflight_common` / `synth_preflight_float` / `synth_preflight_native` —
  every early return either zeroes the (valid, sized) buffer or correctly declines
  to (unsized/overflowing/NULL buffer); the timing dummy is wired before any return.
- `segment_loop_params_init` — a full overflow/finite gauntlet on the
  `start*stretch` / `length*stretch` products (finite, `>= 0`, `< out_len`,
  `<= SIZE_MAX`) before the `size_t` casts; clamps `length` to the remaining buffer
  with an underflow-safe `length > out_len - start_idx` rearrangement; proves the
  final `(float)(length-1)` offset and the derived alpha/beta/d_amp and endpoint
  phase/amp scalars are all finite before the backend loop runs.
- the process-global one-shot GPU tile and segment caches
  (`gpu_tile_cache_*` / `gpu_seg_cache_*`) — thread-local, cleared on every
  read/miss/key-mismatch so a stale pointer can never be reused across calls;
  `gpu_tile_preprocess_cached` re-validates the cached layout words before trusting
  it and falls back to a fresh preprocess otherwise.
- `spectral_gpu_dispatch_plan_init` — guards `segment_bytes` / `tile_ids_bytes` /
  `tile_ranges_bytes` with `spectral_array_bytes`, frees the plan on every error
  exit, and takes the zero-output fast path when `total_refs == 0`.
- `core/spectral_backend.c` — the vtable dispatch is total: unknown/non-compiled
  backends resolve to the `vtable_fallback` and are detected by the `vt->id !=
  backend` guard; `spectral_backend_name` bounds its index by `BACKEND_EXPORT`;
  CPU is always routed through `dispatch_cpu_fallback` (which forwards the real
  `n_threads`), and AUTO/EXPORT are resolved to a concrete backend before dispatch;
  GPU+wavetable, unavailable backend, unsupported timbre, and backend-synth-failure
  all fall back to CPU with a logged resolution. `n_threads < 1` is floored to 1.

## Verification

```text
- five production targets build clean: desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float — the previously-flagged
  synth_zero_output_if_valid -Wunused-function note is now GONE on the
  embedded/daisy targets; only the pre-existing benign -mavx2 / -mno-avx512f
  unused-command-line-arg notes remain on host.
- ctest: 4/4 PASSED — arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift.
- functional parity on resources/testing/sin_440hz.wav (n_fft=1024 hop=256
  thresh=-70): desktop and simulate both detect 340 segments and write output,
  unchanged from pass 168. Removing an uncalled static symbol is behaviour-neutral.
```

## Scope (Phase C increment)

Core synth dispatch / shared-helper cluster, one defect fixed: removed the dead,
never-called `synth_zero_output_if_valid` duplicate (the canonical output-zeroing
lives inline in `synth_preflight_common`), clearing the embedded/daisy
`-Wunused-function` warning with no behavioural change. With this increment the
Phase C sweep has cleared fixed-point (161), analysis/peak-track (162),
port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling + alloc/cache
(165), synth-backends + analysis-orchestration (166), CLI/orchestration (167),
embedded fade envelope (168), and core synth dispatch/internal helpers (169). Phase
D (compiled harness + LUT golden-vector loop) follows.
