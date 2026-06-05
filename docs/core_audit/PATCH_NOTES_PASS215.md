# Patch notes — Pass 215: expose the desktop Q15 compute domain via `--q15` (make the opt-in kernel reachable)

## Scope

The Q15 compute domain (scalar Q15 + the packed 8×Q15 SIMD kernel landed Q5c/Bv,
PASS213–214) was **compiled into every desktop binary but unreachable from the shipped
CLI**: the dispatch is gated by a runtime mask (`g_osc_q15_enable`, default `0`) that only
the test/bench harnesses set — no CLI argument wired `osc_set_q15_enable`. So the
double-lane-packed kernel that was the whole point of Q15 on desktop could not actually be
*run* by a user. This pass adds the `--q15` flag that flips the mask, nothing more: no new
kernel, no precision change, no default-path movement.

**Float stays the desktop default.** `--q15` is strictly opt-in; absent the flag the mask
stays `0` and the render is byte-identical to before (the Q15 sources were already linked,
so this adds no code to the default path either).

## What changed

### `cmd/cli/spectral_cli.h` — one opts field

`SpectralCliOptions` gains `int enable_q15;` (default `0` = float compute domain), beside the
existing `osc_force_scalar` / `osc_quality` oscillator-strategy fields.

### `cmd/cli/spectral_cli.c` — parse + init + help

- `spectral_cli_init`: `opts->enable_q15 = 0;`.
- Parse loop: `--q15` is a boolean flag (same shape as `--cache` / `--simd`), sets
  `enable_q15 = 1` and consumes the token.
- Desktop usage block (`#else`): a help line documenting `--q15` — packed 8-wide SIMD Q15 on
  saw/square/triangle/parabola, scalar Q15 under `--scalar`, forces the CPU backend.

### `cmd/cli/spectral_cli_pipeline.c` — wire it in `run_synthesis`

After the dispatch is set (and mirroring how non-naive `--quality` forces CPU), when
`opts->enable_q15`:

- if `osc_q15_available(opts->timbre)`: call `osc_set_q15_enable(OSC_Q15_BIT(opts->timbre))`
  and set `req_backend = BACKEND_CPU`. A render uses a single timbre, so only that timbre's
  mask bit matters. The log states whether the route is the **packed 8-wide SIMD Q15** path
  (`!osc_force_scalar && osc_simd_q15_available(timbre)`, guarded `#if defined(OSC_SIMD_GENERIC)`)
  or **scalar Q15**.
- else (asin/quantized/pwm — no Q15 path): log "requested but unavailable … rendering float"
  and leave the mask `0`.

The CPU-force is necessary because the Q15 kernels live on the CPU float-synth dispatch; the
Metal/CUDA backends have their own kernels and ignore the mask, so without it `--q15` under an
AUTO backend on Apple Silicon would silently pick Metal and do nothing.

## Verification

```text
- desktop (arm64_metal) rebuilds clean (only the pre-existing, unrelated
  spectral_accum_batch4 -Wunused-function in spectral_synth_arm32.c, an
  already-modified working-tree file — not touched by this pass).
- ctest 11/11 PASSED (unchanged: the kernel and its parity/precision gates are
  what this pass exposes, not what it modifies).
- Flag routing, by render log:
    saw   --q15            -> "ENABLED for saw (packed 8-wide SIMD Q15; forcing CPU backend)", backend used = CPU
    saw   --q15 --scalar   -> "ENABLED for saw (scalar Q15; forcing CPU backend)"
    sine  --q15            -> "ENABLED for sine (scalar Q15; ...)"   (no SIMD Q15 path for sine)
    asin  --q15            -> "requested but unavailable for asin; rendering float"
  Renders complete, output written.
- Default byte-identity: without --q15 the mask stays 0; the Q15 sources were already
  linked, so no default-path bytes move.
```

## Status

The desktop Q15 compute domain is now user-reachable: `--q15` routes the selected algebraic
timbre through the packed 8×Q15 SIMD kernel (~1.4–1.6× over the float-SIMD oscillator,
PASS214 numbers), `--q15 --scalar` through the scalar Q15 reference, and forces the CPU
backend so the choice actually takes effect. Float remains the desktop default; the flag is
pure opt-in plumbing over the already-CI-locked kernel.

**Deferred (unchanged from PASS214):** the sine-pack8 SIMD-Q15 routing re-validation and the
~8 KB `g_osc_q15_sine_lut` `#if !SPECTRAL_EMBEDDED` footprint guard — both independent of this
CLI plumbing.
