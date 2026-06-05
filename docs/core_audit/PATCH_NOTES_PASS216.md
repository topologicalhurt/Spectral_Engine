# Patch notes — Pass 216: guard the desktop Q15 sine LUT + compute path out of embedded firmware

## Scope

The desktop Q15 compute path in `core/oscillator.c` — the scalar `synth_segment_q15`,
its `osc_q15_eval` selector, the packed-SIMD `osc_simd_q15_segment` dispatch, and the
~8 KB `g_osc_q15_sine_lut` they read — is a **host-only** optimisation. Real embedded
firmware synthesises Q15 in `synth/backends/arm/spectral_synth_arm32.c` (the integer NCO)
and **never reaches this dispatch**. But because `g_osc_q15_sine_lut` is a file-scope
`static` referenced by reachable (runtime-gated, not dead) code, the compiler could not
elide it: it sat in the embedded `.bss` as pure waste, with the segment helpers in `.text`
alongside it.

This pass wraps that host-only body in `#if !SPECTRAL_EMBEDDED`. It is the first of the two
deferred follow-ups recorded at PASS214/PASS215.

**No behavior change anywhere.** On desktop (`SPECTRAL_EMBEDDED == 0`) the guard includes
everything exactly as before — the object code is unchanged. On embedded the removed code
was never executed (the path engages only when `g_osc_q15_enable != 0`, which only the
desktop `--q15` flag and the host harnesses ever set), so nothing that ran before stops
running.

## What changed

### `core/oscillator.c` — four `#if !SPECTRAL_EMBEDDED` regions

1. **LUT storage.** `g_osc_q15_sine_lut[SPECTRAL_OSC_LUT_SIZE + 1]` (4097 × `q15_t` =
   **8194 bytes**) and its `g_osc_q15_sine_lut_ready` flag.
2. **LUT init.** The one-shot `spectral_osc_q15_init_sine_lut(...)` block inside
   `osc_set_q15_enable` (the `g_osc_q15_enable = mask;` assignment stays unconditional).
3. **Segment helpers.** `osc_q15_eval` (the per-sample Q15 waveform selector) and
   `synth_segment_q15` (the scalar Q15 sustain path) — defined and used only by the Q15
   dispatch, so they move under the guard with it (otherwise `synth_segment_q15`, a
   non-inline `static`, would warn `-Wunused-function` on embedded).
4. **Dispatch block.** The `if ((g_osc_q15_enable & OSC_Q15_BIT(timbre)) && ...)` block in
   `timbre_synth_segment`, including its nested `#if defined(OSC_SIMD_GENERIC)` packed-SIMD
   twin (which calls the host-port `osc_simd_q15_segment`, not even linked on embedded).

### What stays in every build (deliberately not guarded)

`osc_q15_available()`, `osc_set_q15_enable()`, and `osc_get_q15_enable()` keep their
definitions in all builds because the CLI pipeline's `run_synthesis`
(`cmd/cli/spectral_cli_pipeline.c`) is compiled in **every** mode — including
`SPECTRAL_RESTRICTED_MODE` — and references them unconditionally for the `--q15` opt-in.
On embedded `osc_set_q15_enable` becomes "set the mask, skip the LUT"; the mask is simply
never consulted (the dispatch that would read it is compiled out). `g_osc_q15_enable`
itself (2 bytes) stays — read by the getter, negligible.

`SPECTRAL_EMBEDDED` is reachable in `oscillator.c` via
`spectral_synth_internal.h → spectral_common.h → spectral_config.h` (where it defaults to
`0` under `#ifndef` and is set to `1` by the build system for embedded targets).

## Verification

```text
Desktop (SPECTRAL_EMBEDDED == 0):
  - desktop (arm64_metal) rebuilds clean; oscillator.c emits no new diagnostic
    (only the pre-existing, unrelated spectral_accum_batch4 -Wunused-function in
    spectral_synth_arm32.c, an already-modified working-tree file).
  - ctest 11/11 PASSED — including q15_simd_parity and q15_production_parity, i.e.
    the desktop Q15 path (LUT + scalar + packed SIMD) is byte-for-byte unchanged.
  - Desktop object code is identical: on the host the guard evaluates to "include
    all", so only comments and no-op preprocessor lines were added.

Embedded (SPECTRAL_EMBEDDED == 1):
  - make simulate            -> oscillator.c compiles clean; the Q15 desktop path and
                                the 8 KB g_osc_q15_sine_lut are gone, no -Wunused-function
                                for synth_segment_q15/osc_q15_eval, no undefined reference
                                to the host-only osc_simd_q15_segment.
  - make simulate_daisy      -> EMBEDDED + RESTRICTED_MODE compiles + links clean: the
                                restricted CLI's run_synthesis still resolves
                                osc_q15_available / osc_set_q15_enable (kept unguarded).

Footprint delta on embedded targets: exactly sizeof(g_osc_q15_sine_lut) = 4097 * 2 =
8194 bytes of .bss removed, plus the synth_segment_q15 / osc_q15_eval text. (The daisy
firmware cross-build itself is hardware/SDK-gated and not built here; the saving is the
size of the removed static, not an estimate.)
```

## Status

Deferred follow-up #1 (the `g_osc_q15_sine_lut` `#if !SPECTRAL_EMBEDDED` footprint guard)
is **LANDED**. The desktop Q15 compute domain is now strictly host-compiled; embedded
firmware carries none of its `.bss`/`.text`. Float remains the desktop default and the
opt-in `--q15` path is unchanged on desktop.

**Deferred (unchanged):** follow-up #2 — the sine pack8 SIMD-Q15 routing re-validation
(with the Bv vec phase, sine's pack8 0.761 ns/sample now edges production float-SIMD
0.876; sine stays excluded via `osc_simd_q15_available` pending its own serial-LUT
SIMD-Q15 precision re-validation).
