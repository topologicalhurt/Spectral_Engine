# Patch notes — Pass 200: SIMD becomes the default CPU oscillator path

## Scope

Phase **U1b** of the oscillator-unification effort
(`docs/core_audit/OSCILLATOR_UNIFICATION_PLAN.md`). The host SIMD oscillator
(`core/port/host/oscillator_simd.c`) was complete and op-for-op equivalent to the
scalar reference, but **nothing ever called `osc_set_dispatch()`** — the dispatch
word was hard-coded to `OSC_DISPATCH_ALL_SCALAR`, so the SIMD path was dead code.

Maintainer direction (verbatim):

> "Don't delete it; why isn't it being used??? SIMD would be approached for CPU."
> "So long as the SIMD implementation is faster — why wouldn't it be the default?
> Why the hesitation to keep scalar the default?"

**Outcome:** SIMD is now the default CPU float oscillator path; the scalar
reference stays reachable for parity/debug via a new `--scalar` flag. The
maintainer is the golden authority and authorized this re-baseline.

## Why the SIMD path was dormant

Two reasons, both now answered:
1. **The switch was never thrown.** `osc_set_dispatch()` had zero callers and
   `g_osc_dispatch` defaulted to `OSC_DISPATCH_ALL_SCALAR` in `oscillator.c`.
2. **Flipping it is a golden change.** Under the host build's `-ffast-math
   -ffp-contract=fast`, the scalar Horner `phase0 + j*(…)` contracts to a hardware
   FMA while the SIMD intrinsics emit separate mul+add, so output diverges by a
   sub-epsilon amount despite identical source arithmetic. No automated test pins
   CPU-float golden bytes, so the flip breaks no ctest — but it is not byte-for-byte
   identical, which is why it stayed opt-out until explicitly greenlit.

## The change

1. **`core/oscillator.c`** — `g_osc_dispatch` default flipped
   `OSC_DISPATCH_ALL_SCALAR` → `OSC_DISPATCH_ALL_SIMD`. The existing
   `osc_simd_available(timbre)` guard in `timbre_synth_segment()` already routes
   timbres without a SIMD kernel (asin) back to scalar, and the band-limited
   quality path runs **before** dispatch, so neither is affected.
2. **`cmd/cli/spectral_cli.{h,c}`** — new `opts->osc_force_scalar` field
   (default 0); flags `--scalar` (force the scalar reference) and `-S` / `--simd`
   (explicit affirmative of the default); usage text added.
3. **`cmd/cli/spectral_cli_pipeline.c`** — `run_synthesis()` calls
   `osc_set_dispatch(opts->osc_force_scalar ? OSC_DISPATCH_ALL_SCALAR :
   OSC_DISPATCH_ALL_SIMD)` before dispatch. Added `#include "oscillator.h"` for
   the dispatch symbols. Only the CPU float path is affected; the GPU
   (Metal/CUDA) and ARM-Q15-native backends have their own kernels and ignore it.

## Scope of effect

The dispatch word gates **only** the CPU float synthesis path
(`timbre_synth_segment`). GPU tiles and the Q15 fixed-point ARM synth never route
through it, so this change is invisible to those backends.

## Verification

```text
- Five production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float) — only the pre-existing benign
  -mavx2 / -mno-avx512f notes.
- ctest 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift).
- Performance (saw, stretch 4×, CPU backend, shakespeare input, single thread):
      Synth  SIMD default 2331 ms   vs   --scalar 4261 ms   = 1.83× faster
  (matches the prior microbench: ~1.8× on saw/square, ~1.1× on sinf-bound sine).
- "SIMD is the default" proven: an unflagged render is BYTE-IDENTICAL to an
  explicit `-S/--simd` render (md5 ba70e68b…); `--scalar` produces a different
  file (md5 3f545351…).
- Parity SIMD-default vs --scalar on the full normalized additive mix
  (peak 0.95): max abs error 2.38e-7 (-132 dBFS), RMS error 1.07e-8 (-159 dBFS).
  This is accumulated per-partial FMA-contraction divergence over many summed
  oscillators plus normalization — far below 24-bit quantization (-144 dBFS) and
  inaudible. (Per a single oscillator segment the divergence is ≤1 ULP; the larger
  full-mix figure is summation + renormalization, not a per-sample regression.)
```

## Why this is safe

The faster path is the default precisely because it is measurably faster and
sub-quantization-equivalent to the reference, and the reference remains one flag
away (`--scalar`) for any parity/debug need. The behavior change was sanctioned by
the golden authority; see [[faster-path-should-default]].

## Status

SIMD is the **default CPU float oscillator**; `--scalar` restores the scalar
reference. Builds + ctest green; speedup and parity measured, not asserted.

## Proposed next pass (U1b follow-up)

Add an automated **SIMD-vs-scalar parity ctest** so the equivalence is pinned
going forward instead of measured by hand: render a fixed input through both
dispatch words (now trivially selectable via the flag) and assert the per-segment
divergence stays ≤ a documented budget. This converts the current manual parity
check into a regression guard and is the natural gate before the U1c L1-kernel
extraction.
