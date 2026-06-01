# Patch notes — Pass 187: CTF sweep increment 27 — ARM DWT/ITM debug instrumentation: unsigned-EWMA underflow fix (×3) + analysis-proc / processing-chain / console / log / audio-in audit (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass clears the remaining
genuinely-unswept support and instrumentation files (0 prior PATCH_NOTES mentions
each), and lands one **real defect fix** in the ARM debug monitor:

```text
- synth/backends/arm/spectral_debug_embedded_arm.c   DWT/ITM perf monitor   <-- DEFECT FIXED (×3)
- analysis/spectral_proc_serra_smith_1990.c          deterministic stage    (no-op stub)
- analysis/spectral_proc_johnston_1988.c             masking stage          (no-op stub)
- analysis/spectral_proc_adaptive_track_density.c    track-density stage    (no-op stub)
- analysis/spectral_processing_chain.c               mask parse + dispatch  (clean)
- runtime/spectral_console.c                         table/box formatting   (clean)
- core/spectral_log.c                                log level + vfprintf    (clean)
- core/spectral_in.c                                 libsndfile audio read  (clean)
```

## Defect fixed — unsigned EWMA underflow in the debug perf monitor

The DWT/ITM debug monitor keeps three exponentially-weighted rolling averages with
the classic embedded idiom `avg += (new - avg) >> k`. **All three operands are
`uint32_t`** (`SpectralTimingStats.cycles_avg`, `SpectralSdcardStats.read_latency_us`,
`write_latency_us` — see the header struct defs). That idiom is only correct in
**signed** arithmetic: the `(new - avg)` term must be allowed to go negative so the
shift divides a negative delta and the running mean can move *down* toward a
below-average sample.

Computed in unsigned, when `new < avg` the subtraction wraps:
`(new - avg) mod 2^32 = 2^32 - (avg - new)`, a value `>= 2^31`. The **logical**
`>>` then yields ≈ `2^28` (for `>>4`) instead of a small negative correction, so
`avg` jumps by hundreds of millions on the very next below-average sample.

```text
Worked example (timing EWMA, k=4):  avg = 1000, new(elapsed) = 900
  signed (intended):   delta = 900-1000 = -100;  -100 >> 4 = -7;  avg -> 993   (mean drifts down, correct)
  unsigned (bug):      (900-1000) mod 2^32 = 0xFFFFFF9C = 4294967196
                       4294967196 >> 4 = 268435449;  avg -> 268436449          (garbage)
```

This is **reachable in the normal case**, not an impossible edge: a DSP block running
faster than the running mean (or an SD access quicker than the running latency)
happens constantly. The average is corrupted on the first such sample. The only
reason it has stayed latent is that the monitor is `#ifdef SPECTRAL_DEBUG_ARM`
(built only with `-DSPECTRAL_DEBUG` on a Cortex-M7 arch), so it is not exercised by
the production triad.

### Fix (3 sites — `spectral_debug_timing_end`, `spectral_debug_sdcard_read/_write`)

Form the delta in `int32_t`, apply the arithmetic shift, then fold back into the
`uint32_t` field:

```c
int32_t avg_delta = (int32_t)elapsed - (int32_t)ctx->timing.cycles_avg;
ctx->timing.cycles_avg = (uint32_t)((int32_t)ctx->timing.cycles_avg + (avg_delta >> 4));
```

and the `>>3` latency analogues for the SD-card read/write paths.

**Domain safety of the signed casts:** the averaged quantities are per-block DWT
cycle deltas (`(cpu_freq/sample_rate)*block_size`, order 10^5–10^6 cycles) and
micro-second SD latencies (order 10^3–10^5). Both are far below `INT32_MAX`
(≈2.1×10^9 — ~4.4 s at 480 MHz), so the `(int32_t)` reinterpretation is exact; the
fix is correct for the entire realistic operating range. Arithmetic right shift of a
negative `int32_t` is the well-defined behaviour on every target compiler
(GCC/Clang on ARM), matching the original EWMA intent.

No other instance of the idiom exists: a repo-wide grep for `+= ( … >> n` returns
only these three plus two non-EWMA sites that add strictly non-negative terms
(`spectral_wavetable.c:403` masked checksum byte; `spectral_perf_model.c:114`
`(miss_units+3)>>2` ceiling-divide) — neither can underflow.

## What else was checked and is correct

```text
- spectral_proc_{serra_smith_1990,johnston_1988,adaptive_track_density}.c: all three are
  no-op stubs ((void)-casting their args, returning SPECTRAL_OK). No logic to audit.
- spectral_processing_chain.c: mask parse is overflow-checked (spectral_size_add for the
  strtok buffer, NUL-bounded copy), numeric specs rejected unless within ALL_KNOWN, the
  none/saw_none mutual-exclusion is enforced, and mask_to_string's snprintf advance never
  forms an out-of-range pointer (used < out_size holds at every `out+used`, with an immediate
  break once truncation pushes used >= out_size). Stage dispatch masks-and-applies, accumulates
  applied/pending, propagates the first stage error. Clean.
- spectral_console.c: pure formatting. print_padded_* clamp pad>=0 and snprintf into bounded
  bufs; progress_bar filled=(int)(fraction*width) with fraction pre-clamped to [0,1] so
  filled in [0,width]; table width sums are config-bounded. Clean.
- spectral_log.c: switch->level-name table + NULL-fmt guard + NULL-stream->stdout default,
  paired va_start/va_end. Clean.
- spectral_in.c: spectral_audio_read proves sf_count_t->size_t representability before the
  overflow-checked frames*channels / frames*sizeof(float) allocs, finite-validates the whole
  span, and the mono downmix base=i*channels cannot exceed the already-checked total_samples
  (i<frames => base+ch < frames*channels), freeing on every error path. spectral_audio_window
  clamps start/end seconds into [0,total_frames] (NaN end_sec slips the >=0 guard but is caught
  by the is_finite_f64(end_d) test), rejects end<=start, and returns an in-bounds slice. Clean.
```

## Verification

```text
- Triad re-run on the current tree after the fix:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
  The edited TU is #ifdef SPECTRAL_DEBUG_ARM, undefined in all five triad targets, so the
  fix changes no triad-emitted code (the targets recompiled an empty TU) and the triad is
  green by construction.
- Because the fix is not reachable from the triad, it was separately proven to compile with
  the body active: `clang -fsyntax-only -DSPECTRAL_DEBUG -D__ARM_ARCH_7EM__ -DSPECTRAL_PLATFORM_DAISY`
  over the file compiles cleanly; the three edited EWMA lines emit no diagnostic. (The lone
  warning is the pre-existing line-318 `(uint32_t)&_estack` pointer-to-int cast, a host-only
  64-bit artifact — pointers are 32-bit on the real ARM32 target — untouched by this pass.)
```

## Phase C status

With this increment the sweep has cleared 161-186 (see prior notes) and now the
instrumentation/analysis-stub/support cluster (187 — one real fix: the unsigned-EWMA
underflow in the ARM debug monitor's timing and SD-latency rolling averages, corrected to
signed-delta arithmetic at all three sites; the three psychoacoustic/sinusoidal proc stages
are confirmed no-op stubs; processing-chain mask parse/dispatch, console formatting, the log
shim, and the libsndfile audio-input path are clean). All compute, support, dispatch, I/O,
and now the debug-instrumentation and optional-processing surfaces are audited. Phase D
(compiled harness + LUT golden-vector loop) remains the natural home for the deferred GPU
fade-tail regression vector.
