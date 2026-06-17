# Plan: compute STFT phase only at tracked peak bins (store re/im, not phase)

Status: DESIGNED + critiqued (multi-agent workflow), ready to execute. The immediate x86/Linux
slowness is ALREADY fixed (commit 030d3ee — backend-dependent `SPECTRAL_ENABLE_APPROX_ATAN2`
default). This is the strictly-better follow-up: exact phase everywhere, faster than even Apple's
current `vvatan2f`-over-all-bins, by computing `atan2` only at the ~tens of peak bins.

## Root cause recap (why this is the real fix)

The STFT computes phase = `atan2(im,re)` for EVERY bin, then `reconstruct_complex`
(`spectral_peak_estimator.c:198-236`) does the REVERSE at peaks — `mag=sqrt(magsq);
sincos(phase); re=mag·cos, im=mag·sin` — re-deriving re/im it threw away. Storing re/im instead
retires BOTH the all-bins atan2 AND the per-peak reconstruction, and is more accurate (no
round-trip). Phase is then only needed as the segment's output value at peak bins.

## Chosen design: store `magsq + re + im`; lazy phase in the tracker

Two layouts were critiqued. **Design 1 (re+im only, "2x memory-neutral") is illusory**: the
full-matrix tracker scan parallelizes over frames with OMP and random-accesses scaled magsq at
arbitrary `t`/`t+1` (`spectral_peak_track.c:818,836-838`), so scaled magsq must stay live for all
frames → 3x anyway. **Design 2 (magsq+re+im) is sound** (keeps `magsq[]` byte-identical, so the hot
SIMD peak-scan + the exact segment-COUNT parity assertion are untouched). Memory: +50% on the
full-matrix path (bounded ≤3×256 MB at the 32M-bin chunk threshold); the large-file FUSED path only
adds 2 scratch rows/thread (negligible).

**Lazy-phase variant (lowest risk): leave the estimator and ALL its tests untouched.** The tracker
loop, per peak candidate `cf`, fills a tiny per-thread phase scratch — `phase[cf-1..cf+1]` and
`next_phase[cf]` via `atan2f(im,re)` — and passes it to `emit_segment`/the estimator exactly as
today. This avoids the estimator-side ripples the critique found (amplitude scaling, Quinn's
non-scale-invariance, the `CAP_PHASE_ROW` capability flags, and the 3 Python harnesses that set
`phase[]` literally). Only the producer + storage + tracker loop change.

## Exact migration (file by file)

1. **Producer — `analysis/spectral_analysis_fft.c`** (both `spectral_fft_single_frame` bodies +
   `spectral_fft_frames`; signature in `spectral_analysis_internal.h:122-134`): replace the
   `out_phases` param with `out_re, out_im`.
   - vDSP body (256-324): keep the magsq block; REPLACE the phase block (307-314, the `vvatan2f`)
     with a copy of the 0.5-scaled `split.realp/imagp` into `out_re/out_im`. DC/Nyquist:
     `out_re[0]=realp[0], out_im[0]=0; out_re[n-1]=imagp[0], out_im[n-1]=0`; interior memcpy.
   - FFTW body (328-363): call `spectral_magsq_only` (always, for magsq) + deinterleave `thread_out`
     (already interleaved re/im) into `out_re/out_im`. Retires `spectral_magsq_phase`'s atan2.
   - `out_re==NULL` keeps the magsq-only fast path (replaces `out_phases==NULL`).
2. **Storage — `spectral_analysis_internal.h` + `spectral_analysis.c`**: `SpectralAnalysisStftMatrix`
   `phases` → `re, im` (3 arrays in alloc/free, lines 79-95).
3. **Full driver — `spectral_analysis_full.c`** (`run_full`): write the frame into `matrix->re/im`;
   pass `matrix->re/im` to `spectral_tracker_process`.
4. **Fused driver — `spectral_analysis_fused.c`**: `SpectralFusedScratchRows` (25-30)
   `phase_curr/next` → `re_curr/next, im_curr/next` (6 rows); frame calls + `run_fused_frame` pass re/im.
5. **Tracker — `spectral_peak_track.c`**: `spectral_tracker_process` + `run_fused_frame` +
   `SpectralFrameContext` carry re/im rows instead of phase rows. Add a per-thread phase scratch
   (n_freqs) to the tracker's per-thread buffers (alloc near line 724). In the candidate handler
   (~333), before each `spectral_tracker_emit_segment`, fill `phase_scratch[cf-1..cf+1]` and
   `next_phase_scratch[cf]` from re/im via `atan2f`, and pass `phase_scratch` as `phase_row`.
   `emit_segment` (`spectral_peak_interp.c`) + the estimator stay **byte-identical**.

## Required fixes the critique flagged (must honor)

- The peak-scan key MUST stay the **double-precision scaled** magsq (`spectral_fft_scaled_magsq`),
  not a single-precision `re*re+im*im` — else a strict-local-max tie can flip and break the exact
  segment-COUNT assertion in `full_fused_parity`. Keeping `magsq[]` stored (Design 2) preserves this.
- The same `atan2f(im[cf],re[cf])` value must feed all THREE peak-bin phase reads
  (`spectral_peak_interp.c:166` validity gate, `:202` `seg->phase`, and the estimator input) — one
  atan2, not three.
- DC/Nyquist phase from re/im (`im=0`) reproduces today's `0`/`π` sign special-case exactly.

## Validation

- `full_fused_parity` + `gpu_backend_parity` are tolerance-based and compare full-vs-fused /
  CPU-vs-GPU — both sides shift together, so they stay green without re-baselining. `seg->phase`
  becomes exact libm `atan2f` at peaks (better than vvatan2f/poly) — benign ULP change, invisible to
  those tests, so add a spot-check if a hard pin is wanted.
- On macOS/ARM the vDSP body + the storage/tracker changes are exercised (validates the structure);
  **the FFTW body must be validated on x86/Linux** (the path that benefits most) — run
  `full_fused_parity` there before trusting it.
- Benchmark: `spectral_magsq_phase`'s all-bins atan2 is eliminated; phase work drops from
  O(n_freqs·n_frames) to O(peaks·n_frames·~4). Expect a large analysis speedup on x86/Linux.

## Why not landed in-session

It is a 6-file atomic change to the most-tested, parity-gated analysis hot path, and the FFTW body
(the primary beneficiary) cannot be exercised on the macOS/ARM dev box — so it needs a Linux
validation pass. The immediate perf problem is already fixed, so this carries no urgency that
justifies a rushed big-bang on the critical path.
