# Plan: compute STFT phase only at tracked peak bins (store re/im, not phase)

> **ARCHIVED 2026-06-18.** Landed + validated (see the LANDED section below); the only open item is the external FFTW/x86 CI check.

## LANDED 2026-06-18 (commits 973cf27, 91a66c4, e4da221, a5d5d4b, e258a12 on `minimal`)

Implemented in 5 steps + an adversarial audit. The instructive twist: the *speed*
hinged on the **storage type**, not the atan2 relocation.

- **fp32 re/im (steps 1–3) was a REGRESSION**: storing magsq+re+im as 3 fp32 arrays is
  +50% memory and, on a bandwidth-bound vDSP box, the extra `im` store (+~1.7ms FFT)
  outweighed the removed (already-vectorized) `vvatan2f`. Net +2.2ms / +3% on Apple
  Silicon, with **no** accuracy gain there (vDSP's dense phase was already exact). The
  audit (phase-at-peaks-audit workflow) confirmed correctness but recommended against
  shipping it unconditionally.
- **fp16 re/im (step e258a12) turned it into a WIN**: `magsq(fp32)+re+im(fp16)` =
  8 bytes/bin, *identical* to the old `magsq+phase(fp32)` → memory- AND bandwidth-
  neutral. Phase is taken only at peaks via `atan2f((float)im,(float)re)`, where fp16's
  ~3e-4 rad error is **−99.6 dBFS rms / −75 dBFS worst** (inaudible, tolerance-parity-
  passing). Measured (vDSP, interleaved A/B, 8 threads): fft 35.7→30.6 (−5.1), track
  9.9→12.2 (+2.3, per-peak atan2), **total 65.7→62.8 = −2.9ms / −4.4% faster; RSS
  256→256MB**. Single clean path, faster on both backends → **the backend gate was
  unnecessary**. FFTW/x86 should win more (its dense path was the scalar poly
  `spectral_magsq_phase`); validate `full_fused_parity` there.
- **estimator-gate (a5d5d4b)**: `handle_candidate` computes `atan2f` at `cf`+`next-cf`
  always and `cf±1` only when the estimator carries `CAP_COMPLEX_TRIPLET` (the complex
  estimators), since the default LOG_PARABOLIC reads only `cf`/`next-cf`. Halves the
  per-peak atan2 on the production path.
- Identical 179873 segments; 8 C parity tests + the `peak_estimator_contract` harness
  (updated to a real-valued re/im spectrum) green. `SpectralHalf` (= `_Float16`, float
  fallback) lives in `spectral_common.h`. `SPECTRAL_ENABLE_APPROX_ATAN2` / the
  `EXACT_ATAN2` guarantee no longer affect analysis (phase is exact-at-peaks everywhere).
- *Possible further squeeze (not done):* the track +2.3ms is the per-peak `atan2f`;
  since fp16 already softens to −99.6 dBFS, a fast poly atan2 at peaks could trim it.

### Plan validation — COMPLETE (the doc's Validation section, now closed)

- **Lazy-phase spot-check (the plan's "add a spot-check if a hard pin is wanted") — DONE.**
  `test_phase_at_peaks_lazy_phase_and_complex_gate_contract` (Part A) pins
  `seg->phase == atan2f((float)im[cf], (float)re[cf])` at the peak (within the fp16 band).
- **Complex-estimator gate — DONE.** Part B is a fail-on-bug test: it runs JACOBSEN
  twice with different `cf-1` phase and asserts the frequency offset MOVES; with the
  `CAP_COMPLEX_TRIPLET` gate neutered (cf±1 never computed) the test FAILS (verified) —
  so a future estimator that reads the cf±1 triplet without the cap is caught.
- **FFTW/x86 runtime validation — the ONE remaining item, and it is external by design.**
  The plan ("Why not landed") already scopes it to Linux CI: the macOS/ARM host links
  Accelerate/vDSP at configure time, so the `#else` FFTW producer can't be exercised here.
  It is code-reviewed (the audit) and got a defensive DC/Nyquist `im=0` pin; run
  `full_fused_parity` on an x86/Linux box with `fftw3f` to close it. Expect a bigger win
  there (its dense path was the scalar poly `spectral_magsq_phase`).

---

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
