# Patch notes — Pass 174: CTF sweep increment 14 — file-I/O + CLI boundary cluster (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **untrusted-input
boundary cluster** — the surfaces that ingest external bytes (audio files, CLI
argv) and emit artifacts:

```text
- core/spectral_in.c        audio file reader (libsndfile), channel downmix, time window
- core/spectral_windows.c   window generators (Hann/Hamming/Blackman/rect) + window
                            metrics + Smith log-parabolic peak interpolation/height
- runtime/spectral_utils.c  numeric arg parsers (parse_i32_arg / parse_f32_arg /
                            parse_size_arg / getenv_f64*)
- cmd/cli/spectral_cli.c     CLI validation (n_fft / hop / stretch / pitch / timbre /
                            backend / threads / time window)
- core/spectral_out.c       float-WAV writer + bounds-checked RIFF PEAK-timestamp scrubber
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol
a clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

```text
spectral_in.c (spectral_audio_read / spectral_audio_window)
  - Every libsndfile-reported quantity is validated before use: frames>0,
    channels>0, samplerate in [MIN,MAX]; sf_count_t -> size_t via
    spectral_sf_count_to_size (round-trip overflow check); total_samples and
    mono_bytes via spectral_size_mul / spectral_array_bytes (overflow-checked).
  - Read-back count is verified (read != frames -> ERR) and the whole buffer is
    spectral_f32_span_finite() before any math, so NaN/Inf samples in the input
    cannot propagate.
  - Downmix sums channels in double, range/finite-checks mono_d against +/-FLT_MAX
    before the float cast. base = i*channels stays < total_samples (already proven
    non-overflowing), so no OOB.
  - spectral_audio_window: NaN/Inf start/end_sec all funnel to ERR_PARAM (the
    negative-end sentinel path is finite by construction; a NaN end_sec yields a
    NaN end_d caught by the is_finite_f64 gate); end_frame<=start_frame rejected;
    out_start/out_frames only set on success.

spectral_windows.c
  - Generators guard length==0 and length==1 (avoids /(N-1) division by zero) and
    spectral_window_generate re-checks finiteness of the realized window.
  - DSP math verified exact: spectral_window_interp_magsq_parabolic reduces Smith's
    three-log quadratic to p = 0.5*(y[-1]-y[1])/(y[-1]-2y[0]+y[1]) with
    y = log(power) (centered-ratio form algebraically identical), with a finite/
    tiny-denominator fallback and a hard [-0.5,0.5] clamp applied to BOTH the
    log and the (compile-gated) rational branch. peak_magsq_log_parabolic guards
    all inputs, floors to LOG_FLOOR, finite-checks log_gain/peak, and enforces
    peak>=center_sq.
  - window_sum / window_energy accumulate in double with finite + FLT_MAX overflow
    guards; metrics derive every scale (2/sum, 1/sum, ENBW = N*energy/sum^2) from
    the *realized* window and only set the *_VALID flags when the result is finite
    and positive, so backend window-convention differences stay self-consistent.

spectral_utils.c parsers
  - parse_i32_arg/parse_f32_arg/parse_size_arg/getenv_f64*: strtol/strtof/strtoull/
    strtod each check errno, no-digits (end==s), trailing chars (*end!='\0'),
    range (INT_MIN..INT_MAX / SIZE_MAX) and finiteness (floats reject NaN/Inf).
    No atoi/atof UB anywhere.

spectral_cli.c validation
  - cli_is_valid_fft_size: n_fft >= SPECTRAL_MIN_FFT_SIZE(64) AND power-of-two
    ((n & (n-1))==0); the >=64 gate rejects negatives/zero before the bit test.
    (No SPECTRAL_MAX_FFT_SIZE exists; the largest power-of-two int is 2^30 and all
    downstream buffer sizing is spectral_size_mul-guarded, so the absence of an
    upper cap is not a memory-safety gap.)
  - cli_is_valid_hop_size: hop>0 AND hop<=n_fft (no zero-hop division, no hop>frame).
  - cli_validate_common_synth_params: timbre in [SINE,PWM], stretch via
    spectral_is_finite_positive_f32 (rejects <=0 and non-finite), pitch finite +
    range; stretch upper bound vs SPECTRAL_MAX_STRETCH; backend enum range;
    threads clamped to [1, hw].

spectral_out.c
  - spectral_audio_write: validates path/buffer/frames/channels/samplerate,
    overflow-checks frames_sf + sample_count, rejects non-finite buffer BEFORE
    writing (no NaN/Inf reaches the file), disables the PEAK chunk for determinism,
    and verifies written==frames && close==0.
  - spectral_wav_scrub_peak_timestamp: manual RIFF walk is fully bounds-checked —
    riff_payload_size in [4, file_size-8], riff_end<=file_size, per-chunk
    size <= riff_end-data_pos (no wrap), odd-size word padding with a UINT64_MAX
    guard, next_chunk_pos<=riff_end, and the PEAK timestamp write range-checked
    against riff_end. A size==0 chunk still advances the cursor by the 8-byte
    header each iteration, so the loop cannot spin.
```

## Verification

```text
- No source changed this pass, so the Pass 173 green state is preserved: the five
  production targets (desktop, simulate, simulate_daisy, embedded_arm,
  embedded_arm_float) remain built clean and the host binaries are byte-identical
  to the Pass-173-verified tree.
- ctest re-run to confirm: 4/4 PASSED (arm32_process_correctness, core_contracts,
  core_guarantees, core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared fixed-point (161), analysis/peak-track
(162), port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling +
alloc/cache (165), synth-backends + analysis-orchestration (166), CLI/orchestration
(167), embedded fade envelope (168), core synth dispatch/internal helpers (169),
binary-deserialization/converter (170), host GPU-tile concurrency (171), the
oscillator asin domain guard (172), the host SIMD quantized domain guard (173), and
the file-I/O + CLI untrusted-input boundary cluster (174, clean). Phase D (compiled
harness + LUT golden-vector loop) follows.
