# Patch notes — Pass 175: CTF sweep increment 15 — peak frequency-estimation cluster (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. This pass audits the **peak
frequency/amplitude estimation cluster** — the DSP-dense core that converts a
detected magnitude-squared peak into a sub-bin frequency, amplitude, temporal
frequency-slope (`df`) and phase-vocoder instantaneous frequency:

```text
- analysis/spectral_peak_estimator.c  Jacobsen / Candan / Quinn-second complex
                                      estimators, log- & magnitude-parabolic
                                      interpolation, phase-vocoder advance,
                                      magsq->amp + bounded window-peak gain
- analysis/spectral_peak_interp.c     spectral_tracker_validate_candidate
                                      (the row[cf-1..cf+1] neighborhood load)
- analysis/spectral_peak_track.c      candidate-generation loops + freq_step_df
                                      derivation (verified for the consumer side)
- core/spectral_segment_math.h        the phase/amplitude synthesis model that
                                      consumes omega / df / da (consumer contract)
```

**Outcome: clean audit. No defect found; no code changed.** Per campaign protocol
a clean audit is a legitimate result and a defect must not be fabricated.

## What was checked and why it is correct

### Memory safety — the `validate_candidate` neighborhood load (open lead from Pass 175 entry)

`spectral_tracker_validate_candidate` (`spectral_peak_interp.c:41-46`) reads
`row[cf-1]`, `row[cf]`, `row[cf+1]`, `next_row[cf-1..cf+1]` but only guards
`cf != 0 && cf <= INT_MAX` — it does **not** itself check `cf+1 < n_freqs`. This
is safe by **caller contract**, now verified end-to-end:

```text
- Both candidate generators start at f = 1 and bound the scan strictly below
  n_freqs-1:
    chunked path (spectral_peak_track.c:954,979,1012,1044)
    fused   path (spectral_peak_track.c:1493,1498,1531,1564)
  SIMD lanes: `f + 7 < n_freqs - 1` (AVX2) / `f + 3 < n_freqs - 1` (SSE) keep
  every one of the 8/4 lanes <= n_freqs-2; movemask only sets bits for those
  lanes, so cf = f + ctz(bits) <= n_freqs-2.
  Scalar tail: `f < n_freqs - 1` => cf = f <= n_freqs-2.
- Therefore every emitted candidate cf in [1, n_freqs-2]:
    cf-1 >= 0  and  cf+1 <= n_freqs-1 < n_freqs  => both neighbor loads in-bounds.
- The cf==0 guard inside validate_candidate additionally protects the public
  surface (cf-1 underflow) for any future direct caller.
Conclusion: NOT a defect. The missing upper-bound check is covered by the
generator invariant; SIMD loads `row + f - 1` are safe because f starts at 1.
```

### DSP / math correctness — every estimator verified against its published form

```text
Jacobsen (spectral_peak_complex_offset_jacobsen, 258-289)
  offset = Re{(X[k-1]-X[k+1]) / (2X[k]-X[k-1]-X[k+1])}
  implemented as Re{num*conj(den)}/|den|^2 = (num_re*den_re+num_im*den_im)/den_mag2.
  Exactly Jacobsen & Kootsookos (2007); peak at bin k+offset. Sign matches the
  parabolic estimators (positive offset == peak above center). Quotient in double
  with a |den|^2 >= 1e-30 guard before the divide.

Candan (spectral_peak_offset_candan, 349-374; correction 141-160)
  offset = Jacobsen_offset * (tan(alpha)/alpha), alpha = pi/n_fft,
  n_fft = (n_freqs-1)*2. This is Candan's (2011) finite-N bias correction
  N/pi*tan(pi/N). Overflow-guarded ((n_freqs-1) <= SIZE_MAX/2) and correction
  clamped to 1.0 on any non-finite/<=0 result.

Magnitude-parabolic (spectral_peak_offset_mag_parabolic, 315-348)
  p = 0.5*(a-c)/(a-2b+c) with a,b,c = sqrt(magsq) — Smith's quadratic peak
  interpolation, denom in double with SPECTRAL_TRACK_PARABOLIC_DENOM_EPS floor
  (returns offset 0 on a flat triplet, not a divide).

Log-parabolic (spectral_peak_offset_log_parabolic, 292-314)
  routes to spectral_window_interp_magsq_parabolic — the Smith log-power form
  already verified exact in Pass 174.

Quinn second (spectral_peak_offset_quinn_second + quinn_tau, 376-449)
  ap = Re(X[k+1]*conj(X[k]))/|X[k]|^2, dp = -ap/(1-ap);
  am = Re(X[k-1]*conj(X[k]))/|X[k]|^2, dm =  am/(1-am);
  d = 0.5*(dp+dm) + tau(dp^2) - tau(dm^2),
  tau(x) = 0.25*log(3x^2+6x+1) - (sqrt6/24)*log((x+1-sqrt(2/3))/(x+1+sqrt(2/3))).
  Matches the canonical published Quinn-second pseudocode term-for-term, incl.
  the asymmetric -ap/(1-ap) vs +am/(1-am) sign. Constants exact:
  sqrt6=2.449489742783178, sqrt(2/3)=0.816496580927726. |1-ap|,|1-am| >= 1e-12
  guards, dp^2/dm^2 <= FLT_MAX before the float tau, tau guards x>=0 / a>0 / ratio>0.

Phase-vocoder advance (spectral_peak_estimate_phase_advance, 457-549)
  residual = princarg(dphi - k*freq_step_omega*hop),
  phase_bin_offset = residual/(freq_step_omega*hop),
  phase_omega = (k+phase_bin_offset)*freq_step_omega,
  phase_error = princarg(dphi - model_omega*hop).
  Textbook instantaneous-frequency relation; all products formed in double with
  finite + |.|<=FLT_MAX guards and a |denom| >= 1e-12 floor before the divide.
  princarg via x - 2pi*floor(x/2pi + 0.5).
```

### Dimensional consistency — the `0.5` in `freq_step_df` is correct (not an off-by-factor)

This was the one factor that looked suspicious and was chased to the consumer:

```text
Producer (spectral_peak_track.c:706):
  freq_step_df = 0.5 * (sr/n_fft) * (1/hop) * (2pi/sr) = 0.5 * freq_step_omega / hop
  df = bin_delta * freq_step_df          (spectral_peak_estimator.c:829)

Consumer (spectral_segment_math.h:40-44, shared by ARM/sim/CPU/Metal/CUDA):
  phase(n) = phase0 + n*(alpha + beta*n) = phase0 + omega*n + df*n^2
  => instantaneous omega(n) = dphase/dn = omega + 2*df*n

For the segment to sweep omega -> omega_next across one hop:
  omega + 2*df*hop = omega_next
  => df = (omega_next-omega)/(2*hop) = bin_delta*freq_step_omega/(2*hop)
        = bin_delta * (0.5*freq_step_omega/hop) = bin_delta*freq_step_df.  EXACT.

The 0.5 in freq_step_df precisely cancels the 2 from d/dn of the df*n^2 term, so
the chirp passes through the correct endpoint frequency at n=hop. The amplitude
path is the linear analogue (da = (next_amp-amp)*inv_hop; amp(n)=amp0+da*n;
amp(hop)=next_amp) with no 0.5, also correct. Producer and consumer agree.
```

### Bounds / overflow / NaN hygiene

```text
- best_next validation (70-90): best_next in [bin-1,bin+1] AND < n_freqs; the
  int df subtraction is guarded against INT_MIN on the public wrapper.
- bounded_magsq_gain (581-602): gain limit center_magsq*max_gain computed in
  double so a large center cannot overflow to Inf and wave the bound through.
- next-frame triplet loads reject center_bin==0 / center_bin+1>=n_freqs, so a
  best_next at the spectrum edge degrades to "no refinement", never an OOB.
- every estimator funnels non-finite intermediate results to a 0/return-0 path;
  offsets hard-clamped to [-0.5, 0.5]; amp/next_amp/da/omega/df finite-checked
  before the estimate is accepted.
```

## Verification

```text
- No source changed this pass, so the Pass 174 green state is preserved: the five
  production targets (desktop, simulate, simulate_daisy, embedded_arm,
  embedded_arm_float) build clean (only the pre-existing benign -mavx2 /
  -mno-avx512f notes) and the host binaries are byte-identical to the
  Pass-174-verified tree by construction.
- ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared fixed-point (161), analysis/peak-track
(162), port/SIMD/out (163), hashing/parsing/path (164), DSP-math/FFT-scaling +
alloc/cache (165), synth-backends + analysis-orchestration (166), CLI/orchestration
(167), embedded fade envelope (168), core synth dispatch/internal helpers (169),
binary-deserialization/converter (170), host GPU-tile concurrency (171), the
oscillator asin domain guard (172), the host SIMD quantized domain guard (173), the
file-I/O + CLI untrusted-input boundary cluster (174, clean), and the peak
frequency-estimation cluster (175, clean — Jacobsen/Candan/Quinn/parabolic math,
phase-vocoder advance, the validate_candidate neighborhood-load bound, and the
freq_step_df 0.5 producer/consumer factor all verified). Phase D (compiled harness
+ LUT golden-vector loop) follows.
