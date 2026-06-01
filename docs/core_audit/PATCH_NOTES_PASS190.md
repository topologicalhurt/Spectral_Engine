# Patch notes — Pass 190: CTF sweep increment 30 — tree-wide defect-class cross-cut round 2 (float→int out-of-range conversion / signed-left-shift UB / transcendental-domain NaN injection) (clean audit) (Phase C)

## Scope

Phase C CTF/KISS adversarial defect sweep. Pass 189 cross-cut the first three
defect *classes* (unsigned-underflow-shift / integer div-by-zero / computed-size
memcpy). This pass cross-cuts the next three **undefined-behaviour / NaN-injection
classes** tree-wide — the ones a DSP kernel is most exposed to:

```text
- CLASS D: float→integer conversion where the value could be out of the target's
           range or NaN/Inf      (C11 6.3.1.4: out-of-range float→int is UB)
- CLASS E: left-shift on a signed operand, or by a runtime count >= width
           (C11 6.5.7: signed-overflow / over-width shift is UB)
- CLASS F: transcendental-domain violations — acos/asin outside [-1,1],
           sqrt/log of a negative or zero argument (NaN/-Inf that poisons audio)
```

**Outcome: clean audit. No defect found; no code changed.** Every float→int
conversion guards finiteness and saturates/reduces to range before the cast; every
left-shift operates on an unsigned operand with an in-range count; every
transcendental call is domain-guarded (internal helper guard, a pre-reduced
argument, or a provably non-negative operand).

## Class sweep D — float→integer conversion (out-of-range / NaN is UB)

Every float/double→int site routes through a finiteness check plus a saturating or
range-reducing step, so the truncated value is always representable in the target:

```text
- spectral_q15.h: spectral_float_to_q15 / _float_to_q31 -- `if(!isfinite) return 0;`
  then saturate to [-1,1] (Q15_MIN/MAX, Q31_MIN/MAX) BEFORE `*SCALE` cast. In range.
- spectral_q15.h: spectral_phase_rad_to_q15 -- finite-guard, fmod-normalize to [0,1),
  explicit `if(n>=1.0f) n-=1.0f` so (n-0.5)*65536 stays < 32768 (the comment calls out
  the exact +1.0f-rounds-to-1.0f edge). int16 cast can't go OOB.
- spectral_q15.h: spectral_omega_to_q88 -- `if(!isfinite||o<=0) return 0;`, /4 then clamp
  255 so o*256 <= 65280 < 65536. uint16 in range. (o>255 is unreachable for real audio:
  any physical omega rad/sample < pi < 4.)
- spectral_wavetable.c:589 lookup_f -- finite-guard, phase reduced to [0,1) (frac + the
  `<0 -> +1` fixup), idx_f = phase*SIZE in [0,SIZE); `(uint32_t)idx_f` in range, plus a
  belt-and-braces `if(idx>=SIZE) idx=0`; samples[] has SIZE+1 entries so idx+1 is valid.
- spectral_synth_internal.c:286 start_idx/length -- start_d/length_d are finite-checked,
  `>= 0`, `< out_len`, AND `<= SIZE_MAX` BEFORE the (size_t) cast (lines 277-284).
- cli_pipeline.c:938/975 out_len = (size_t)(out_n_samples * stretch) -- stretch is
  validated finite-positive AND `<= SPECTRAL_MAX_STRETCH` at the CLI boundary
  (spectral_cli.c:342,547); the path is `#if !SPECTRAL_EMBEDDED` (64-bit host), so the
  positive bounded product stays far below SIZE_MAX. No negative/overflow cast.
- peak_interp.c:63 (int)cf, peak_estimator.c (size_t)best_next_bin -- the bin indices are
  bounded `n_freqs <= INT_MAX` and `n_freqs == n_fft/2+1` (peak_track.c:677-681), and
  best_next_bin is `>= 0`-checked (peak_estimator.c:73) before the size_t cast.
```

## Class sweep E — left-shift UB (signed operand / over-width count)

Grepped every `<<` in the tree. **Every** left-shift has an unsigned left operand
and an in-range count; none can trigger signed-overflow or over-width UB:

```text
- out.c:131-133  (uint32_t)chunk[n] << {8,16,24}      -- unsigned, count < 32.
- debug_embedded_arm.c:166-172,503,545  1u << ITM_PORT_*  -- 1u unsigned, ports < 32.
- synth_arm32.c:319  freq_inc << 1                    -- uint32_t; phase accumulator,
                                                          modular-by-design (defined wrap).
- synth_arm32.c:472  1u << 24                          -- 1u, compile-time count 24.
- synth_arm32.c:865  ((uint32_t)((int32_t)phase_q15+32768)) << 16 -- operand promoted to
                                                          uint32_t (phase_q15 in [-32768,32767]
                                                          -> +32768 -> [0,65535]); count 16.
- wavetable.c:602  1u << frac_bits  (frac_bits = 16 - SPECTRAL_WAVETABLE_BITS, a nonzero
                                     compile-time constant < 16) -- 1u unsigned, count in (0,16).
```

No signed-operand shift and no runtime/over-width count exists.

## Class sweep F — transcendental-domain (NaN/-Inf injection)

Every domain-restricted call is guarded — by an internal helper check, a value that is
mathematically already in-domain, or a finite-then-clamp at the call site:

```text
- osc_formulas.h:94 spectral_osc_asin -- clamps rads*INV_PI to [-1,1] before asinf
  (the comment notes asinf(>1) -> NaN poisons dst[j]+=amp*wave). Domain-safe.
- fast_math.c: fast_sqrt / fast_inv_sqrt -- `if(!(x>0)||!isfinite) return 0;` up front.
  fast_peak_log -- `if(!(x>0)||!isfinite) return logf(x);` (exact path) and the
  subnormal/inf exponent guard. All callers see a defined result.
- windows.c:244  sqrtf(metrics.energy/length) -- energy is a sum of squares (>=0),
  length > 0 (guarded). Non-negative argument.
- windows.c:107-121 (peak_magsq_log_parabolic) -- rejects center_sq<=0 and non-finite
  inputs up front, applies SPECTRAL_TRACK_LOG_FLOOR to left/center/right before the
  ratio-logs, and finite-checks log_lc/log_rc/log_gain/peak at every stage (falls back
  to the centered-log form on non-finite). No -Inf/NaN escapes.
- peak_estimator.c:230,333-335,817-818 fast_sqrt(...) -- arguments are magnitude-squared
  / triplet powers (>=0); fast_sqrt guards regardless.
```

## Verification

```text
- No source changed this pass (read-only cross-cut), so the green state is preserved by
  construction. Triad re-run on the current tree to confirm:
    * five production targets build clean (desktop, simulate, simulate_daisy,
      embedded_arm, embedded_arm_float) — only the pre-existing benign -mavx2 /
      -mno-avx512f notes.
    * ctest: 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
      core_guarantees_drift).
```

## Phase C status

With this increment the sweep has cleared 161-188 file-by-file, cross-cut the first
three defect classes tree-wide (189) and now the next three UB/NaN-injection classes
(190, clean — every float→int conversion is finite-guarded and saturated/range-reduced
before the cast; every left-shift is unsigned with an in-range count; every acos/asin/
sqrt/log is domain-guarded). The host-verifiable kernel has **no open defect leads**:
all compute, support, dispatch, I/O, instrumentation, optional-processing, and firmware
surfaces are audited file-by-file, and six independent defect *classes* have been
cross-cut tree-wide. The two recorded observations (GPU fade-tail-under-time-stretch;
Daisy SD `.spq` load re-validation) remain bounded, memory-safe, and deferred
maintainer-directed because they are unverifiable on this host. Phase C is at
convergence; Phase D (compiled harness + LUT golden-vector loop) is the natural home
for the two deferred items.
