# Q-island audit — desktop hot paths (refactor Thread B3)

**Purpose.** The maintainer's ask was to leverage Q15 on "every path that could benefit"
— and explicitly *not to be myopic* about it. The trap on the other side is equally real:
"Q15 everywhere" is a net-negative scattershot, because PASS210 showed a float↔Q round-trip
mid-kernel costs *more* than the float work it replaces. Q15 only pays inside a **contiguous
integer island** — phase + eval + (one) widen — with no float round-trip in the middle.

This audit enumerates every desktop hot path, classifies each as a **Q-island candidate**
vs **inherently float**, and gives the *reason* each is in or out. It makes the coverage
decision evidence-based and reviewable rather than aspirational. It is grounded in the code,
not assumptions (per [[avoid-assumptions]]).

## The desktop pipeline (the timed hot paths)

The CLI runs five timed stages (`spectral_cli_pipeline.c`, `t_fft/t_track/t_synth/t_norm/
t_write`, summed at line 1321):

```
read WAV ──► FFT ──► Track ──► Synth ──► Norm ──► Write ──► float WAV
            (analysis)        (synthesis)        (output)
```

| # | Stage | Hot data | Class | Why |
|---|-------|----------|-------|-----|
| 1 | **FFT** | complex spectra (vDSP) | **inherently float** | STFT is a complex float transform; `spectral_fft_frames` (`analysis/spectral_analysis_full.c:44`) is vDSP/float32. No fixed-point form without a bespoke integer FFT — not a packing opportunity, a rewrite. |
| 2 | **Track** | `const float* magsq, const float* phases` | **inherently float** | Peak tracking (`analysis/spectral_peak_track.c:1408`) works on magnitudes (huge dynamic range → needs the float exponent), atan2-derived phase, and **sub-bin parabolic peak interpolation** (fractional bin positions). Q15's ~92 dB SNR ceiling can't hold the spectral dynamic range. |
| 3 | **Synth** | per-sample waveform | **mixed — the one island lives here** | See the breakdown below. |
| 4 | **Norm** | assembled float mix | **inherently float** | `spectral_normalize_float` (`core/spectral_io.h:62`) scans the global peak of the **summed** signal then scales. The sum routinely exceeds ±1.0 pre-norm (additive partials) — not Q15-representable; finding a peak across the whole buffer needs float range. |
| 5 | **Write** | output buffer | **inherently float** | Output is 32-bit float WAV: `SF_FORMAT_WAV | SF_FORMAT_FLOAT` + `sf_writef_float` (`core/spectral_out.c:202,215`). There is **no int16 boundary to repack** — a 16-bit output mode would be a new lossy feature, not a free packing win (this is the Q1-closed-by-audit finding, still true). |

**Four of the five stages are inherently float.** Two are float by *information content*
(FFT/Track: transcendental, wide dynamic range), two by *structural requirement* (Norm/Write:
additive headroom past ±1.0, float output format). None is a Q15 candidate; forcing Q15 on
them is the myopic trap in reverse.

## Stage 3 (Synth) — the only place a Q-island pays, dissected

The opt-in `--q15` kernel `osc_simd_q15_segment` (`core/port/host/oscillator_simd.c`) is the
sole contiguous integer island on desktop. Walking its sustain block samples (lines ~239–258):

| Sub-step | Domain | Class | Why |
|----------|--------|-------|-----|
| Phase NCO | uint64 / uint32 (uq32) integer fwd-difference | **Q-ISLAND ✓ (shipped)** | `spectral_phase_nco{,8}` — exact integer adds, no float. Prerequisite that made the island possible (Q5a/Q5b/Bv). |
| Waveform eval (4 algebraic + sine LUT) | 8×Q15 in one 128-bit reg | **Q-ISLAND ✓ (shipped)** | `osc_q15_pack8_eval` — the double-lane-packing win (8×int16 vs 4×float32). Sine joined at B1 (PASS217). |
| **Widen Q15→float** | int16 → float | **boundary (unavoidable)** | The accumulator and output are float; the island *must* end here. 1-op sign-extend ×2, the minimal crossing. |
| Amp ramp × wave | float | **inherently float** | **B2 (PASS218) measured the alternative and declined it:** amp-in-Q15 `mulhrs` is 1.03–1.10× *slower* on the algebraic timbres and regresses amp precision ~35 dB. The float widen is already near the ceiling. |
| Accumulate into `dst` | float `+=` | **inherently float** | Cross-segment additive sum; partials exceed ±1.0 before norm, which Q15 would saturate. The float accumulator is required for headroom, not incidental. |
| Band-limited variant (PolyBLEP/additive/oversample) | float | **inherently float** | `osc_bandlimited_synth_segment` is documented "CPU float synthesis path only" (`core/spectral_osc_bandlimited.h:19`): fractional edge positions + oversampling headroom are float by nature. |
| Wavetable LUT read | q15_t → float via macro | **boundary (storage), not compute** | The q15 sine/wavetable LUT is a storage/transport boundary consumed through `SPECTRAL_SAMPLE_TO_FLOAT`; it is not a compute-in-Q15 hot path (Q1 finding). |

## Conclusions

1. **The island is exactly one contiguous region — `integer phase → 8×Q15 eval → widen` —
   and it is already shipped** (`--q15`, host-only). Its boundaries are *structural*: it
   starts where phase can be integer and ends where the additive/float-output reality forces
   float. B1 widened it to the last eligible timbre (sine); B2 proved it cannot extend past
   the widen. There is no un-captured contiguous island left on desktop.

2. **"Leverage Q15 on every path" is answered, with evidence:** 4 of 5 pipeline stages are
   inherently float (information content or structural headroom/format), and within the 5th
   the float boundary is at the widen. The non-myopic reading of the ask is therefore *"widen
   the one island's lane count,"* not *"convert more stages."*

3. **The remaining throughput lever is lane COUNT, not island EXTENT** — i.e. **Thread C**:
   16×Q15@256 on AVX2 doubles the packing of the *same* eval island. That is x86-only and
   unmeasurable on this NEON=128 dev Mac, so it is gated on x86 CI. It does not change this
   audit's boundaries; it scales the one island that already pays.

**Net:** the Q15-on-desktop coverage decision is closed and reviewable. Float stays the
default and the inherently-float majority of the pipeline; the opt-in `--q15` island covers
exactly the region where integer packing wins, and no further. See [[qtype-domain]],
[[minimal-decline-on-data]], [[faster-path-should-default]].
