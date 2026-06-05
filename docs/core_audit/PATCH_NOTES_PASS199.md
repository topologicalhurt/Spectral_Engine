# Patch notes — Pass 199: Band-limited ("musical") oscillator quality modes

## Scope

User-requested, outside the numbered optimisation track, greenlighting the
PASS198 *investigate-only* harsh-timbre diagnosis:

> "PolyBLEP/BLEP step+slope correction, a Nyquist-limited harmonic count per
> partial, 2–4× oversample-then-decimate. Do this but keep the implementation in
> a separate file and ensure everything wired correctly."

**Outcome:** a new self-contained module `core/spectral_osc_bandlimited.{h,c}`
adds three opt-in anti-aliasing renderers, wired end-to-end (CLI → pipeline → CPU
float synth) so the user can actually audition them. The **default build output
is bit-identical** (default mode is NAIVE, which never enters the new code).

## Why (the harsh-timbre root cause, from PASS198)

Resynthesis renders each detected partial as the chosen timbre *at that partial's
frequency*. The naive waveforms are point-sampled, so any harmonic a hard edge or
sharp cusp generates above Nyquist folds back as inharmonic aliasing — the
"harsh" timbres (square, saw, pwm, quantized, asin). The three modes attack that.

## The new module — `core/spectral_osc_bandlimited.{h,c}`

Process-global quality state mirroring `osc_set_dispatch()` in `oscillator.c`:
`osc_set_quality()` / `osc_get_quality()`, plus `osc_quality_name()` and
`osc_quality_parse()` (accepts `naive|polyblep|additive|oversample`, aliases
`blep|add|os`, or an integer `0..3`).

```text
enum SpectralOscQuality { NAIVE=0 (default), POLYBLEP=1, ADDITIVE=2, OVERSAMPLE=3 }
```

`int osc_bandlimited_synth_segment(dst, lp, timbre, quality)` returns **1** if it
fully rendered the segment, **0** if the timbre/quality pair is unsupported (the
caller then falls through to the naive scalar path). It reuses the canonical
per-sample bookkeeping from `synth_segment_scalar()`: `spectral_segment_phase_at_cubic_f32`,
`spectral_segment_amp_at_f32`, and the same Hann `fade_envelope` — so each mode is
a drop-in quality swap, not a different envelope/phase model.

### Mode 1 — POLYBLEP / polyBLAMP edge correction (cheap, per-sample, no alloc)
Adds a 2nd-order PolyBLEP step residual at each amplitude discontinuity and a
polyBLAMP slope residual at each corner, sized by the instantaneous normalized
frequency `dt = |dθ/dj|/2π` (from the cubic phase derivative
`alpha + 2·c2·j + 3·c3·j²`, clamped `[1e-6, 0.5]`). Per our specific naive
waveform orientations (`t = rads/2π + 0.5 ∈ [0,1)`):
- saw `1-2t`  (+2 jump at t=0):            `+ poly_blep(t)`
- square `(t>0.5)?1:-1`:                    `- poly_blep(t) + poly_blep(frac(t+0.5))`
- pwm `(t<w)?1:-1` (0<w<1):                 `+ poly_blep(t) - poly_blep(frac(t-w))`
- triangle (corners at t=0,0.5; slope ±4): `+ 4·dt·(poly_blamp(t) - poly_blamp(frac(t+0.5)))`
- sine / parabola → rendered naive (continuous enough)
- asin / quantized → **return 0** (no BLEP form; caller falls back to naive)

### Mode 2 — ADDITIVE, Nyquist-capped Fourier reconstruction
Per sample, harmonic count `N = floor(0.5/dt)` (highest harmonic below Nyquist),
clamped `[1, 512]`. Series summed with a **Chebyshev recurrence** for
`sin(kθ)/cos(kθ)` — one `sinf`/`cosf` per sample, then add-only per harmonic:
- saw      `(2/π) Σ_{k=1..N} ((-1)^k/k) sin(kθ)`
- square   `(4/π) Σ_{k odd≤N} (1/k) sin(kθ)`
- triangle `(8/π²) Σ_{k odd≤N} (1/k²) cos(kθ)`
- parabola `2/3 − (4/π²) Σ_{k=1..N} ((-1)^k/k²) cos(kθ)`
- sine     `sin(θ)`
- pwm / asin / quantized → **return 0** (no compact series; caller falls back)

Exactly band-limited by construction (it never synthesizes a harmonic above
Nyquist).

### Mode 3 — OVERSAMPLE + windowed-sinc decimation (universal)
Renders the naive waveform 4× into a heap buffer (`malloc`, freed per call),
convolves with a zero-phase windowed-sinc low-pass (cutoff = original Nyquist,
`L=65` taps, Hann window, DC-normalized, built on the stack per call → thread
safe), and decimates centered on the integer-aligned oversample index (no
fractional delay). The **only** mode that tames the cusp/staircase timbres
(`asin`, `quantized`) which have no closed-form BLEP or compact series. Guards:
`len > 4M` → return 0 (fall back); `malloc` fail → return 0.

### Coverage matrix
```text
            saw  square  tri  pwm  parabola  sine  asin  quant
POLYBLEP     ✓     ✓      ✓    ✓    naive    naive   0     0
ADDITIVE     ✓     ✓      ✓    0     ✓        ✓      0     0
OVERSAMPLE   ✓     ✓      ✓    ✓     ✓        ✓      ✓     ✓
```
(`0` = returns 0 → caller renders the naive scalar path; still audible, just not
band-limited.)

## Wiring

1. **`core/oscillator.c`** — `timbre_synth_segment()`: after the `len==0` guard,
   before the SIMD/scalar dispatch, query `osc_get_quality()`; if non-NAIVE and
   `osc_bandlimited_synth_segment()` returns 1, return. NAIVE returns 0 → the
   existing point-sampled path runs **unchanged** (bit-identical default).
2. **`cmake/source-manifest.cmake`** — `spectral_osc_bandlimited.c` added to
   `SPECTRAL_SOURCES_CORE` (linked by every target that builds `oscillator.c`).
3. **CLI** — `-q` / `--quality <mode>` flag (`spectral_cli.{h,c}`): new
   `opts->osc_quality` field (default NAIVE), parsed via `osc_quality_parse`,
   usage text added.
4. **Pipeline** — `run_synthesis()` (`spectral_cli_pipeline.c`) calls
   `osc_set_quality(opts->osc_quality)` before dispatch. Because quality is a
   CPU-float-path feature (the GPU/Metal/CUDA and Q15-native backends ignore it),
   a non-NAIVE mode **forces `BACKEND_CPU`** and logs it, so the user reliably
   auditions the chosen mode.

## Why this is golden-safe

Default quality is NAIVE → `timbre_synth_segment` short-circuits before reaching
any new code; the only added work on the default path is a static-int load + a
branch that is not taken. Sample output is therefore unchanged. This is an
audition/quality feature, **not** a golden contract (documented in the header).

## Verification

```text
- Five production targets build clean (desktop, simulate, simulate_daisy,
  embedded_arm, embedded_arm_float) — only the pre-existing benign
  -mavx2 / -mno-avx512f notes.
- ctest 4/4 PASSED (arm32_process_correctness, core_contracts, core_guarantees,
  core_guarantees_drift).
- Flag plumbing is inert for NAIVE: on the deterministic CPU single-thread path,
  an unflagged render is BYTE-IDENTICAL to `-q naive` (cmp clean). (The default
  Metal/auto backend is nondeterministic run-to-run on its own — a pre-existing
  property of the parallel/GPU path — so bit-identity is asserted on the CPU path,
  where the naive branch is provably the same code with/without the flag.)
- All three modes are reachable (log: "Oscillator quality: <mode> (... forcing
  CPU backend)", Backend used: CPU) and measurably reduce aliasing-region energy
  on a 220 Hz harmonic input:
      square  HF(>12k):  naive 1.07%  -> polyblep 0.30%, additive 0.68%, oversample 0.64%
      asin    HF(>12k):  naive 2.65%  -> oversample 1.53%
- Fallback timbre/mode pairs (polyblep+asin/quantized, additive+pwm/quantized)
  render finite, non-silent audio (rc=0, peak 0.95) via the naive scalar path.
```

## Status

Three band-limited oscillator quality modes **implemented** in a separate module
and **wired** CLI→pipeline→CPU-float-synth; default **bit-identical**; all modes
auditioned and shown to cut aliasing-region energy. The feature affects the CPU
float synthesis path only; GPU and Q15-native backends are untouched.
