# Patch notes — Pass 219: Q-island audit (refactor Thread B3)

## Scope

Thread B3 is the **anti-myopia deliverable**: a reviewable enumeration of every desktop
hot path, classifying each as a contiguous **Q-island candidate** vs **inherently float**,
so the maintainer's "leverage Q15 on every path that could benefit" becomes an
evidence-based coverage decision rather than a scattershot. Documentation only — no code.

New doc: **`docs/core_audit/QTYPE_ISLAND_AUDIT.md`**, grounded in the code (file:line
citations, not assumptions).

## Findings

The CLI pipeline has five timed stages: **FFT → Track → Synth → Norm → Write**.

- **Four of five are inherently float** — two by information content (FFT is a complex
  float transform; Track works on wide-dynamic-range magnitudes + atan2 phase + sub-bin
  parabolic peak interpolation, which Q15's ~92 dB SNR ceiling can't hold), two by
  structural requirement (Norm scans/scales the summed mix that exceeds ±1.0 pre-norm;
  Write is `SF_FORMAT_FLOAT` 32-bit float WAV — no int16 boundary to repack).
- **The one Q-island lives in Synth**, and it is already shipped: the `--q15` kernel's
  `integer phase NCO → 8×Q15 waveform eval → widen`. Its boundaries are *structural*: it
  begins where phase can be integer and **ends at the widen**, because the amp ramp +
  cross-segment accumulate need float headroom (additive partials exceed ±1.0) and the
  output is float. B1 (PASS217) widened it to the last eligible timbre (sine); B2 (PASS218)
  measured the attempt to push past the widen and declined it.

## Conclusions (the non-myopic reading)

1. The desktop Q15 island is **exactly one contiguous region and is fully captured** — no
   un-exploited contiguous island remains.
2. "Q15 everywhere" is the wrong target *with evidence*: the inherently-float majority is
   float by nature, and the island's float boundary is forced at the widen.
3. The only remaining throughput lever is **lane count, not island extent** — Thread C
   (16×Q15@256 on AVX2), which scales the *same* eval island and is x86-CI-gated (the dev
   Mac is NEON=128, already maximal). It does not move this audit's boundaries.

## Verification

Documentation-only; no source, build, or test change. Citations spot-checked against the
tree (`spectral_out.c:202/215`, `spectral_io.h:62`, `spectral_peak_track.c:1408`,
`spectral_analysis_full.c:44`, `spectral_osc_bandlimited.h:19`,
`core/port/host/oscillator_simd.c` sustain block).

## Status

**B3 — LANDED.** With B1 (sine routed), B2 (amp-in-Q declined), and B3 (audit), **Thread B
is complete**. The Q15-on-desktop coverage decision is closed and reviewable. The remaining
refactor work is **Thread C** (SIMD max-width, x86-CI-gated) per the A→B→C order — pending
maintainer go-ahead and an x86 build/CI to validate the wider tiers.
