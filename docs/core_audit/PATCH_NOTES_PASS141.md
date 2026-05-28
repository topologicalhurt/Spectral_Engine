# Patch notes — Pass 141: ARM A1 cleanups (LUT-reader dedup, uq32_t); freq-units verified

## LUT-reader dedup

`synth_core_m7`'s unrolled body used a local `spectral_osc_lookup`; the scalar
tail and fade paths used the canonical `spectral_lut_sin` (spectral_lut.h). At
the default `SPECTRAL_OPT_LEVEL=1` the two are bit-identical (same table index,
same Q8 fractional interpolation — verified by working the frac math and by the
oracle staying green). They diverged only at `OPT_LEVEL>=2`, where `osc_lookup`
dropped to nearest-neighbour while the tail kept interpolating — a discontinuity
at the 4-sample unroll boundary.

Removed `spectral_osc_lookup`; the body now calls `spectral_lut_sin` like the
tail (AI_CANON §7/§17 — no duplicated formula). At `OPT_LEVEL>=2` the body now
interpolates consistently with the tail; the nearest-neighbour opt-out was buggy
(inconsistent) and unproven (AI_CANON §11) — re-add as a documented,
consistently-applied mode behind a benchmark if ever wanted.

## uq32_t typedef

`spectral_q15.h` declared `typedef uint32_t uq32_t;` three times (copy-paste).
Reduced to one. No behavior change.

## freq-units: verified correct (no change)

The standing concern that `freq_inc = freq_q88 * (2^24/sr)` was dimensionally off
(predicting pitch ~sr/2pi too low) was tested, not assumed: a 430.66 Hz input
renders at 430.00 Hz output (FFT of the sim output). The scaling is correct; the
earlier hand-analysis mis-set a factor. The relationship is non-obvious and
spread across files (omega -> freq_q88 -> freq_inc_scale_q24 -> phase), so it is a
Phase B units-contract *test* candidate (AI_CANON §9), not a fix.

## Verification

`make simulate` clean; interim oracle (tests/arm_oracle) green on all 6 cases —
behavior-preserving at the default build.

## Follow-up surfaced

`spectral_arm32_validate_segment_data` is defined but never called (pre-existing
-Wunused-function): load-path boundary validation (overflow / monotonic ordering
/ output_len bound / active-count bound) is not wired in. Addressed in pass 142.
