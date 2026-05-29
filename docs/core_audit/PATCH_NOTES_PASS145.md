# Patch notes — Pass 145: fix ARM output amplitude (Q30 accumulator read as Q31, -6 dB)

## Problem

`spectral_arm32_process` accumulates Q30 values (sums of Q15*Q15 MAC products)
but converted them to Q15 output with `spectral_q31_to_q15_scaled`, which shifts
by 16 (Q31->Q15). The correct Q15-MAC convert-back is `>>15` (CMSIS convention).
The extra bit halved every sample: a single `amp=0.5` segment rendered at peak
~0.25 (-6 dB) — 6 dB quieter than the reference float CPU backend, which renders
amp directly (`out += amp*osc`, peak = amp). A cross-backend amplitude
inconsistency (AI_CANON §7).

Like the Pass-144 frequency bug, this was masked by the host sim: it reimplements
synthesis with its own int64 accumulator and normalizes output, so it never
exercised this conversion. Caught by `tests/arm_core` (the real process on host).

## Fix

Replace the misused `spectral_q31_to_q15_scaled` (>>16, one caller) and the dead
`spectral_q31_to_q15_bulk` (zero callers) with `spectral_q30_to_q15_scaled`
(`>>15`), named for the accumulator's actual Q30 format. Update the one synth
caller and its (previously "Q31 accumulator") comment.

## Verification

`tests/arm_core`: a nominal 1000 Hz / amp 0.5 segment now renders at peak 0.499
(was 0.250); frequency unchanged (990 Hz, Pass 144). The harness amplitude
assertion is now an exact ~0.5 (matching the float backend), not the placeholder
audible+non-clipping bound. `ctest` green; sim oracle unchanged (it never used
this conversion). Real-ARM cross builds get the same fix (format-correct,
architecture-independent); verify on-target when a toolchain is available.
