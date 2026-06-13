/* spectral_osc_q15.h - production Q15 oscillator waveform evaluators.
 *
 * The opt-in Q15 *compute* twin of the float L0 formulas in
 * spectral_osc_formulas.h.  Q3 of QTYPE_DOMAIN_PLAN.md adds a per-path,
 * opt-in Q15 compute domain for throughput-bound L1 oscillator paths; this
 * header is the single source of those evaluators, shared by production
 * (spectral_oscillator.c) and the precision/parity CTests (no drift between what we
 * measure and what we ship).  Float stays the DEFAULT domain — these are
 * reached only when a timbre's Q15 bit is explicitly set (spectral_osc_set_q15_enable).
 *
 * SCOPE: this evaluates the WAVEFORM in Q15.  Phase accumulation stays in
 * float (the cubic NCO); QTYPE_DOMAIN_PLAN.md treats integer-NCO phase
 * resolution as an orthogonal, deferred axis.  So the precision these carry is
 * exactly the waveform-eval floor characterized by the q15_compute_precision
 * CTest: sine -85.1, saw -91.5, square -90.0, triangle -92.6,
 * parabola -91.0 dBFS vs the float reference.
 *
 * Each evaluator mirrors the matching spectral_osc_<t>(rads) formula; the
 * five here are the algebraically-clean / LUT throughput-bound set
 * (quantized/pwm/asin are width-/transcendental-shaped — not Q15 candidates).
 */
#ifndef SPECTRAL_OSC_Q15_H
#define SPECTRAL_OSC_Q15_H

#include "spectral_q.h"
#include "spectral_lut.h"
#include "spectral_consts.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Contract version for the Q15 oscillator waveform evaluators + boundary
 * helpers in this header.  Bump on ANY change to spectral_osc_q15_phase_from_rads,
 * spectral_osc_q15_init_sine_lut, or any spectral_osc_q15_<timbre> evaluator.
 * Production consumers (spectral_oscillator.c scalar-Q15 and arch/simd/spectral_oscillator_simd.c
 * pack8) pin this with _Static_assert, so a silent contract edit fails their
 * build -- the Q15 twin of SPECTRAL_OSC_FORMULAS_VERSION (spectral_osc_formulas.h).
 *
 * v1: five evaluators (sine/saw/square/triangle/parabola), the rads->Q15
 *   boundary helper, and the full-scale (Q15_MAX) sine-LUT builder. */
#define SPECTRAL_OSC_Q15_VERSION 1

/* float rads in [-pi, pi) -> signed Q15 phase (rads/pi -> [-1, 1)).
 *
 * This is the lone float->Q boundary of the Q15 oscillator; it stays OUTSIDE
 * the Q-domain region below.  It is NOT the spectral_q.h PHASE_RAD_TO_Q15
 * macro: that maps [0,2pi) with a -0.5 center (phase 0 -> Q15 -16384), whereas
 * the L0 waveform formulas need rads/pi centered so phase 0 -> 0.  Different
 * convention, dedicated helper. */
static inline q15_t spectral_osc_q15_phase_from_rads(float rads) {
    float n = rads * SPECTRAL_INV_PI;     /* [-1, 1) */
    if (n >= 1.0f) n = 0.99996948f;       /* clamp below +1.0 so the cast stays in range */
    if (n < -1.0f) n = -1.0f;
    return (q15_t)(n * SPECTRAL_Q15_SCALE);
}

/* Build the gain-matched FULL-SCALE (peak Q15_MAX) sine table for the Q15
 * oscillator.  Deliberately NOT spectral_lut_init_sine()'s table: production's
 * 32700 LUT carries a linear-interp-overflow headroom that is a ~-0.02 dB
 * intentional GAIN, which would read as an amplitude error against the float
 * path.  Full-scale keeps the Q15 sine amplitude matched to float so the only
 * residual is quantization, not gain. */
static inline void spectral_osc_q15_init_sine_lut(q15_t* lut) {
    for (uint32_t i = 0; i <= SPECTRAL_OSC_LUT_SIZE; i++) {
        float phase = (float)i / (float)SPECTRAL_OSC_LUT_SIZE * SPECTRAL_TWO_PI;
        lut[i] = (q15_t)(sinf(phase) * (float)Q15_MAX);
    }
    lut[0] = 0;
    lut[SPECTRAL_OSC_LUT_SIZE / 2] = 0;
    lut[SPECTRAL_OSC_LUT_SIZE / 4] = Q15_MAX;
    lut[3 * SPECTRAL_OSC_LUT_SIZE / 4] = (q15_t)(-Q15_MAX);
}

/* Pure Q15 waveform evaluators.  `pq` is the signed Q15 phase (rads/pi).  No
 * float/double appears between the markers — the q_domain_contract CTest
 * enforces it; conversions stay in the boundary helpers above. */
// SPECTRAL_Q_DOMAIN BEGIN
static inline q15_t spectral_osc_q15_sine(q15_t pq, const q15_t* sine_lut) {
    /* sin(rads) = sin(pi*pq/32768); the LUT computes sin(2*pi*u/65536) for a
     * uq16 index u.  (uint16_t)pq supplies that index directly (the 2*pi alias
     * makes the signed->unsigned reinterpret exact). */
    return spectral_lut_sin((uq16_t)(uint16_t)pq, sine_lut);
}
static inline q15_t spectral_osc_q15_saw(q15_t pq) {
    /* osc_saw(rads) = -rads*INV_PI = -(pq/32768).  Saturating negate. */
    return spectral_qsub16(Q15_ZERO, pq);
}
static inline q15_t spectral_osc_q15_square(q15_t pq) {
    /* osc_square(rads) = sign(rads). */
    return (pq > 0) ? Q15_MAX : Q15_MIN;
}
static inline q15_t spectral_osc_q15_triangle(q15_t pq) {
    /* osc_triangle(rads) = 1 - 2*|rads|*INV_PI = 1 - |pq|/16384.
     * value*32768 = 32768 - 2*|pq|; saturate into Q15. */
    q31_t a = (pq < 0) ? -(q31_t)pq : (q31_t)pq;
    return spectral_ssat16((q31_t)Q15_MAX - (a << 1));
}
static inline q15_t spectral_osc_q15_parabola(q15_t pq) {
    /* osc_parabola(rads) = 1 - (rads*INV_PI)^2 = 1 - (pq/32768)^2.
     * mul_q15(pq,pq) is (pq/32768)^2 in Q15. */
    return spectral_ssat16((q31_t)Q15_MAX - (q31_t)spectral_mul_q15(pq, pq));
}
// SPECTRAL_Q_DOMAIN END

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_OSC_Q15_H */
