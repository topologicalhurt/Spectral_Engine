/* spectral_q15.c - Q15 Fixed-Point DSP Implementation
 *
 * Scalar primitives are inline in spectral_q15.h for zero call overhead.
 * This file provides only the bulk conversion routines.
 */
#include "spectral_q15.h"

/* Bulk Q30 -> Q15 conversion with per-sample master scaling.
 *
 * The ARM synthesis accumulator holds Q30 values (sums of Q15*Q15 MAC products),
 * so converting back to Q15 is a >>15 shift (the CMSIS Q15-MAC convention), NOT
 * >>16. The former code reused a Q31->Q15 (>>16) helper here, which halved every
 * output sample (-6 dB) and diverged from the float CPU backend that renders amp
 * directly (out += amp*osc) — caught by tests/arm_core, fixed in pass 145.
 * (A hand-written __ARM_NEON path here was removed in pass 140; it was
 * bit-identical scalar and dead for the NEON-less Cortex-M target.)
 */
void spectral_q30_to_q15_scaled(const q31_t* accum, q15_t* dst, uint32_t count, q15_t scale) {
    if (count > 0u && (!accum || !dst)) return;
    for (uint32_t i = 0; i < count; i++) {
        q15_t sample = spectral_ssat16(accum[i] >> 15); /* Q30 -> Q15, saturating */
        dst[i] = spectral_mul_q15(sample, scale);
    }
}

/* Wide q63 accumulator -> Q15. The accumulator holds the EXACT sum of Q15*Q15 (Q30)
 * products, so the additive mix is clipped exactly once here (proper mix headroom),
 * rather than per-MAC into a Q31 as the legacy path did. */
void spectral_q63_to_q15_scaled(const q63_t* accum, q15_t* dst, uint32_t count, q15_t scale) {
    if (count > 0u && (!accum || !dst)) return;
    for (uint32_t i = 0; i < count; i++) {
        q63_t v = accum[i] >> 15;            /* Q30 sum -> Q15 scale (exact) */
        q15_t sample = (v > Q15_MAX) ? Q15_MAX : (v < Q15_MIN) ? Q15_MIN : (q15_t)v;
        dst[i] = spectral_mul_q15(sample, scale);
    }
}
