/* spectral_q.c - fixed-point Q-domain bulk conversions
 *
 * Scalar primitives are inline in spectral_q.h for zero call overhead.
 * This file provides only the bulk conversion routines.
 */
#include "spectral_q.h"

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
