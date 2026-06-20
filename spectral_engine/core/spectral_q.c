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
        /* Q2.30 MAC sum -> Q15: arithmetic >>15, then a BRANCHLESS saturating narrow. ssat16 on a
         * q31 (one SSAT on the M7) replaces a q63 ternary clamp whose 64-bit compares GCC lowers to
         * branches (~30% of this function). The >>15 result fits q31 at <=512 active voices (worst
         * sum ~2^39 -> 2^24), so the q31 narrow is exact and this is byte-identical over the design
         * range; ssat16 still clamps the out-of-Q15 magnitude exactly as the q63 clamp did. */
        q15_t sample = spectral_ssat16((q31_t)(accum[i] >> 15));
        dst[i] = spectral_mul_q15(sample, scale);
    }
}
