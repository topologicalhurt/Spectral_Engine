/* spectral_q15.c - Q15 Fixed-Point DSP Implementation
 *
 * Scalar primitives are now inline in spectral_q15.h for zero call overhead.
 * This file provides only the bulk (vectorized) conversion routines.
 */
#include "spectral_q15.h"

/* Bulk Q31 to Q15 conversion with saturation
 *
 * Converts an array of Q31 values to Q15 by shifting right 16 and saturating.
 * On ARM with NEON, uses vectorized SIMD for 4x throughput.
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

void spectral_q31_to_q15_bulk(const q31_t* src, q15_t* dst, uint32_t count) {
    if (count > 0u && (!src || !dst)) return;
    uint32_t i = 0;
    uint32_t count4 = count & ~3U;

    while (i < count4) {
        int32x4_t q31_vec = vld1q_s32(&src[i]);
        int16x4_t q15_vec = vqshrn_n_s32(q31_vec, 16);
        vst1_s16(&dst[i], q15_vec);
        i += 4;
    }

    while (i < count) {
        dst[i] = spectral_q31_to_q15_sat(src[i]);
        i++;
    }
}

/* Bulk Q31 to Q15 with scaling (fused convert + amplitude in one pass) */
void spectral_q31_to_q15_scaled(const q31_t* src, q15_t* dst, uint32_t count, q15_t scale) {
    if (count > 0u && (!src || !dst)) return;
    uint32_t i = 0;
    uint32_t count4 = count & ~3U;

    int16x4_t scale_vec = vdup_n_s16(scale);

    while (i < count4) {
        int32x4_t q31_vec = vld1q_s32(&src[i]);
        int16x4_t q15_vec = vqshrn_n_s32(q31_vec, 16);
        int32x4_t scaled = vmull_s16(q15_vec, scale_vec);
        int16x4_t result = vqshrn_n_s32(scaled, 15);
        vst1_s16(&dst[i], result);
        i += 4;
    }

    while (i < count) {
        q15_t sample = spectral_q31_to_q15_sat(src[i]);
        dst[i] = spectral_mul_q15(sample, scale);
        i++;
    }
}

#else
/* Portable fallback for non-NEON platforms */

void spectral_q31_to_q15_bulk(const q31_t* src, q15_t* dst, uint32_t count) {
    if (count > 0u && (!src || !dst)) return;
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = spectral_q31_to_q15_sat(src[i]);
    }
}

void spectral_q31_to_q15_scaled(const q31_t* src, q15_t* dst, uint32_t count, q15_t scale) {
    if (count > 0u && (!src || !dst)) return;
    for (uint32_t i = 0; i < count; i++) {
        q15_t sample = spectral_q31_to_q15_sat(src[i]);
        dst[i] = spectral_mul_q15(sample, scale);
    }
}

#endif
