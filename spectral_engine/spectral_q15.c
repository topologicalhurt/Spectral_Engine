/* spectral_q15.c - Q15 Fixed-Point DSP Implementation
 * 
 * Provides saturating arithmetic for Q15/Q31 fixed-point math.
 * 
 * On ARM Cortex-M with DSP extension (__ARM_FEATURE_DSP), these compile
 * to single-cycle hardware instructions (QADD, QSUB, SSAT, SMLAD, etc.).
 * 
 * On desktop/emulator builds, the portable fallbacks provide identical
 * mathematical behavior for testing embedded algorithms.
 */
#include "spectral_q15.h"

#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP
#include <arm_acle.h>

q15_t spectral_qadd16(q15_t a, q15_t b) {
    return (q15_t)__qadd16((int32_t)a, (int32_t)b);
}

q15_t spectral_qsub16(q15_t a, q15_t b) {
    return (q15_t)__qsub16((int32_t)a, (int32_t)b);
}

q31_t spectral_qadd32(q31_t a, q31_t b) {
    return __qadd(a, b);
}

q31_t spectral_qsub32(q31_t a, q31_t b) {
    return __qsub(a, b);
}

q15_t spectral_ssat16(q31_t val) {
    return (q15_t)__ssat(val, 16);
}

q15_t spectral_q31_to_q15_sat(q31_t q) {
    return (q15_t)__ssat(q >> 16, 16);
}

q31_t spectral_smlad(q31_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1) {
    uint32_t packed_a = ((uint32_t)(uint16_t)a1 << 16) | (uint16_t)a0;
    uint32_t packed_b = ((uint32_t)(uint16_t)b1 << 16) | (uint16_t)b0;
    return __smlad(packed_a, packed_b, acc);
}

q31_t spectral_smulbb(q15_t a, q15_t b) {
    return __smulbb(a, b);
}

#else
/* Portable C Fallback */

q15_t spectral_qadd16(q15_t a, q15_t b) {
    int32_t sum = (int32_t)a + (int32_t)b;
    return (sum > Q15_MAX) ? Q15_MAX : (sum < Q15_MIN) ? Q15_MIN : (q15_t)sum;
}

q15_t spectral_qsub16(q15_t a, q15_t b) {
    int32_t diff = (int32_t)a - (int32_t)b;
    return (diff > Q15_MAX) ? Q15_MAX : (diff < Q15_MIN) ? Q15_MIN : (q15_t)diff;
}

q31_t spectral_qadd32(q31_t a, q31_t b) {
    int64_t sum = (int64_t)a + (int64_t)b;
    return (sum > Q31_MAX) ? Q31_MAX : (sum < Q31_MIN) ? Q31_MIN : (q31_t)sum;
}

q31_t spectral_qsub32(q31_t a, q31_t b) {
    int64_t diff = (int64_t)a - (int64_t)b;
    return (diff > Q31_MAX) ? Q31_MAX : (diff < Q31_MIN) ? Q31_MIN : (q31_t)diff;
}

q15_t spectral_ssat16(q31_t val) {
    return (val > Q15_MAX) ? Q15_MAX : (val < Q15_MIN) ? Q15_MIN : (q15_t)val;
}

q15_t spectral_q31_to_q15_sat(q31_t q) {
    q31_t shifted = q >> 16;
    return (shifted > Q15_MAX) ? Q15_MAX : (shifted < Q15_MIN) ? Q15_MIN : (q15_t)shifted;
}

q31_t spectral_smlad(q31_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1) {
    return acc + ((q31_t)a0 * b0) + ((q31_t)a1 * b1);
}

q31_t spectral_smulbb(q15_t a, q15_t b) {
    return (q31_t)a * (q31_t)b;
}

#endif

/* Higher-Level Q15 Operations */
q15_t spectral_mul_q15(q15_t a, q15_t b) {
    return spectral_ssat16(spectral_smulbb(a, b) >> 15);
}

q31_t spectral_mac_q15(q31_t acc, q15_t a, q15_t b) {
    return spectral_qadd32(acc, spectral_smulbb(a, b));
}

q15_t spectral_scale_q15(q15_t sample, q15_t amplitude) {
    return spectral_mul_q15(sample, amplitude);
}

/* Bulk Q31 to Q15 conversion with saturation
 * 
 * Converts an array of Q31 values to Q15 by shifting right 16 and saturating.
 * On ARM with NEON, uses vectorized SIMD for 4x throughput.
 * 
 * Parameters:
 *   src    - Input Q31 array
 *   dst    - Output Q15 array
 *   count  - Number of elements to convert
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

void spectral_q31_to_q15_bulk(const q31_t* src, q15_t* dst, uint32_t count) {
    /* Process 4 elements at a time with NEON */
    uint32_t i = 0;
    uint32_t count4 = count & ~3U;
    
    while (i < count4) {
        /* Load 4 Q31 values */
        int32x4_t q31_vec = vld1q_s32(&src[i]);
        
        /* Shift right by 16 to get Q15 range, with narrowing saturation */
        int16x4_t q15_vec = vqshrn_n_s32(q31_vec, 16);
        
        /* Store 4 Q15 values */
        vst1_s16(&dst[i], q15_vec);
        
        i += 4;
    }
    
    /* Handle remaining 0-3 elements */
    while (i < count) {
        dst[i] = spectral_q31_to_q15_sat(src[i]);
        i++;
    }
}

/* Bulk Q31 to Q15 with scaling (for amplitude application)
 * 
 * Converts Q31 to Q15 and applies a Q15 scale factor in one pass.
 * Formula: dst[i] = saturate_q15((src[i] >> 16) * scale >> 15)
 */
void spectral_q31_to_q15_scaled(const q31_t* src, q15_t* dst, uint32_t count, q15_t scale) {
    uint32_t i = 0;
    uint32_t count4 = count & ~3U;
    
    /* Duplicate scale factor to all lanes */
    int16x4_t scale_vec = vdup_n_s16(scale);
    
    while (i < count4) {
        /* Load and convert to Q15 range */
        int32x4_t q31_vec = vld1q_s32(&src[i]);
        int16x4_t q15_vec = vqshrn_n_s32(q31_vec, 16);
        
        /* Multiply by scale factor (Q15 * Q15 -> Q30, shift to Q15) */
        int32x4_t scaled = vmull_s16(q15_vec, scale_vec);
        int16x4_t result = vqshrn_n_s32(scaled, 15);
        
        vst1_s16(&dst[i], result);
        i += 4;
    }
    
    /* Scalar cleanup */
    while (i < count) {
        q15_t sample = spectral_q31_to_q15_sat(src[i]);
        dst[i] = spectral_mul_q15(sample, scale);
        i++;
    }
}

#else
/* Portable fallback for non-NEON platforms */

void spectral_q31_to_q15_bulk(const q31_t* src, q15_t* dst, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = spectral_q31_to_q15_sat(src[i]);
    }
}

void spectral_q31_to_q15_scaled(const q31_t* src, q15_t* dst, uint32_t count, q15_t scale) {
    for (uint32_t i = 0; i < count; i++) {
        q15_t sample = spectral_q31_to_q15_sat(src[i]);
        dst[i] = spectral_mul_q15(sample, scale);
    }
}

#endif

