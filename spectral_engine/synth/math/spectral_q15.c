/* spectral_q15.c - Q15 Fixed-Point DSP Implementation
 *
 * Scalar primitives are inline in spectral_q15.h for zero call overhead.
 * This file provides only the bulk conversion routines.
 */
#include "spectral_q15.h"

/* Bulk Q31 -> Q15 conversion with saturation.
 *
 * Portable scalar implementation. A hand-written __ARM_NEON path was removed in
 * pass 140: it only ever compiled on desktop ARM (Cortex-M, the embedded target,
 * has no NEON), it was bit-identical to this scalar form
 * (vqshrn_n_s32(x,16) == arithmetic (x>>16) + signed saturate;
 *  vmull_s16 + vqshrn_n(.,15) == smulbb + (>>15) + saturate),
 * and this conversion is not on the embedded hot path. The campaign keeps no
 * unproven SIMD (AI_CANON 11). If profiling shows it hot, re-add via SIMDe (the
 * portable SIMD layer used in oscillator_simd.c) behind a benchmark, not raw
 * architecture intrinsics.
 */
void spectral_q31_to_q15_bulk(const q31_t* src, q15_t* dst, uint32_t count) {
    if (count > 0u && (!src || !dst)) return;
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = spectral_q31_to_q15_sat(src[i]);
    }
}

/* Bulk Q31 -> Q15 with scaling (fused convert + amplitude in one pass) */
void spectral_q31_to_q15_scaled(const q31_t* src, q15_t* dst, uint32_t count, q15_t scale) {
    if (count > 0u && (!src || !dst)) return;
    for (uint32_t i = 0; i < count; i++) {
        q15_t sample = spectral_q31_to_q15_sat(src[i]);
        dst[i] = spectral_mul_q15(sample, scale);
    }
}
