/* spectral_out_kernels.c (embedded profile) - normalization + Q15 stereo kernels.
 *
 * Build-selected for bare-metal Cortex-M builds. These are the profile-divergent
 * bodies extracted from core/spectral_out.c: float normalization is portable
 * scalar (no host SIMDe vector ops on embedded); the Q15 normalize and Q15
 * mono->stereo use CMSIS-DSP / Cortex-M code when the capability is present,
 * falling back to scalar otherwise. The host (SIMDe) counterpart lives in
 * arch/simd/spectral_out_kernels.c. The SPECTRAL_USE_CMSIS and
 * SPECTRAL_ARM_M7 branches below are capability gates within the embedded port,
 * inert (scalar) on any build lacking the capability. */
#include "spectral_io.h"
#include "spectral_q15.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include <math.h>

#if SPECTRAL_USE_CMSIS
#include "arm_math.h"
#endif

float spectral_normalize_float(float* buffer, size_t len, float headroom) {
    if (!buffer || len == 0) return 0.0f;
    if (!spectral_is_finite_f32(headroom) || headroom < 0.0f) return 0.0f;
    if (!spectral_f32_span_finite(buffer, len)) return 0.0f;

    float max_amp = 0.0f;

    for (size_t i = 0; i < len; i++) {
        float a = fabsf(buffer[i]);
        if (a > max_amp) max_amp = a;
    }
    if (!spectral_is_finite_f32(max_amp) || max_amp < 0.0f) return 0.0f;
    if (max_amp > 0.0f) {
        float scale = headroom / max_amp;
        if (!spectral_is_finite_f32(scale)) return 0.0f;
        SPECTRAL_UNROLL_4
        for (size_t i = 0; i < len; i++) {
            buffer[i] *= scale;
        }
    }

    return max_amp;
}

q15_t spectral_normalize_q15(q15_t* buffer, size_t len, int* shift) {
    if (shift) *shift = 0;
    if (!buffer || len == 0) {
        return 0;
    }
    if (len > (size_t)UINT32_MAX) {
        return 0;
    }

    /* Find maximum absolute value */
    q15_t max_val = 0;

#if SPECTRAL_USE_CMSIS
    uint32_t len_u32 = (uint32_t)len;
    uint32_t max_idx;
    arm_absmax_q15(buffer, len_u32, &max_val, &max_idx);
#else
    for (size_t i = 0; i < len; i++) {
        q15_t abs_val;
        if (buffer[i] == Q15_MIN) {
            abs_val = Q15_MAX;
        } else {
            abs_val = (buffer[i] < 0) ? (q15_t)-buffer[i] : buffer[i];
        }
        if (abs_val > max_val) max_val = abs_val;
    }
#endif

    /* Determine if normalization needed (prevent clipping) */
    int shift_amt = 0;
    if (max_val > Q15_MAX / 2) {
        /* Need to scale down - find shift amount */
        q15_t test = max_val;
        while (test > Q15_MAX / 2) {
            test >>= 1;
            shift_amt++;
        }

        /* Apply shift - CMSIS arm_shift_q15 uses SIMD on M7 */
#if SPECTRAL_USE_CMSIS
        arm_shift_q15(buffer, -shift_amt, buffer, len_u32);
#else
        for (size_t i = 0; i < len; i++) {
            buffer[i] >>= shift_amt;
        }
#endif
    }

    if (shift) *shift = shift_amt;
    return max_val;
}

void spectral_mono_to_stereo_q15(const q15_t* mono, q15_t* stereo, size_t num_frames) {
    if (!mono || !stereo || num_frames == 0 || num_frames > SIZE_MAX / 2u) return;

#if SPECTRAL_ARM_M7 && defined(__ARM_FEATURE_DSP)
    size_t pairs = num_frames & ~(size_t)1;
    size_t i = 0;
    for (; i < pairs; i += 2) {
        size_t stereo_i = i * 2u;
        stereo[stereo_i]     = mono[i];
        stereo[stereo_i + 1u] = mono[i];
        stereo[stereo_i + 2u] = mono[i + 1u];
        stereo[stereo_i + 3u] = mono[i + 1u];
    }
    if (i < num_frames) {
        size_t stereo_i = i * 2u;
        stereo[stereo_i]     = mono[i];
        stereo[stereo_i + 1u] = mono[i];
    }
#else
    SPECTRAL_UNROLL_4
    for (size_t i = 0; i < num_frames; i++) {
        size_t stereo_i = i * 2u;
        stereo[stereo_i]     = mono[i];
        stereo[stereo_i + 1u] = mono[i];
    }
#endif
}
