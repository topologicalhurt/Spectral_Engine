/* spectral_out_kernels.c (host profile) - normalization + Q15 stereo kernels.
 *
 * Build-selected for host/simulation builds. These are the profile-divergent
 * output bodies: float normalization uses the SIMDe
 * vector ops (spectral_vmaxmgv / spectral_vsmul); the Q15 normalize and Q15
 * mono->stereo use portable scalar code (host/sim has no CMSIS DSP or Cortex-M
 * unroll). The CMSIS / Cortex-M counterparts live in
 * arch/ref/spectral_out_kernels.c. The device-agnostic remainder of
 * spectral_out.c (file I/O, float mono->stereo) stays in core/. */
#include "spectral_io.h"
#include "spectral_q.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include "spectral_vector_ops.h"

float spectral_normalize_float(float* buffer, size_t len, float headroom) {
    if (!buffer || len == 0) return 0.0f;
    if (!spectral_is_finite_f32(headroom) || headroom < 0.0f) return 0.0f;
    if (!spectral_f32_span_finite(buffer, len)) return 0.0f;

    float max_amp = 0.0f;

    spectral_vmaxmgv(buffer, &max_amp, len);
    if (!spectral_is_finite_f32(max_amp) || max_amp < 0.0f) return 0.0f;
    if (max_amp > 0.0f) {
        float scale = headroom / max_amp;
        if (!spectral_is_finite_f32(scale)) return 0.0f;
        spectral_vsmul(buffer, scale, buffer, len);
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

    for (size_t i = 0; i < len; i++) {
        q15_t abs_val;
        if (buffer[i] == Q15_MIN) {
            abs_val = Q15_MAX;
        } else {
            abs_val = (buffer[i] < 0) ? (q15_t)-buffer[i] : buffer[i];
        }
        if (abs_val > max_val) max_val = abs_val;
    }

    /* Determine if normalization needed (prevent clipping) */
    int shift_amt = 0;
    if (max_val > Q15_MAX / 2) {
        /* Need to scale down - find shift amount */
        q15_t test = max_val;
        while (test > Q15_MAX / 2) {
            test >>= 1;
            shift_amt++;
        }

        for (size_t i = 0; i < len; i++) {
            buffer[i] >>= shift_amt;
        }
    }

    if (shift) *shift = shift_amt;
    return max_val;
}

void spectral_mono_to_stereo_q15(const q15_t* mono, q15_t* stereo, size_t num_frames) {
    if (!mono || !stereo || num_frames == 0 || num_frames > SIZE_MAX / 2u) return;

    SPECTRAL_UNROLL_4
    for (size_t i = 0; i < num_frames; i++) {
        size_t stereo_i = i * 2u;
        stereo[stereo_i]     = mono[i];
        stereo[stereo_i + 1u] = mono[i];
    }
}
