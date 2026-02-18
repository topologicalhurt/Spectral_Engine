/* spectral_osc_formulas.h - Canonical Synthesis Math Contract
 *
 * Single source of truth for all oscillator math:
 *   - Phase normalization
 *   - Pade [5/4] sine approximation
 *   - 8 waveform generators (sine, saw, square, triangle, asin, parabola, quantized, pwm)
 *   - Fade envelope (Hann-window ramp)
 *
 * CPU and CUDA backends include this header directly.
 * Metal MSL backend must match these formulas exactly; constants are injected
 * with SPECTRAL_STR(...) in oscillator.c.
 *
 * IMPORTANT: Any formula change here MUST be mirrored in the Metal shader string
 * in oscillator.c. Run `make parity-test` to verify cross-backend consistency.
 */
#ifndef SPECTRAL_OSC_FORMULAS_H
#define SPECTRAL_OSC_FORMULAS_H

#include "spectral_consts.h"
#include <math.h>

/* Dual-compile: C inline or CUDA device inline */
#ifdef __CUDACC__
#define OSC_FORMULA_FUNC __device__ __forceinline__
#else
#define OSC_FORMULA_FUNC static inline
#endif

/* Phase normalization (canonical formula; all backends must match).
 * Maps arbitrary phase to [-pi, pi):
 *   norm = p / (2*pi)
 *   result = 2*pi * (norm - floor(norm) - 0.5) */
OSC_FORMULA_FUNC float spectral_normalize_phase(float p) {
    float norm = p * SPECTRAL_INV_TWO_PI;
    return SPECTRAL_TWO_PI * (norm - floorf(norm) - 0.5f);
}

/* Padé [5/4] sine approximation (canonical fast sine).
 * Max error ~1e-5 vs sinf(). Single divide, no branch.
 * Input: any float (range-reduced internally to [-pi, pi]). */
OSC_FORMULA_FUNC float spectral_fast_sin_inline(float x) {
    x = x - SPECTRAL_TWO_PI * floorf(x * SPECTRAL_INV_TWO_PI + 0.5f);
    float x2 = x * x;
    float num = x * (1.0f - x2 * (SPECTRAL_PADE_SIN_C1 - x2 * SPECTRAL_PADE_SIN_C2));
    float den = 1.0f + x2 * SPECTRAL_PADE_SIN_C3;
    return num / den;
}

/* Waveform generators.
 * All take rads in [-pi, pi) (output of spectral_normalize_phase) and
 * an optional width parameter (used by quantized and pwm only). */

OSC_FORMULA_FUNC float spectral_osc_sine(float rads, float width) {
    (void)width;
    return spectral_fast_sin_inline(rads);
}

OSC_FORMULA_FUNC float spectral_osc_saw(float rads, float width) {
    (void)width;
    return rads * -SPECTRAL_INV_PI;
}

OSC_FORMULA_FUNC float spectral_osc_square(float rads, float width) {
    (void)width;
    return (rads > 0.0f) ? 1.0f : -1.0f;
}

OSC_FORMULA_FUNC float spectral_osc_triangle(float rads, float width) {
    (void)width;
    return (1.0f - fabsf(rads) * SPECTRAL_INV_PI) * 2.0f - 1.0f;
}

OSC_FORMULA_FUNC float spectral_osc_asin(float rads, float width) {
    (void)width;
    return asinf(rads * SPECTRAL_INV_PI);
}

OSC_FORMULA_FUNC float spectral_osc_parabola(float rads, float width) {
    (void)width;
    return 1.0f - rads * rads * SPECTRAL_INV_PI_SQ;
}

OSC_FORMULA_FUNC float spectral_osc_quantized(float rads, float width) {
    if (width <= 0.0f) return 0.0f;
    float inv_w = 1.0f / width;
    return (float)(int)(rads * width) * inv_w;
}

OSC_FORMULA_FUNC float spectral_osc_pwm(float rads, float width) {
    return (width > 0.0f) ? (((rads + SPECTRAL_PI) * SPECTRAL_INV_TWO_PI < width) ? 1.0f : -1.0f) : 1.0f;
}

/* Fade envelope (Hann-window ramp for segment boundaries).
 * fade_in:  0.5 * (1 - fast_sin((j * inv_fade - 0.5) * pi))
 * fade_out: 0.5 * (1 - fast_sin((from_end * inv_fade - 0.5) * pi)) */

OSC_FORMULA_FUNC float spectral_fade_envelope_in(float j, float inv_fade) {
    return 0.5f * (1.0f - spectral_fast_sin_inline((j * inv_fade - 0.5f) * SPECTRAL_PI));
}

OSC_FORMULA_FUNC float spectral_fade_envelope_out(float j, float seg_len, float inv_fade) {
    float from_end = seg_len - 1.0f - j;
    return 0.5f * (1.0f - spectral_fast_sin_inline((from_end * inv_fade - 0.5f) * SPECTRAL_PI));
}

/* Combined fade envelope for GPU kernels (single function, float indices) */
OSC_FORMULA_FUNC float spectral_fade_envelope_gpu(float j, float seg_len, float fade_len, float inv_fade) {
    if (j < fade_len) {
        return spectral_fade_envelope_in(j, inv_fade);
    }
    float from_end = seg_len - 1.0f - j;
    if (from_end < fade_len) {
        return spectral_fade_envelope_out(j, seg_len, inv_fade);
    }
    return 1.0f;
}

#endif /* SPECTRAL_OSC_FORMULAS_H */
