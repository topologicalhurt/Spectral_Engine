/* spectral_osc_formulas.h - Canonical Synthesis Math Contract
 *
 * Single source of truth for all oscillator math:
 *   - Phase normalization
 *   - Canonical sine implementation
 *   - 8 waveform generators (sine, saw, square, triangle, asin, parabola, quantized, pwm)
 *   - Fade envelope (Hann-window ramp)
 *
 * CPU and CUDA backends include this header directly.
 * Metal MSL backend must match these formulas exactly; constants are injected
 * with SPECTRAL_STR(...) in oscillator.c.
 *
 * IMPORTANT: Any formula change here MUST be mirrored in the Metal shader string
 * in oscillator.c. Run the core math/static tests to verify cross-backend consistency.
 */
#ifndef SPECTRAL_OSC_FORMULAS_H
#define SPECTRAL_OSC_FORMULAS_H

#include "spectral_consts.h"

/* Bump when any oscillator formula, fast_sin, normalize_phase, or
 * fade_envelope changes.  Metal shader (oscillator.c) duplicates these
 * as MSL strings and checks this version at compile time. */
#define SPECTRAL_OSC_FORMULAS_VERSION 3
#include <math.h>

/* Dual-compile: C inline or CUDA device inline */
#ifdef __CUDACC__
#define OSC_FORMULA_FUNC __device__ __forceinline__
#else
#define OSC_FORMULA_FUNC static inline
#endif

/* Phase normalization (canonical formula; all backends must match).
 * Maps arbitrary phase to [-pi, pi), preserving phase 0 as 0:
 *   result = p - 2*pi * floor(p/(2*pi) + 0.5)
 * This convention keeps sine/saw/square/triangle phase semantics aligned
 * across scalar, SIMD, CUDA and Metal implementations. */
OSC_FORMULA_FUNC float spectral_normalize_phase(float p) {
    float norm = p * SPECTRAL_INV_TWO_PI;
    return p - SPECTRAL_TWO_PI * floorf(norm + 0.5f);
}

/* Canonical sine.
 *
 * Default: libm/CUDA sinf for correctness. The previous Padé [5/4]
 * coefficients had large endpoint error near +/-pi and were not an acceptable
 * kernel contract. SPECTRAL_ENABLE_APPROX_TRIG enables a bounded odd Taylor
 * polynomial on the reduced interval [-pi, pi]; keep it disabled unless the
 * target backend has passed numerical and perceptual regression tests.
 */
OSC_FORMULA_FUNC float spectral_fast_sin_inline(float x) {
#if SPECTRAL_ENABLE_APPROX_TRIG
    x = x - SPECTRAL_TWO_PI * floorf(x * SPECTRAL_INV_TWO_PI + 0.5f);
    float x2 = x * x;
    return x * (1.0f + x2 *
        (-0.16666666666666666f + x2 *
        ( 0.008333333333333333f + x2 *
        (-0.0001984126984126984f + x2 *
        ( 0.0000027557319223985893f + x2 *
        (-0.00000002505210838544172f + x2 *
        ( 0.00000000016059043836821613f + x2 *
        (-0.0000000000007647163731819816f))))))));
#else
    return sinf(x);
#endif
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
 * fade_in:  0.5 * (1 + fast_sin((j * inv_fade - 0.5) * pi))
 * fade_out: 0.5 * (1 + fast_sin((from_end * inv_fade - 0.5) * pi)) */

OSC_FORMULA_FUNC float spectral_fade_envelope_in(float j, float inv_fade) {
    return 0.5f * (1.0f + spectral_fast_sin_inline((j * inv_fade - 0.5f) * SPECTRAL_PI));
}

OSC_FORMULA_FUNC float spectral_fade_envelope_out(float j, float seg_len, float inv_fade) {
    float from_end = seg_len - 1.0f - j;
    return 0.5f * (1.0f + spectral_fast_sin_inline((from_end * inv_fade - 0.5f) * SPECTRAL_PI));
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
