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
#define SPECTRAL_OSC_FORMULAS_VERSION 6
#include <math.h>
#include <limits.h>

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
 * Default (SPECTRAL_ENABLE_APPROX_TRIG == 1): a degree-9 odd minimax polynomial
 * on the principal quadrant [-pi/2, pi/2]. The argument is folded there by
 *   q  = round(x / pi)          (= floor(x/pi + 0.5))
 *   x' = x - q*pi               (in [-pi/2, pi/2])
 *   sin(x) = (-1)^q * sin(x')   (sin(q*pi)=0, cos(q*pi)=(-1)^q)
 * Measured vs libm: 1.4 ULP over the oscillator's [-pi,pi] operating range,
 * 3.0 ULP worst case over [-4pi,4pi]; the float32 evaluation noise dominates,
 * the polynomial's own residual is ~3e-9 (two orders below the float floor), so
 * the coefficients are minimax-equivalent in single precision. This is faster
 * than libm and SIMD-vectorizable (its twin lives in simde_fast_sin_ps), which
 * is why it is the default.
 *
 * SPECTRAL_ENABLE_APPROX_TRIG == 0 restores the exact libm/CUDA sinf reference
 * (the EXACT_TRIG guarantee); used by reproducible/parity builds.
 *
 * The earlier Padé [5/4] kernel was dropped for excessive endpoint error and
 * the degree-15 odd-Taylor (1.75e-6) was superseded by this minimax fold.
 */
OSC_FORMULA_FUNC float spectral_fast_sin_inline(float x) {
#if SPECTRAL_ENABLE_APPROX_TRIG
    float q  = floorf(x * SPECTRAL_INV_PI + 0.5f);   /* nearest multiple of pi  */
    float xr = x - q * SPECTRAL_PI;                  /* folded to [-pi/2, pi/2] */
    float x2 = xr * xr;
    float p = xr * (1.0f + x2 *
        (SPECTRAL_MINIMAX_SIN_C3 + x2 *
        (SPECTRAL_MINIMAX_SIN_C5 + x2 *
        (SPECTRAL_MINIMAX_SIN_C7 + x2 *
        (SPECTRAL_MINIMAX_SIN_C9)))));
    float frac = q - 2.0f * floorf(q * 0.5f);        /* q mod 2 in {0,1}        */
    float sign = 1.0f - 2.0f * frac;                 /* (-1)^q                  */
    return p * sign;
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
    /* spectral_normalize_phase() can return a value a fraction of an ULP outside
     * [-pi, pi) (catastrophic cancellation when reducing large accumulated phase),
     * so rads*INV_PI can land just outside [-1, 1]. asinf() then returns NaN, which
     * poisons the synthesized output via dst[j] += amp*wave. Clamp to asin's domain;
     * the endpoints map to the correct +/- pi/2 boundary value. */
    float a = rads * SPECTRAL_INV_PI;
    if (a > 1.0f) a = 1.0f;
    else if (a < -1.0f) a = -1.0f;
    return asinf(a);
}

OSC_FORMULA_FUNC float spectral_osc_parabola(float rads, float width) {
    (void)width;
    return 1.0f - rads * rads * SPECTRAL_INV_PI_SQ;
}

OSC_FORMULA_FUNC float spectral_osc_quantized(float rads, float width) {
    float scaled = 0.0f;
    float inv_w = 0.0f;

    if (!isfinite(rads) || !isfinite(width) || width <= 0.0f) return 0.0f;

    scaled = rads * width;
    inv_w = 1.0f / width;
    if (!isfinite(scaled) || !isfinite(inv_w) ||
        scaled < (float)INT_MIN || scaled > (float)INT_MAX) {
        return 0.0f;
    }

    return (float)(int)scaled * inv_w;
}


OSC_FORMULA_FUNC float spectral_osc_pwm(float rads, float width) {
    if (!isfinite(rads) || !isfinite(width)) return 0.0f;
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
