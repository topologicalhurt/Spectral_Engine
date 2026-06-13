/* spectral_fast_math.h - Fast Math Approximations
 *
 * Provides shared exact-by-default math wrappers for oscillator and analysis
 * modules. Approximation flags are intentionally centralized here so peak,
 * vector and synth code do not grow local copies of the same bit hacks.
 * GPU shaders must match fast_sin.
 */
#ifndef SPECTRAL_FAST_MATH_H
#define SPECTRAL_FAST_MATH_H

#include "spectral_config.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define FAST_MATH_HOT __attribute__((hot))
#else
#define FAST_MATH_HOT
#endif

FAST_MATH_HOT float fast_sin(float x);
FAST_MATH_HOT float fast_inv_sqrt(float x);
FAST_MATH_HOT float fast_sqrt(float x);
FAST_MATH_HOT float fast_peak_log(float x);
float fast_atan2(float y, float x);
float phase_to_rads(float p);

/* Polynomial atan2(y, x), ALWAYS approximate (no exact-mode gate). This is the
 * single scalar reference shared by fast_atan2()'s approximate branch and the
 * scalar tails of the SIMD vatan2 / magsq_phase kernels, so the tail can never
 * drift from the vectorized body.
 *
 * Method: reduce to the ratio a = min(|y|,|x|) / max(|y|,|x|) in [0,1], on which
 * atan is approximated by an odd minimax polynomial
 *   atan(a) ~= a + a^3 * (A2 + A1*a^2 + A0*a^4)   (Horner in s = a^2),
 * then reconstruct the quadrant via the |y|>|x|, x<0 and y<0 fixups. Output is
 * radians in [-pi, pi]. Coefficients (A0/A1/A2) live in spectral_consts.h;
 * atan2f is the exact oracle the parity test compares against. */
static inline float spectral_atan2_poly(float y, float x) {
    float abs_x = fabsf(x), abs_y = fabsf(y);
    float a = (abs_x < abs_y) ? abs_x / (abs_y + SPECTRAL_ATAN2_EPS)
                              : abs_y / (abs_x + SPECTRAL_ATAN2_EPS);
    float s = a * a;
    float r = ((SPECTRAL_ATAN2_A0 * s + SPECTRAL_ATAN2_A1) * s + SPECTRAL_ATAN2_A2) * s * a + a;
    if (abs_y > abs_x) r = SPECTRAL_HALF_PI - r;
    if (x < 0.0f)      r = SPECTRAL_PI_F - r;
    if (y < 0.0f)      r = -r;
    return r;
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_FAST_MATH_H */
