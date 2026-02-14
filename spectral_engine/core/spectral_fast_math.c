/* spectral_fast_math.c - Fast Math Approximations */
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"

/* fast_atan2: Polynomial atan2 approximation (~0.07 max error, ~10x faster than libm) */
float fast_atan2(float y, float x) {
    float abs_x = fabsf(x), abs_y = fabsf(y);
    float a = (abs_x < abs_y) ? abs_x / (abs_y + SPECTRAL_ATAN2_EPS) : abs_y / (abs_x + SPECTRAL_ATAN2_EPS);
    float s = a * a;
    float r = ((SPECTRAL_ATAN2_A0 * s + SPECTRAL_ATAN2_A1) * s + SPECTRAL_ATAN2_A2) * s * a + a;
    if (abs_y > abs_x) r = SPECTRAL_HALF_PI - r;
    if (x < 0) r = SPECTRAL_PI_F - r;
    if (y < 0) r = -r;
    return r;
}

/* fast_sin: Canonical sine for all oscillators. GPU shaders must match.
 * Pade [5/4] approximation (~1e-5 max error) — delegates to spectral_osc_formulas.h */
float fast_sin(float x) {
    return spectral_fast_sin_inline(x);
}

/* Phase normalization — delegates to canonical formula in spectral_osc_formulas.h.
 * Maps arbitrary phase to [-pi, pi): 2*pi * (p/(2*pi) - floor(p/(2*pi)) - 0.5) */
float phase_to_rads(float p) {
    return spectral_normalize_phase(p);
}
