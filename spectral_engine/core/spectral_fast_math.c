/* spectral_fast_math.c - Fast Math Approximations */
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"

/* fast_atan2: exact by default.
 *
 * Approximate atan2 is useful only behind an explicit backend/profile decision.
 * Phase error propagates directly into resynthesis, so the default kernel
 * contract uses atan2f.
 */
float fast_atan2(float y, float x) {
#if SPECTRAL_ENABLE_APPROX_ATAN2
    float abs_x = fabsf(x), abs_y = fabsf(y);
    float a = (abs_x < abs_y) ? abs_x / (abs_y + SPECTRAL_ATAN2_EPS) : abs_y / (abs_x + SPECTRAL_ATAN2_EPS);
    float s = a * a;
    float r = ((SPECTRAL_ATAN2_A0 * s + SPECTRAL_ATAN2_A1) * s + SPECTRAL_ATAN2_A2) * s * a + a;
    if (abs_y > abs_x) r = SPECTRAL_HALF_PI - r;
    if (x < 0) r = SPECTRAL_PI_F - r;
    if (y < 0) r = -r;
    return r;
#else
    return atan2f(y, x);
#endif
}

/* fast_inv_sqrt: exact by default.
 *
 * The historical Quake-style approximation is intentionally gated behind
 * SPECTRAL_ENABLE_APPROX_INV_SQRT. It is not an acceptable default for analysis
 * or resynthesis amplitude math because its error is signal-dependent and can
 * be audible when propagated across many emitted segments.
 */
float fast_inv_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
#if defined(SPECTRAL_ENABLE_APPROX_INV_SQRT) && SPECTRAL_ENABLE_APPROX_INV_SQRT
    union { float f; uint32_t i; } conv;
    float x2 = x * 0.5f, y = x;
    conv.f = y;
    conv.i = 0x5f3759df - (conv.i >> 1);
    y = conv.f;
    y = y * (1.5f - (x2 * y * y));
    y = y * (1.5f - (x2 * y * y));
    return y;
#else
    return 1.0f / sqrtf(x);
#endif
}

/* fast_sqrt: exact by default; approximate only when explicitly requested. */
float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
#if defined(SPECTRAL_ENABLE_APPROX_INV_SQRT) && SPECTRAL_ENABLE_APPROX_INV_SQRT
    return x * fast_inv_sqrt(x);
#else
    return sqrtf(x);
#endif
}

/* fast_sin: Canonical sine for all oscillators; delegates to spectral_osc_formulas.h. */
float fast_sin(float x) {
    return spectral_fast_sin_inline(x);
}

/* Phase normalization — delegates to canonical formula in spectral_osc_formulas.h.
 * Maps arbitrary phase to [-pi, pi), preserving phase zero as zero */
float phase_to_rads(float p) {
    return spectral_normalize_phase(p);
}
