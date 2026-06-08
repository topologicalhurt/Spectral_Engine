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
    return spectral_atan2_poly(y, x);
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
    if (!(x > 0.0f) || !isfinite(x)) return 0.0f;
#if defined(SPECTRAL_ENABLE_APPROX_INV_SQRT) && SPECTRAL_ENABLE_APPROX_INV_SQRT
    union { float f; uint32_t i; } conv;
    float x2 = x * 0.5f, y = x;
    conv.f = y;
    conv.i = 0x5f3759df - (conv.i >> 1);
    y = conv.f;
    y = y * (1.5f - (x2 * y * y));
    y = y * (1.5f - (x2 * y * y));
    return isfinite(y) && y > 0.0f ? y : 0.0f;
#else
    return 1.0f / sqrtf(x);
#endif
}


/* fast_sqrt: exact by default; approximate only when explicitly requested. */
float fast_sqrt(float x) {
    if (!(x > 0.0f) || !isfinite(x)) return 0.0f;
#if defined(SPECTRAL_ENABLE_APPROX_INV_SQRT) && SPECTRAL_ENABLE_APPROX_INV_SQRT
    {
        float y = x * fast_inv_sqrt(x);
        return isfinite(y) && y > 0.0f ? y : 0.0f;
    }
#else
    return sqrtf(x);
#endif
}


/* fast_peak_log: exact by default.
 *
 * Peak interpolation uses logarithms in two places: the log-power parabolic
 * window callback and Quinn's tau correction. If a target profile decides to
 * trade tiny emitted-field error for speed, the approximation must be enabled
 * through SPECTRAL_ENABLE_APPROX_PEAK_LOG and verified against the reference
 * tests. Keep this helper shared; do not paste the polynomial into estimator
 * modules.
 *
 * Approximation source/derivation when enabled:
 *   1. Use IEEE-754 binary32 layout to split x into x = 2^e * m with
 *      m in [1, 2). Source for the floating-point format assumption:
 *      https://standards.ieee.org/standard/754-2019.html
 *   2. ln(x) = e*ln(2) + ln(m).
 *   3. Let z = (m - 1) / (m + 1). Then ln(m) = 2*atanh(z).
 *      This follows from the inverse-hyperbolic-tangent logarithmic form
 *      in NIST DLMF 4.37.25:
 *      https://dlmf.nist.gov/4.37.E25
 *   4. Approximate atanh(z) with the odd power series
 *      atanh(z) = z + z^3/3 + z^5/5 + ...
 *      truncated here after z^11/11. Series source: NIST DLMF 4.37.31:
 *      https://dlmf.nist.gov/4.37.E31
 *
 * Because m in [1, 2), z is in [0, 1/3), so the series converges quickly.
 * It is still an approximation and must remain behind
 * SPECTRAL_ENABLE_APPROX_PEAK_LOG.
 */
float fast_peak_log(float x) {
#if defined(SPECTRAL_ENABLE_APPROX_PEAK_LOG) && SPECTRAL_ENABLE_APPROX_PEAK_LOG
    union { float f; uint32_t u; } in;
    union { float f; uint32_t u; } mant;
    uint32_t exp_bits = 0u;
    int exponent = 0;
    float m = 0.0f;
    float z = 0.0f;
    float z2 = 0.0f;
    float series = 0.0f;

    if (!(x > 0.0f) || !isfinite(x)) return logf(x);

    in.f = x;
    exp_bits = (in.u >> 23u) & 0xffu;
    if (exp_bits == 0u || exp_bits == 0xffu) return logf(x);

    exponent = (int)exp_bits - 127;
    mant.u = (in.u & 0x007fffffu) | 0x3f800000u;
    m = mant.f;
    z = (m - 1.0f) / (m + 1.0f);
    z2 = z * z;
    series = z * (2.0f + z2 *
        (0.6666666666666666f + z2 *
        (0.4f + z2 *
        (0.2857142857142857f + z2 *
        (0.2222222222222222f + z2 *
         0.1818181818181818f)))));
    return (float)exponent * 0.6931471805599453f + series;
#else
    return logf(x);
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
