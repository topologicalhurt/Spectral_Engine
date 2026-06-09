/* spectral_osc_recursive.h - Coupled-form (true-rotation) Q31 sinusoidal oscillator.
 *
 * Generates sin(omega*n + phi) by rotating a unit (cos, sin) vector by angle omega each
 * sample -- no phase->table gather, so it has DETERMINISTIC latency (no cache miss) and
 * needs no LUT. Chosen over the magic-circle form, which is non-viable in fixed point
 * (ellipse amplitude error collapses SNR near Nyquist; Q15 state is unusable). The coupled
 * form with Q31 state matches the 12-bit-LUT SNR (~76-87 dB) and, with per-block renorm,
 * exceeds it -- see the characterization in tests (test_osc_recursive).
 *
 * Split: the per-sample STEP is pure Q31 fixed point (the hot loop). INIT is float
 * (activation-time only, rare) and uses the self-contained minimax sine -- no libm, M7 FPU.
 *
 * Per voice: state (c, s) Q31 + constants (cos_w, sin_w) Q31. Per sample: 4 q31xq31->q63
 * multiplies + 2 add/sub. The recurrence is SERIAL (each sample depends on the previous),
 * so unlike the LUT's pipelineable loads it has no cross-sample ILP -- its win is removing
 * the memory gather, not adding parallelism.
 */
#ifndef SPECTRAL_OSC_RECURSIVE_H
#define SPECTRAL_OSC_RECURSIVE_H

#include "spectral_q15.h"        /* q31_t, q63_t, Q31_MAX/MIN */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpectralCoupledOsc {
    q31_t c;   /* cos component of the rotating unit vector (Q31) */
    q31_t s;   /* sin component (Q31) -- this is the oscillator output */
} SpectralCoupledOsc;

/* Per-sample rotation: (c,s) <- R(omega)*(c,s). Returns the Q31 sine (the new s).
 * Pure fixed point. cos_w = cos(omega), sin_w = sin(omega) in Q31 (from _init). */
static inline q31_t spectral_coupled_step(SpectralCoupledOsc* o, q31_t cos_w, q31_t sin_w) {
    q31_t nc = (q31_t)(((q63_t)o->c * cos_w - (q63_t)o->s * sin_w) >> 31);
    q31_t ns = (q31_t)(((q63_t)o->s * cos_w + (q63_t)o->c * sin_w) >> 31);
    o->c = nc;
    o->s = ns;
    return ns;
}

/* Re-normalize the state toward unit magnitude with ONE Newton 1/sqrt step (no sqrt):
 * for m2 = c^2+s^2 near 1, g = (3 - m2)/2 ~= 1/sqrt(m2). Bounds the slow Q31 quantization
 * drift; call once per block, not per sample. g is Q30 (range ~[1,1.01], no overflow). */
static inline void spectral_coupled_renorm(SpectralCoupledOsc* o) {
    q63_t m2 = ((q63_t)o->c * o->c + (q63_t)o->s * o->s) >> 31;   /* ~2^31 at unit */
    q31_t g  = (q31_t)(0x60000000LL - (m2 >> 2));                 /* (3 - m2_norm)/2 in Q30 */
    o->c = (q31_t)(((q63_t)o->c * g) >> 30);
    o->s = (q31_t)(((q63_t)o->s * g) >> 30);
}

/* Self-contained DOUBLE-precision sine for activation-time init ONLY (rare; no libm, never
 * the hot path). Q31 rotation constants need ~1e-9 accuracy: the float minimax (~1e-7) leaves
 * the rotation angle slightly off, which drifts the phase over a long partial and collapses
 * SNR (measured: erratic -3..60 dB). This degree-15 odd Taylor folded to [-pi/2, pi/2] is
 * < 1e-11 over that range -- below Q31 resolution. Args here are bounded (|x| <= 3pi/2), so
 * the reduction uses cast-rounding (no libm floor). pi is the mathematical constant in f64. */
static inline double spectral_osc_sin_init_f64(double x) {
    const double PI = 3.14159265358979323846, INV_PI = 0.31830988618379067154;
    double t = x * INV_PI;
    long k = (long)(t >= 0.0 ? t + 0.5 : t - 0.5);   /* nearest multiple of pi */
    double xr = x - (double)k * PI, x2 = xr * xr;    /* xr in [-pi/2, pi/2] */
    double p = xr * (1.0 + x2 * (-1.0/6.0 + x2 * (1.0/120.0 + x2 * (-1.0/5040.0
             + x2 * (1.0/362880.0 + x2 * (-1.0/39916800.0 + x2 * (1.0/6227020800.0
             + x2 * (-1.0/1307674368000.0))))))));
    return (k & 1L) ? -p : p;                        /* sin(x) = (-1)^k sin(xr) */
}

/* Round a double in [-1,1] to Q31 (full double precision -- do NOT route through a float
 * converter, which would re-introduce the float-precision frequency error). */
static inline q31_t spectral_double_to_q31_round(double f) {
    if (f >= 1.0) return Q31_MAX;
    if (f <= -1.0) return Q31_MIN;
    double v = f * 2147483647.0;
    return (q31_t)(v + (v >= 0.0 ? 0.5 : -0.5));
}

/* Activation-time init (rare). omega = rad/sample, phi = initial phase rad. Fills the
 * per-voice rotation constants (cos_w, sin_w) and the initial state (c, s) to Q31 precision
 * so the recurrence frequency is exact to the fixed-point floor. cos(x) = sin(x + pi/2). */
static inline void spectral_coupled_init(SpectralCoupledOsc* o, double omega, double phi,
                                         q31_t* cos_w, q31_t* sin_w) {
    const double HALF_PI = 1.57079632679489661923;
    *cos_w = spectral_double_to_q31_round(spectral_osc_sin_init_f64(omega + HALF_PI));
    *sin_w = spectral_double_to_q31_round(spectral_osc_sin_init_f64(omega));
    o->c   = spectral_double_to_q31_round(spectral_osc_sin_init_f64(phi + HALF_PI));
    o->s   = spectral_double_to_q31_round(spectral_osc_sin_init_f64(phi));
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_OSC_RECURSIVE_H */
