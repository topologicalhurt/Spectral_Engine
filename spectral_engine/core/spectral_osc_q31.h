/* spectral_osc_q31.h - Coupled-form (true-rotation) Q31 sinusoidal oscillator.
 *
 * Generates sin(omega*n + phi) by rotating a unit (cos, sin) vector by angle omega each
 * sample -- no phase->table gather, so it has DETERMINISTIC latency (no cache miss) and
 * needs no LUT. Chosen over the magic-circle form, which is non-viable in fixed point
 * (ellipse amplitude error collapses SNR near Nyquist; Q15 state is unusable). The coupled
 * form with Q31 state matches the 12-bit-LUT SNR (~76-87 dB). The characterization
 * (test_osc_recursive) drives one long recurrence with a cheap per-block ALU renorm
 * (spectral_coupled_renorm) and exceeds the LUT SNR. NOTE: the shipping arm32 synth does
 * NOT renorm -- it bounds drift by re-seeding from the exact uint32 phase every block (so
 * its SNR is anchored on that regime, not the renorm one; closing that test gap and adopting
 * the cheaper carry-state+renorm model are tracked in EMBEDDED_RESOURCE_SEPARATION_PLAN.md).
 *
 * Split: the per-sample STEP is pure Q31 fixed point (the hot loop). INIT is DOUBLE
 * precision (spectral_osc_sin_init_f64, a degree-15 Taylor -- no libm, M7 FPU). On the arm32
 * path INIT runs PER BLOCK PER VOICE (not "rare" -- a per-block FPU burst); see the plan.
 *
 * Per voice: state (c, s) Q31 + constants (cos_w, sin_w) Q31. Per sample: 4 q31xq31->q63
 * multiplies + 2 add/sub. The recurrence is SERIAL (each sample depends on the previous),
 * so unlike the LUT's pipelineable loads it has no cross-sample ILP -- its win is removing
 * the memory gather, not adding parallelism.
 */
#ifndef SPECTRAL_OSC_Q31_H
#define SPECTRAL_OSC_Q31_H

#include "spectral_q.h"        /* q31_t, q63_t, Q31_MAX/MIN */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpectralCoupledOsc {
    q31_t c;   /* cos component of the rotating unit vector (Q31) */
    q31_t s;   /* sin component (Q31) -- this is the oscillator output */
} SpectralCoupledOsc;

/* Per-sample rotation: (c,s) <- R(omega)*(c,s). Returns the Q31 sine (the new s).
 * Pure fixed point. cos_w = cos(omega), sin_w = sin(omega) in Q31 (from _init).
 *
 * HIGH-WORD form: each product is taken at >>32 (the smull high word, free) and the pair is
 * restored to the Q31 scale with one <<1, rather than keeping the full q63 product and doing a
 * 64-bit >>31 extraction. GCC emits ~14 fewer instructions/iter -> -22% cyc per voice-sample on
 * the M7 (20.0 -> 16.0, measured llvm-mca/CortexM7Model). NOT bit-identical to the >>31 form
 * (each product truncates a bit earlier; the <<1 forces output LSB 0, an effectively-Q30 result)
 * but SNR-equivalent: measured >=74 dB over 50 Hz..23 kHz with the per-block renorm (floor 70), at
 * or above the >>31 form at every tested frequency. Portable: on a 64-bit host the >>32 is a
 * native shift, so the desktop/sim render tracks the device. */
static inline q31_t spectral_coupled_step(SpectralCoupledOsc* o, q31_t cos_w, q31_t sin_w) {
    q31_t hcc = (q31_t)((q63_t)o->c * cos_w >> 32);
    q31_t hss = (q31_t)((q63_t)o->s * sin_w >> 32);
    q31_t hsc = (q31_t)((q63_t)o->s * cos_w >> 32);
    q31_t hcs = (q31_t)((q63_t)o->c * sin_w >> 32);
    /* Restore the Q31 scale with a SATURATING double (qadd(x,x) == saturating <<1): the
     * per-product >>32 can round the pair +1 LSB over the >>31 form, so a full-scale peak
     * (low omega, cos_w ~ 2^31) would otherwise reach +2^31 and wrap. QADD/QSUB are single-cycle
     * branchless on the M7 -- same instruction count as a bare sub+lsl, but overflow-safe. */
    q31_t dc = spectral_qsub32(hcc, hss);
    q31_t ds = spectral_qadd32(hsc, hcs);
    o->c = spectral_qadd32(dc, dc);
    o->s = spectral_qadd32(ds, ds);
    return o->s;
}

/* Re-normalize the state toward unit magnitude with ONE Newton 1/sqrt step (no sqrt):
 * for m2 = c^2+s^2 near 1, g = (3 - m2)/2 ~= 1/sqrt(m2). Bounds the slow Q31 quantization
 * drift; call once per block, not per sample. g is Q30 (range ~[1,1.01], no overflow).
 * Pure Q31/ALU drift control (no FPU). Test-only today: the arm32 synth re-seeds from the
 * exact phase each block instead of carrying state + renorm; adopting this as the production
 * drift control (and dropping the per-block FPU re-seed) is tracked in the plan. */
static inline void spectral_coupled_renorm(SpectralCoupledOsc* o) {
    q63_t m2 = ((q63_t)o->c * o->c + (q63_t)o->s * o->s) >> 31;   /* ~2^31 at unit */
    q31_t g  = (q31_t)(0x60000000LL - (m2 >> 2));                 /* (3 - m2_norm)/2 in Q30 */
    o->c = (q31_t)(((q63_t)o->c * g) >> 30);
    o->s = (q31_t)(((q63_t)o->s * g) >> 30);
}

/* Self-contained DOUBLE-precision sine for oscillator init (no libm; not the per-SAMPLE hot
 * loop, but on the arm32 path it runs per block per voice -- a per-block FPU cost, not "rare").
 * Q31 rotation constants need ~1e-9 accuracy: the float minimax (~1e-7) leaves
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

/* Rotation constants from omega (rad/sample): cos_w = cos(omega), sin_w = sin(omega) in Q31.
 * They depend ONLY on the frequency, which is constant over a non-chirp partial's life, so
 * compute them ONCE at voice activation and reuse every block -- never recompute per block
 * (that is two wasted f64 minimax evals per block per voice). cos(x) = sin(x + pi/2). */
static inline void spectral_coupled_freq_constants(double omega, q31_t* cos_w, q31_t* sin_w) {
    const double HALF_PI = 1.57079632679489661923;
    *cos_w = spectral_double_to_q31_round(spectral_osc_sin_init_f64(omega + HALF_PI));
    *sin_w = spectral_double_to_q31_round(spectral_osc_sin_init_f64(omega));
}

/* Seed the rotating state (c, s) from the initial phase phi (rad), to Q31 precision. This
 * is the per-block resync: re-seeding (c, s) from the exact phase each block bounds the slow
 * Q31 drift without a renorm. The rotation constants above do NOT change block-to-block. */
static inline void spectral_coupled_seed_state(SpectralCoupledOsc* o, double phi) {
    const double HALF_PI = 1.57079632679489661923;
    o->c = spectral_double_to_q31_round(spectral_osc_sin_init_f64(phi + HALF_PI));
    o->s = spectral_double_to_q31_round(spectral_osc_sin_init_f64(phi));
}

/* Full per-voice seed (constants + state). omega = rad/sample, phi = initial phase rad.
 * Byte-identical to calling spectral_coupled_freq_constants(omega,...) then
 * spectral_coupled_seed_state(o, phi); kept for callers that want both at once (tests). */
static inline void spectral_coupled_init(SpectralCoupledOsc* o, double omega, double phi,
                                         q31_t* cos_w, q31_t* sin_w) {
    spectral_coupled_freq_constants(omega, cos_w, sin_w);
    spectral_coupled_seed_state(o, phi);
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_OSC_Q31_H */
