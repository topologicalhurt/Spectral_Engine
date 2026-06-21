/* spectral_phase_nco.h - Integer-NCO cubic phase accumulator (Q5, QTYPE_DOMAIN_PLAN.md).
 *
 * The integer twin of the float cubic phase spectral_segment_phase_at_cubic_f32():
 * it accumulates theta(t) = phase0 + alpha*t + c2*t^2 + c3*t^3 in a FIXED-POINT phase
 * via cubic forward-differencing -- 3 integer adds per sample, zero float on the hot
 * path -- so an opt-in Q15 oscillator can derive its phase index WITHOUT the per-sample
 * float->Q15 conversion, measured (bench_q15_pack8) as too costly to pay if Q15
 * lane-packing is ever to beat float-SIMD. This is the desktop lift of the embedded integer NCO
 * (SpectralActiveSegQ15 / spectral_phase_batch4 in spectral_synth_arm32.c), widened from
 * its linear+quadratic (phase_inc/freq_delta) form to the full cubic (third difference).
 *
 * PHASE DOMAIN: a uint64 where one full circle (2*pi rad) == 2^64, so the uint64 overflow
 * IS the mod-2*pi wrap (no conditional). The top 16 bits are the signed Q15 phase index
 * the spectral_osc_q15_* evaluators consume -- matching spectral_osc_q15_phase_from_rads
 * (index = rads/pi in Q15; its (uint16) reinterpret is the full-circle LUT index). The
 * low 48 bits are fractional headroom: the cubic third-difference D3 = 6*c3 is far below
 * one Q15 LSB for realistic chirp, so without those bits it would round to zero and the
 * cubic term would silently vanish.
 *
 * FORWARD DIFFERENCES of the cubic at unit step (exact, in units of turns*2^64):
 *   init: p=phase0, D1=alpha+c2+c3, D2=2*c2+6*c3, D3=6*c3
 *   step: emit p; p+=D1; D1+=D2; D2+=D3
 *
 * The float->fixed init (spectral_phase_nco_init) is the lone boundary and stays OUTSIDE
 * the Q-domain region; the per-sample step is pure integer and lives inside it.
 */
#ifndef SPECTRAL_PHASE_NCO_H
#define SPECTRAL_PHASE_NCO_H

#include <stdint.h>
#include <math.h>

#include "spectral_q.h"
#include "spectral_consts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t acc;   /* current phase, full circle == 2^64 */
    uint64_t d1;    /* first forward difference  */
    uint64_t d2;    /* second forward difference */
    uint64_t d3;    /* third forward difference (constant) */
} SpectralPhaseNco;

/* Fractional turn [0,1) -> uint64 phase (full circle == 2^64), computed in two 32-bit
 * steps so no double >= 2^32 is ever cast to an integer (the >= 2^64 cast is UB). Cold
 * path: runs once per segment in init, never on the sample loop. Boundary helper -- float
 * in, fixed out -- so it stays OUTSIDE the Q-domain region. */
static inline uint64_t spectral_phase_turns_to_u64(double turns) {
    double f = turns - floor(turns);                 /* [0,1) */
    if (!(f >= 0.0)) f = 0.0;                         /* NaN guard */
    if (f >= 1.0) f -= 1.0;
    double hi_d = f * 4294967296.0;                  /* * 2^32, [0, 2^32) */
    uint32_t hi = (hi_d >= 4294967296.0) ? 0xFFFFFFFFu : (uint32_t)hi_d;
    double lo_d = (hi_d - (double)hi) * 4294967296.0;
    if (lo_d < 0.0) lo_d = 0.0;
    uint32_t lo = (lo_d >= 4294967296.0) ? 0xFFFFFFFFu : (uint32_t)lo_d;
    return ((uint64_t)hi << 32) | lo;
}

/* Seed the accumulators from the float cubic coefficients. phase0/alpha/c2/c3 carry the
 * SAME meaning as spectral_segment_phase_at_cubic_f32 (radians and rad/sample^k). */
static inline void spectral_phase_nco_init(
    SpectralPhaseNco* s, float phase0, float alpha, float c2, float c3)
{
    const double inv_two_pi = 1.0 / (double)SPECTRAL_TWO_PI;
    double p0 = (double)phase0 * inv_two_pi;
    double a  = (double)alpha  * inv_two_pi;
    double c  = (double)c2     * inv_two_pi;
    double d  = (double)c3     * inv_two_pi;
    s->acc = spectral_phase_turns_to_u64(p0);
    s->d1  = spectral_phase_turns_to_u64(a + c + d);
    s->d2  = spectral_phase_turns_to_u64(2.0 * c + 6.0 * d);
    s->d3  = spectral_phase_turns_to_u64(6.0 * d);
}

/* Read the current Q15 phase index (top 16 bits, round-to-nearest via a half-LSB bias)
 * and advance the cubic forward-difference chain. Pure integer -- no float, no conversion
 * on the sample loop; the q_domain_contract CTest enforces that between the markers below
 * (so this prose, which names the boundary types, stays OUTSIDE the region). */
// SPECTRAL_Q_DOMAIN BEGIN
static inline q15_t spectral_phase_nco_step(SpectralPhaseNco* s) {
    uint16_t idx = (uint16_t)((s->acc + 0x0000800000000000ULL) >> 48);
    s->acc += s->d1;
    s->d1  += s->d2;
    s->d2  += s->d3;
    return (q15_t)idx;   /* uint16->q15 reinterpret: the full-circle alias is exact */
}
// SPECTRAL_Q_DOMAIN END

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PHASE_NCO_H */
