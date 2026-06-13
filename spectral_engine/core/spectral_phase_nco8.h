/* spectral_phase_nco8.h - vectorized 8-wide cubic integer-NCO phase.
 *
 * The 8-lane SIMD twin of spectral_phase_nco.h: eight independent cubic forward-
 * difference chains in uint32 lanes (full circle == 2^32, so the top 16 bits are the
 * signed Q15 phase index the spectral_osc_q15_* evaluators read and the low 16 bits are
 * fractional headroom), each lane offset by one sample and advancing by 8 (stride-8
 * differences) so ONE step emits 8 CONSECUTIVE sample indices in a simde__m128i. This
 * removes the serial scalar phase, measured at ~31-35% of the packed-Q15 kernel
 * (QTYPE_DOMAIN_PLAN.md) -- the gap between the shipped 1.2-1.4x and the ~2x
 * eval+accumulate floor.
 *
 * PRECISION: uint32 keeps only 16 fractional bits where the uint64 scalar NCO keeps 48,
 * so the stride-8 third difference (and its cubic accumulation over a segment) has less
 * headroom. The lanes are seeded EXACTLY from a scalar uint64 NCO, so linear and
 * quadratic phase stay tight; the cubic c3 path is the one to watch. Whether the cubic
 * term survives the narrowing is a MEASURED property -- see phase_nco_precision Part 3,
 * the gate for wiring this into production (decline-on-data if it drifts).
 *
 * Host generic-SIMD only (SIMDe backend); embedded keeps the scalar NCO.
 */
#ifndef SPECTRAL_PHASE_NCO8_H
#define SPECTRAL_PHASE_NCO8_H

#include "spectral_oscillator_dispatch.h"     /* OSC_SIMD_GENERIC + simde/x86/sse2.h */

#if defined(OSC_SIMD_GENERIC)

#include <stdint.h>
#include "spectral_phase_nco.h"      /* scalar uint64 NCO -- the exact seed source */
#include "simde/x86/sse4.1.h"        /* packus_epi32 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    simde__m128i acc_lo, acc_hi;   /* phase, lanes 0..3 / 4..7 (uint32, circle == 2^32) */
    simde__m128i d1_lo, d1_hi;     /* stride-8 first difference  */
    simde__m128i d2_lo, d2_hi;     /* stride-8 second difference */
    simde__m128i d3;               /* stride-8 third difference (constant across lanes) */
} SpectralPhaseNco8;

/* Seed 8 stride-8 cubic lanes EXACTLY from a scalar uint64 NCO positioned at the first
 * sample of the block (lane k -> sample start+k). Reads 25 consecutive scalar phases
 * (cold, once per block) and forms each lane's stride-8 forward differences in full
 * uint64 precision, narrowing to uint32 only at the end (>> 32). Wrap is handled for
 * free: the differences are taken in mod-2^64 uint64 arithmetic. Seeding consumes a copy,
 * so the caller's NCO is untouched. Boundary helper (it does the uint64->uint32 narrow),
 * so it stays OUTSIDE the Q-domain region. */
static inline SpectralPhaseNco8 spectral_phase_nco8_seed(const SpectralPhaseNco* start) {
    SpectralPhaseNco s = *start;
    uint64_t th[25];
    for (int i = 0; i < 25; i++) { th[i] = s.acc; s.acc += s.d1; s.d1 += s.d2; s.d2 += s.d3; }

    uint32_t acc[8], d1[8], d2[8];
    for (int k = 0; k < 8; k++) {
        acc[k] = (uint32_t)(th[k] >> 32);
        d1[k]  = (uint32_t)((th[k + 8]  - th[k]) >> 32);
        d2[k]  = (uint32_t)((th[k + 16] - 2u * th[k + 8] + th[k]) >> 32);
    }
    uint32_t d3 = (uint32_t)((th[24] - 3u * th[16] + 3u * th[8] - th[0]) >> 32);

    SpectralPhaseNco8 v;
    v.acc_lo = simde_mm_loadu_si128((const simde__m128i*)&acc[0]);
    v.acc_hi = simde_mm_loadu_si128((const simde__m128i*)&acc[4]);
    v.d1_lo  = simde_mm_loadu_si128((const simde__m128i*)&d1[0]);
    v.d1_hi  = simde_mm_loadu_si128((const simde__m128i*)&d1[4]);
    v.d2_lo  = simde_mm_loadu_si128((const simde__m128i*)&d2[0]);
    v.d2_hi  = simde_mm_loadu_si128((const simde__m128i*)&d2[4]);
    v.d3     = simde_mm_set1_epi32((int32_t)d3);
    return v;
}

/* Emit 8 consecutive Q15 phase indices (top 16 bits, round-to-nearest via a half-LSB
 * bias) and advance every lane by 8 samples. Pure integer SIMD -- q_domain_contract
 * enforces no float token between the markers (so the prose above stays outside). The
 * packed result is 8 uint16 indices; reinterpreting each as int16 yields the signed Q15
 * index, exactly as the scalar NCO's (uint16 -> q15_t) reinterpret does. */
// SPECTRAL_Q_DOMAIN BEGIN
static inline simde__m128i spectral_phase_nco8_step(SpectralPhaseNco8* v) {
    const simde__m128i bias = simde_mm_set1_epi32(0x00008000);  /* half of 1 index LSB */
    simde__m128i i_lo = simde_mm_srli_epi32(simde_mm_add_epi32(v->acc_lo, bias), 16);
    simde__m128i i_hi = simde_mm_srli_epi32(simde_mm_add_epi32(v->acc_hi, bias), 16);
    simde__m128i idx  = simde_mm_packus_epi32(i_lo, i_hi);      /* 8x uint16 index */
    v->acc_lo = simde_mm_add_epi32(v->acc_lo, v->d1_lo);
    v->acc_hi = simde_mm_add_epi32(v->acc_hi, v->d1_hi);
    v->d1_lo  = simde_mm_add_epi32(v->d1_lo, v->d2_lo);
    v->d1_hi  = simde_mm_add_epi32(v->d1_hi, v->d2_hi);
    v->d2_lo  = simde_mm_add_epi32(v->d2_lo, v->d3);
    v->d2_hi  = simde_mm_add_epi32(v->d2_hi, v->d3);
    return idx;
}
// SPECTRAL_Q_DOMAIN END

#ifdef __cplusplus
}
#endif

#endif /* OSC_SIMD_GENERIC */
#endif /* SPECTRAL_PHASE_NCO8_H */
