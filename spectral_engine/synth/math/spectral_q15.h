/* spectral_q15.h - Q15 fixed-point types and arithmetic
 *
 * CANONICAL Q-DOMAIN HEADER (all targets). This is the single source of truth
 * for the fixed-point types (q15_t/q31_t/uq16_t/uq32_t), their saturating
 * primitives, and the float<->Q boundary macros. It compiles host-side as well
 * as embedded, so desktop hot paths may select the Q domain too.
 *
 * TWO LAYERS (keep them separate — see QTYPE_DOMAIN_PLAN.md §2):
 *   1. Storage / transport packing. Boundaries that are *already* int16 (final
 *      PCM out, segment storage, sine LUT). Packing to Q15 here is free and
 *      lossless at the boundary — a throughput/bandwidth win at no precision
 *      cost.
 *   2. Compute-in-Q15 intermediates. Lossy: Q15 has a ~92 dB SNR ceiling vs the
 *      float oscillator's -155 dBFS, so computing audio *in* Q15 spends ~60 dB
 *      of headroom. Opt-in, per-path, only where 15 bits is provably enough and
 *      the path is throughput-bound. Float stays the default domain.
 *
 * THE BOUNDARY-MACRO RULE (enforced by the q_domain_contract CTest):
 *   Every float<->Q conversion goes through a NAMED boundary macro — never an
 *   ad-hoc multiply by a raw scale constant. The sanctioned macros are
 *   FLOAT_TO_Q15 / Q15_TO_FLOAT / FLOAT_TO_Q31 / Q31_TO_FLOAT / PHASE_RAD_TO_Q15
 *   / OMEGA_TO_Q88 here, plus SPECTRAL_SAMPLE_TO_FLOAT / FLOAT_TO_SPECTRAL_SAMPLE
 *   (the PCM-sample boundary) in spectral_config.h. The raw scale constants
 *   (SPECTRAL_Q15_SCALE / SPECTRAL_INV_Q15_SCALE / SPECTRAL_Q31_SCALE /
 *   SPECTRAL_INV_Q31_SCALE) may ONLY appear inside those macro definitions; the
 *   contract test fails the build if they leak into any other file.
 *
 * THE Q-DOMAIN REGION MARKER:
 *   Pure fixed-point compute blocks are bracketed by
 *     // SPECTRAL_Q_DOMAIN BEGIN   ...   // SPECTRAL_Q_DOMAIN END
 *   markers. The contract test rejects any `float`/`double` token inside such a
 *   region — that is the "no float arithmetic in a Q kernel" half of the
 *   no-mixing rule. Wrap future vectorized Q15 kernels (Q3) in these markers so
 *   they inherit enforcement; keep boundary conversions OUTSIDE the region.
 */
#ifndef SPECTRAL_Q15_H
#define SPECTRAL_Q15_H

#include <stdint.h>
#include <math.h>
#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* THE Q-DOMAIN MAP -- every fixed-point quantity in the engine: its carrier
 * type, binary-point format, value range, wrap behavior, and where it lives.
 * The binary-point location *is* the type -- the same int16 carries an
 * amplitude (Q1.15) and a phase index (full circle == 2^16), so the carrier
 * alone is not the contract; read the format column. Conversions between any
 * two rows, or to/from float, go ONLY through the named boundary macros below.
 *
 *   quantity           carrier    format   range / meaning              wrap
 *   -----------------  ---------  -------  ---------------------------  --------------
 *   amplitude/sample   q15_t      Q1.15    [-1, +1)  (+1 not exact)     saturate
 *   phase index        q15_t      signed   full circle == 2^16,         free (int16
 *                                 index    [-pi,+pi) -> [-32768,32767]  wraps)
 *   phase accumulator  uq32_t     UQ0.32   full circle == 2^32; top 16  free mod-2^32
 *                                          bits = the phase index       (= 2pi, no fmod)
 *   phase acc (narrow) uq16_t     UQ0.16   full circle == 2^16          free mod-2^16
 *   frequency (omega)  uint16_t   UQ8.8    rad/sample, [0, 255.99];     -
 *                      ("q88")             values > 255 pre-scaled /4
 *   phase increment    q31_t      Q1.31    per-sample step added to     free (wraps
 *                                          phase_acc                    with the acc)
 *   chirp slope        q31_t      Q1.31    per-sample step of freq_inc  -
 *   MAC accumulator    q31_t      Q2.30    sum of Q15*Q15 products;     saturate on
 *                      ("q30")             >>15 + master scale -> q15    final pack
 *
 * NAMING: a field/var suffix states the format -- *_q15, *_q88 (freq_q88),
 * phase_acc, freq_inc/freq_delta. q88 (frequency) and q30 (the MAC accumulator)
 * have NO dedicated typedef yet -- they ride uint16_t and q31_t, and the suffix
 * is the only contract. Promoting them to named typedefs is a Thread-A
 * follow-up (see QTYPE_REFACTOR_PLAN.md). */
typedef int16_t  q15_t;
typedef int32_t  q31_t;
typedef int64_t  q63_t;   /* wide accumulator for exact multi-voice MAC (SMLALD) */
typedef uint16_t uq16_t;
typedef uint32_t uq32_t;

#define Q15_MAX     ((q15_t)32767)
#define Q15_MIN     ((q15_t)-32768)
#define Q15_HALF    ((q15_t)16384)
#define Q15_ZERO    ((q15_t)0)
#define Q31_MAX     ((q31_t)2147483647L)
#define Q31_MIN     ((q31_t)-2147483648L)

static inline q15_t spectral_float_to_q15(float f) {
    if (!isfinite(f)) return Q15_ZERO;
    if (f >= 1.0f) return Q15_MAX;
    if (f <= -1.0f) return Q15_MIN;
    return (q15_t)(f * SPECTRAL_Q15_SCALE);
}

static inline q31_t spectral_float_to_q31(float f) {
    if (!isfinite(f)) return (q31_t)0;
    if (f >= 1.0f) return Q31_MAX;
    if (f <= -1.0f) return Q31_MIN;
    return (q31_t)(f * SPECTRAL_Q31_SCALE);
}

#define FLOAT_TO_Q15(f) spectral_float_to_q15((float)(f))
#define Q15_TO_FLOAT(q) ((float)(q) * SPECTRAL_INV_Q15_SCALE)
#define FLOAT_TO_Q31(f) spectral_float_to_q31((float)(f))
#define Q31_TO_FLOAT(q) ((float)(q) * SPECTRAL_INV_Q31_SCALE)

/* Phase conversion: radians [0, 2pi) -> signed Q15 [-32768, 32767]
 * Normalizes to [0,1), subtracts 0.5 to center at 0, scales to Q15. */
static inline q15_t spectral_phase_rad_to_q15(float rad) {
    float n = 0.0f;

    if (!isfinite(rad)) return Q15_ZERO;
    n = fmodf(rad, (float)SPECTRAL_TWO_PI) / (float)SPECTRAL_TWO_PI;
    if (!isfinite(n)) return Q15_ZERO;
    if (n < 0.0f) n += 1.0f;
    /* n += 1.0f can round a tiny negative up to exactly 1.0f; keep n in [0,1)
     * so (n-0.5)*65536 stays < 32768 and the int16 cast cannot be out of range. */
    if (n >= 1.0f) n -= 1.0f;
    return (q15_t)((n - 0.5f) * 65536.0f);
}

/* Omega (rad/sample) to Q8.8 frequency format.
 * Values > 255 are divided by 4 before encoding. */
static inline uint16_t spectral_omega_to_q88(float omega) {
    float o = omega;

    if (!isfinite(o) || o <= 0.0f) return 0u;
    if (o > 255.0f) o /= 4.0f;
    if (o > 255.0f) o = 255.0f;
    return (uint16_t)(o * 256.0f);
}

#define PHASE_RAD_TO_Q15(rad) spectral_phase_rad_to_q15((float)(rad))
#define OMEGA_TO_Q88(omega)   spectral_omega_to_q88((float)(omega))

#if defined(__GNUC__)
#define Q15_HOT __attribute__((hot))
#else
#define Q15_HOT
#endif

/* Inline Q15 primitives — eliminates function-call overhead in hot loops.
 * ARM DSP intrinsic path compiles to single-cycle instructions. */

#define SPECTRAL_Q15_INLINE_DEFINED 1

#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP
#include <arm_acle.h>

static inline q15_t spectral_qadd16(q15_t a, q15_t b) {
    return (q15_t)__qadd16((int32_t)a, (int32_t)b);
}
static inline q15_t spectral_qsub16(q15_t a, q15_t b) {
    return (q15_t)__qsub16((int32_t)a, (int32_t)b);
}
static inline q31_t spectral_qadd32(q31_t a, q31_t b) {
    return __qadd(a, b);
}
static inline q31_t spectral_qsub32(q31_t a, q31_t b) {
    return __qsub(a, b);
}
static inline q15_t spectral_ssat16(q31_t val) {
    return (q15_t)__ssat(val, 16);
}
static inline q15_t spectral_q31_to_q15_sat(q31_t q) {
    return (q15_t)__ssat(q >> 16, 16);
}
static inline q31_t spectral_smlad(q31_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1) {
    uint32_t packed_a = ((uint32_t)(uint16_t)a1 << 16) | (uint16_t)a0;
    uint32_t packed_b = ((uint32_t)(uint16_t)b1 << 16) | (uint16_t)b0;
    return __smlad(packed_a, packed_b, acc);
}
/* Dual 16-bit MAC into a q63 accumulator: acc + a0*b0 + a1*b1, exact (no overflow). */
static inline q63_t spectral_smlald(q63_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1) {
    uint32_t packed_a = ((uint32_t)(uint16_t)a1 << 16) | (uint16_t)a0;
    uint32_t packed_b = ((uint32_t)(uint16_t)b1 << 16) | (uint16_t)b0;
    return (q63_t)__smlald(packed_a, packed_b, (long long)acc);
}
static inline q31_t spectral_smulbb(q15_t a, q15_t b) {
    return __smulbb(a, b);
}

#else
/* Portable C fallback */

static inline q15_t spectral_qadd16(q15_t a, q15_t b) {
    int32_t sum = (int32_t)a + (int32_t)b;
    return (sum > Q15_MAX) ? Q15_MAX : (sum < Q15_MIN) ? Q15_MIN : (q15_t)sum;
}
static inline q15_t spectral_qsub16(q15_t a, q15_t b) {
    int32_t diff = (int32_t)a - (int32_t)b;
    return (diff > Q15_MAX) ? Q15_MAX : (diff < Q15_MIN) ? Q15_MIN : (q15_t)diff;
}
static inline q31_t spectral_qadd32(q31_t a, q31_t b) {
    int64_t sum = (int64_t)a + (int64_t)b;
    return (sum > Q31_MAX) ? Q31_MAX : (sum < Q31_MIN) ? Q31_MIN : (q31_t)sum;
}
static inline q31_t spectral_qsub32(q31_t a, q31_t b) {
    int64_t diff = (int64_t)a - (int64_t)b;
    return (diff > Q31_MAX) ? Q31_MAX : (diff < Q31_MIN) ? Q31_MIN : (q31_t)diff;
}
static inline q15_t spectral_ssat16(q31_t val) {
    return (val > Q15_MAX) ? Q15_MAX : (val < Q15_MIN) ? Q15_MIN : (q15_t)val;
}
static inline q15_t spectral_q31_to_q15_sat(q31_t q) {
    q31_t shifted = q >> 16;
    return (shifted > Q15_MAX) ? Q15_MAX : (shifted < Q15_MIN) ? Q15_MIN : (q15_t)shifted;
}
static inline q31_t spectral_smlad(q31_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1) {
    /* Wrap in uint32 to match the ARM __smlad (non-saturating) accumulator and
     * avoid signed-overflow UB: two Q15 products can sum past INT32_MAX. */
    uint32_t prod = (uint32_t)((q31_t)a0 * b0) + (uint32_t)((q31_t)a1 * b1);
    return (q31_t)((uint32_t)acc + prod);
}
/* Dual 16-bit MAC into a q63 accumulator: acc + a0*b0 + a1*b1, exact (no overflow). */
static inline q63_t spectral_smlald(q63_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1) {
    return acc + (q63_t)((q31_t)a0 * b0) + (q63_t)((q31_t)a1 * b1);
}
static inline q31_t spectral_smulbb(q15_t a, q15_t b) {
    return (q31_t)a * (q31_t)b;
}

#endif /* __ARM_FEATURE_DSP */

/* Higher-level Q15 operations (always portable, call inline primitives above).
 * Pure fixed-point: no float/double may appear between the markers below — the
 * q_domain_contract CTest enforces it. Conversions stay outside the region. */
// SPECTRAL_Q_DOMAIN BEGIN
static inline q15_t spectral_mul_q15(q15_t a, q15_t b) {
    return spectral_ssat16(spectral_smulbb(a, b) >> 15);
}
static inline q31_t spectral_mac_q15(q31_t acc, q15_t a, q15_t b) {
    return spectral_qadd32(acc, spectral_smulbb(a, b));
}
/* Wide MAC: accumulate exactly into a q63 (cannot overflow for realistic voice
 * counts), so the additive mix needs no per-MAC saturation and the dual-MAC SMLALD
 * applies. Saturate to Q15 once at the output (spectral_q63_to_q15_scaled). */
static inline q63_t spectral_mac_q15_64(q63_t acc, q15_t a, q15_t b) {
    return acc + (q63_t)spectral_smulbb(a, b);
}
static inline q15_t spectral_scale_q15(q15_t sample, q15_t amplitude) {
    return spectral_mul_q15(sample, amplitude);
}
// SPECTRAL_Q_DOMAIN END

/* Q30 accumulator (sum of Q15*Q15 MAC products) -> Q15 output with master
 * scaling, via a >>15 shift. See spectral_q15.c / pass 145. */
void spectral_q30_to_q15_scaled(const q31_t* accum, q15_t* dst, uint32_t count, q15_t scale);
/* Wide (q63) accumulator variant: exact sum of Q15*Q15 products, saturated to Q15
 * once at the end (>>15) before applying the master scale. */
void spectral_q63_to_q15_scaled(const q63_t* accum, q15_t* dst, uint32_t count, q15_t scale);

#define SPECTRAL_Q15_TYPES

/* Embedded Q15 segment - 14 bytes (compact) or 16 bytes (full) */
#if SPECTRAL_Q15_COMPACT

typedef struct __attribute__((packed, aligned(2))) {
    uint32_t start;
    uint16_t length;
    uint16_t freq_q88;
    int16_t  phase_q15;
    int16_t  amp_q15;
    int16_t  da_q15;
} SpectralSegmentQ15;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SpectralSegmentQ15) == 14, "size mismatch");
#endif

#else

typedef struct __attribute__((packed, aligned(2))) {
    uint32_t start;
    uint16_t length;
    uint16_t freq_q88;
    int16_t  phase_q15;
    int16_t  amp_q15;
    int16_t  da_q15;
    int16_t  df_q15;
} SpectralSegmentQ15;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SpectralSegmentQ15) == 16, "size mismatch");
#endif

#endif

/* Active segment state at runtime */
typedef struct __attribute__((aligned(4))) {
    uq32_t   phase_acc;
    q31_t    freq_inc;
#if SPECTRAL_HAS_CHIRP
    q31_t    freq_delta;
#endif
    uint32_t seg_start;
    uint32_t seg_end;
    uint32_t seg_idx;
    uint16_t seg_length;
    uint16_t fade_len;
    q15_t    amp_current;
    q15_t    amp_delta;
} SpectralActiveSegQ15;

#ifdef __cplusplus
}
#endif

#endif
