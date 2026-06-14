/* spectral_q.h - fixed-point Q-domain types and arithmetic (q15/q31/q63, uq16/uq32)
 *
 * CANONICAL Q-DOMAIN HEADER (all targets). This is the single source of truth
 * for the fixed-point types (q15_t/q31_t/q63_t/uq16_t/uq32_t), their saturating
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
 *   / OMEGA_TO_Q88 here, plus spectral_sample_to_float / float_to_spectral_sample
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
#ifndef SPECTRAL_Q_H
#define SPECTRAL_Q_H

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
 *   frequency (omega)  uq88_t     UQ8.8    rad/sample, [0, 255.99];     -
 *                                          values > 255 pre-scaled /4
 *   phase increment    q31_t      Q1.31    per-sample step added to     free (wraps
 *                                          phase_acc                    with the acc)
 *   chirp slope        q31_t      Q1.31    per-sample step of phase_inc  -
 *   MAC accumulator    q31_t      Q2.30    sum of Q15*Q15 products;     saturate on
 *                      ("q30")             >>15 + master scale -> q15    final pack
 *   MAC accum (wide)   q63_t      Q2.30    live multi-voice mix sum;    saturate on
 *                                          >>15 + master scale -> q15    final pack
 *   format ladder      q63_t      Q1.63    [-1, +1); widens from / nar- saturate
 *                                          rows to q15 & q31 (Q ladder)  (narrow only)
 *
 * The SAME int64 carrier (q63_t) plays both q63 rows: a Q2.30 accumulator for the
 * mix-down (spectral_q63_to_q15_scaled, >>15) and a Q1.63 fraction for the signed
 * Q ladder (spectral_q15_to_q63 .. spectral_q63_to_q15_round) -- disambiguated by
 * use-site, exactly as q31_t carries both Q1.31 and the "q30" accumulator above.
 *
 * NAMING: a field/var suffix states the format -- *_q15, *_q88 (freq_q88),
 * phase_acc, phase_inc/freq_delta. The frequency UQ8.8 carrier is uq88_t; the
 * Q2.30 MAC accumulator ("q30") rides q31_t with the suffix as its only contract
 * (the live multi-voice mix accumulates in the wider q63_t). */
typedef int16_t  q15_t;
typedef int32_t  q31_t;
typedef int64_t  q63_t;   /* wide accumulator for exact multi-voice MAC (SMLALD) */
typedef uint16_t uq16_t;
typedef uint32_t uq32_t;
typedef uint16_t uq88_t;  /* UQ8.8 frequency (omega); same carrier as uq16_t, different binary point */

/* Q range macros. #ifndef-guarded for CMSIS-DSP coexistence: arm_math_types.h
 * (vendored CMSIS-DSP) defines the same four names with identical VALUES but
 * its definitions are unguarded -- in a TU that needs both headers (the IFFT
 * synthesis path), include CMSIS BEFORE this header and these guards keep the
 * build warning-free under -Werror. The boundary rule stands regardless:
 * CMSIS-DSP is for leaf algorithm blocks (FFT); engine Q semantics live here. */
#ifndef Q15_MAX
#define Q15_MAX     ((q15_t)32767)
#endif
#ifndef Q15_MIN
#define Q15_MIN     ((q15_t)-32768)
#endif
#define Q15_HALF    ((q15_t)16384)
#define Q15_ZERO    ((q15_t)0)
#ifndef Q31_MAX
#define Q31_MAX     ((q31_t)2147483647L)
#endif
#ifndef Q31_MIN
#define Q31_MIN     ((q31_t)-2147483648L)
#endif
/* q63 extremes for the signed ladder. Written as (max-1)-form so no token
 * overflows at parse, and #ifndef-guarded to coexist with CMSIS-DSP. */
#ifndef Q63_MAX_L
#define Q63_MAX_L   ((q63_t)9223372036854775807LL)
#endif
#ifndef Q63_MIN_L
#define Q63_MIN_L   ((q63_t)(-9223372036854775807LL - 1))
#endif

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
 * Normalizes to [0,1), subtracts 0.5 to center at 0, scales by 65536 = 2^16
 * (the phase index spans the full circle as 2^16 — see the Q-domain map). */
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

/* Omega (rad/sample) to Q8.8 frequency format (scale by 256 = 2^8, the 8
 * fractional bits). Values > 255 are pre-divided by 4 so they still fit UQ8.8. */
static inline uq88_t spectral_omega_to_q88(float omega) {
    float o = omega;

    if (!isfinite(o) || o <= 0.0f) return 0u;
    if (o > 255.0f) o /= 4.0f;
    if (o > 255.0f) o = 255.0f;
    return (uq88_t)(o * 256.0f);
}

#define PHASE_RAD_TO_Q15(rad) spectral_phase_rad_to_q15((float)(rad))
#define OMEGA_TO_Q88(omega)   spectral_omega_to_q88((float)(omega))

/* Inline Q15 primitives — eliminates function-call overhead in hot loops.
 * ARM DSP intrinsic path compiles to single-cycle instructions. */

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
/* ACLE has no __smulbb (GCC arm_acle.h: __smlabb yes, __smulbb no — the DSP
 * branch failed to compile under arm-none-eabi-gcc). A widened 16x16 multiply
 * is exact and GCC selects SMULBB for it (codegen-verified). */
static inline q31_t spectral_smulbb(q15_t a, q15_t b) {
    return (q31_t)a * (q31_t)b;
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

/* Wide (q63) accumulator variant: exact sum of Q15*Q15 products, saturated to Q15
 * once at the end (>>15) before applying the master scale. */
void spectral_q63_to_q15_scaled(const q63_t* accum, q15_t* dst, uint32_t count, q15_t scale);

/* ============================ SIGNED Q LADDER ============================
 * Cross-product format conversions between the signed Q types, treating every
 * code as the SAME fraction in [-1, 1): q15 = Q1.15 (code/2^15), q31 = Q1.31
 * (code/2^31), q63 = Q1.63 (code/2^63). The three widenings are exact arithmetic
 * left shifts (gain fractional bits, lossless); the three narrowings are
 * round-half-up (add a half-LSB, then arithmetic right shift) followed by
 * saturation to the asymmetric two's-complement range. Each is pinned bit-exact
 * by the q_ladder_parity CTest against an integer (__int128) oracle.
 *
 * This is the FORMAT ladder. It is deliberately distinct from the other two q63
 * roles on the same int64 carrier: spectral_q63_to_q15_scaled treats q63 as a
 * Q2.30 MAC accumulator (>>15 + master scale), and spectral_q31_to_q15_sat is a
 * TRUNCATING >>16 narrow. The ladder narrowings ROUND; keep the names separate.
 *
 * The +half-LSB is computed in the next-wider type so it never overflows; on the
 * q63 narrowings the only value that could overflow the add already saturates, so
 * it is short-circuited before the add. ssat clamps via __ssat on ARM. */
// SPECTRAL_Q_DOMAIN BEGIN
/* Widen in UNSIGNED then reinterpret: left-shifting a negative SIGNED value is
 * undefined behavior (C11 6.5.7p4) and the min code is negative. The unsigned
 * shift wraps deterministically; the reinterpret recovers the exact widened
 * value (and keeps the conversions UBSan-clean, like the SMLAD packers above). */
static inline q31_t spectral_q15_to_q31(q15_t x) { return (q31_t)((uint32_t)(int32_t)x << 16); }
static inline q63_t spectral_q15_to_q63(q15_t x) { return (q63_t)((uint64_t)(int64_t)x << 48); }
static inline q63_t spectral_q31_to_q63(q31_t x) { return (q63_t)((uint64_t)(int64_t)x << 32); }

static inline q15_t spectral_q31_to_q15_round(q31_t x) {
    int64_t w = ((int64_t)x + (int64_t)(1 << 15)) >> 16;   /* round-half-up */
    return spectral_ssat16((q31_t)w);                      /* |w| <= 32768 -> fits q31 */
}
static inline q31_t spectral_q63_to_q31_round(q63_t x) {
    if (x > Q63_MAX_L - (q63_t)(1LL << 31)) return Q31_MAX; /* add would overflow -> sat */
    int64_t w = (x + (q63_t)(1LL << 31)) >> 32;
    return (w > Q31_MAX) ? Q31_MAX : (w < Q31_MIN) ? Q31_MIN : (q31_t)w;
}
static inline q15_t spectral_q63_to_q15_round(q63_t x) {
    if (x > Q63_MAX_L - (q63_t)(1LL << 47)) return Q15_MAX; /* add would overflow -> sat */
    int64_t w = (x + (q63_t)(1LL << 47)) >> 48;
    return spectral_ssat16((q31_t)w);                      /* |w| <= 32768 -> fits q31 */
}
// SPECTRAL_Q_DOMAIN END

/* Embedded Q15 segment - 14 bytes (compact) or 16 bytes (full) */
#if SPECTRAL_Q15_COMPACT

typedef struct __attribute__((packed, aligned(2))) {
    uint32_t start;
    uint16_t length;
    uq88_t   freq_q88;
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
    uq88_t   freq_q88;
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
    q31_t    phase_inc;
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
