/* spectral_q15.h - Q15 fixed-point types and arithmetic */
#ifndef SPECTRAL_Q15_H
#define SPECTRAL_Q15_H

#include <stdint.h>
#include <math.h>
#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int16_t  q15_t;
typedef int32_t  q31_t;
typedef uint16_t uq16_t;
typedef uint32_t uq32_t;

#define Q15_MAX     ((q15_t)32767)
#define Q15_MIN     ((q15_t)-32768)
#define Q15_HALF    ((q15_t)16384)
#define Q15_ZERO    ((q15_t)0)
#define Q31_MAX     ((q31_t)2147483647L)
#define Q31_MIN     ((q31_t)-2147483648L)
/* Canonical embedded fade step used by Q15 synthesis paths. */
#define SPECTRAL_FADE_STEP_Q15 ((q15_t)(Q15_MAX / SPECTRAL_FADE_SAMPLES_EMBEDDED))

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
    return acc + ((q31_t)a0 * b0) + ((q31_t)a1 * b1);
}
static inline q31_t spectral_smulbb(q15_t a, q15_t b) {
    return (q31_t)a * (q31_t)b;
}

#endif /* __ARM_FEATURE_DSP */

/* Higher-level Q15 operations (always portable, call inline primitives above) */
static inline q15_t spectral_mul_q15(q15_t a, q15_t b) {
    return spectral_ssat16(spectral_smulbb(a, b) >> 15);
}
static inline q31_t spectral_mac_q15(q31_t acc, q15_t a, q15_t b) {
    return spectral_qadd32(acc, spectral_smulbb(a, b));
}
static inline q15_t spectral_scale_q15(q15_t sample, q15_t amplitude) {
    return spectral_mul_q15(sample, amplitude);
}

/* Bulk Q31->Q15 conversion (portable scalar; see spectral_q15.c / pass 140) */
void spectral_q31_to_q15_bulk(const q31_t* src, q15_t* dst, uint32_t count);
void spectral_q31_to_q15_scaled(const q31_t* src, q15_t* dst, uint32_t count, q15_t scale);

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
