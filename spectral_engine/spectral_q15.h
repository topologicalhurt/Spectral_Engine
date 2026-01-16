/* spectral_q15.h - Q15 fixed-point types and arithmetic */
#ifndef SPECTRAL_Q15_H
#define SPECTRAL_Q15_H

#include <stdint.h>
#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int16_t  q15_t;
typedef int32_t  q31_t;
typedef uint16_t uq16_t;

#define Q15_MAX     ((q15_t)32767)
#define Q15_MIN     ((q15_t)-32768)
#define Q15_ONE     Q15_MAX
#define Q15_HALF    ((q15_t)16384)
#define Q15_ZERO    ((q15_t)0)
#define Q31_MAX     ((q31_t)2147483647L)
#define Q31_MIN     ((q31_t)-2147483648L)

#define FLOAT_TO_Q15(f) ((q15_t)((f) >= 1.0f ? Q15_MAX : (f) <= -1.0f ? Q15_MIN : (q15_t)((f) * SPECTRAL_Q15_SCALE)))
#define Q15_TO_FLOAT(q) ((float)(q) * SPECTRAL_INV_Q15_SCALE)
#define FLOAT_TO_Q31(f) ((q31_t)((f) >= 1.0f ? Q31_MAX : (f) <= -1.0f ? Q31_MIN : (q31_t)((f) * SPECTRAL_Q31_SCALE)))
#define Q31_TO_FLOAT(q) ((float)(q) * SPECTRAL_INV_Q31_SCALE)

#if defined(__GNUC__)
#define Q15_HOT __attribute__((hot))
#else
#define Q15_HOT
#endif

/* Saturating arithmetic */
Q15_HOT q15_t spectral_qadd16(q15_t a, q15_t b);
Q15_HOT q15_t spectral_qsub16(q15_t a, q15_t b);
Q15_HOT q31_t spectral_qadd32(q31_t a, q31_t b);
Q15_HOT q31_t spectral_qsub32(q31_t a, q31_t b);
Q15_HOT q15_t spectral_ssat16(q31_t val);
Q15_HOT q15_t spectral_q31_to_q15_sat(q31_t q);
Q15_HOT q31_t spectral_smlad(q31_t acc, q15_t a0, q15_t b0, q15_t a1, q15_t b1);
Q15_HOT q31_t spectral_smulbb(q15_t a, q15_t b);

/* Q15 operations */
Q15_HOT q15_t spectral_mul_q15(q15_t a, q15_t b);
Q15_HOT q31_t spectral_mac_q15(q31_t acc, q15_t a, q15_t b);
Q15_HOT q15_t spectral_scale_q15(q15_t sample, q15_t amplitude);

/* Bulk conversion (NEON-optimized on ARM) */
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
    q31_t    phase_acc;
    q31_t    freq_inc;
#if SPECTRAL_HAS_CHIRP
    q31_t    freq_delta;
#endif
    uint32_t seg_idx;
    q15_t    amp_current;
    q15_t    amp_delta;
} SpectralActiveSegQ15;

#ifdef __cplusplus
}
#endif

#endif
