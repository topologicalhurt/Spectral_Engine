/* spectral_config.h - Build configuration and platform abstraction */
#ifndef SPECTRAL_CONFIG_H
#define SPECTRAL_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include "spectral_error.h"

/* Build mode defaults */
#ifndef SPECTRAL_EMBEDDED
#define SPECTRAL_EMBEDDED       0
#endif
#ifndef SPECTRAL_RESTRICTED_MODE
#define SPECTRAL_RESTRICTED_MODE 0
#endif
#ifndef SPECTRAL_COMPACT_SEG
#define SPECTRAL_COMPACT_SEG    0
#endif
#ifndef SPECTRAL_NO_PERF
#define SPECTRAL_NO_PERF        0
#endif
#ifndef SPECTRAL_MAX_SEGS
#define SPECTRAL_MAX_SEGS       0
#endif

/* Q15 compact mode: 14-byte segments, no chirp */
#ifndef SPECTRAL_Q15_COMPACT
#define SPECTRAL_Q15_COMPACT    0
#endif

#if SPECTRAL_Q15_COMPACT
#define SPECTRAL_HAS_CHIRP      0
#define SPECTRAL_Q15_SEG_SIZE   14
#else
#define SPECTRAL_HAS_CHIRP      1
#define SPECTRAL_Q15_SEG_SIZE   16
#endif

/* Q15/Q31 conversion constants */
#define SPECTRAL_Q15_SCALE      32768.0f
#define SPECTRAL_INV_Q15_SCALE  3.0517578125e-5f
#define SPECTRAL_Q31_SCALE      2147483648.0f
#define SPECTRAL_INV_Q31_SCALE  4.6566128730773926e-10f

/* Sample type abstraction */
#if SPECTRAL_EMBEDDED
typedef int16_t  spectral_sample_t;
typedef int32_t  spectral_acc_t;

#define SPECTRAL_SAMPLE_MAX         32767
#define SPECTRAL_SAMPLE_MIN         (-32768)
#define SPECTRAL_SAMPLE_ZERO        0
#define SPECTRAL_SAMPLE_HALF        16384

#define SPECTRAL_SAMPLE_TO_FLOAT(s) ((float)(s) * SPECTRAL_INV_Q15_SCALE)
#define FLOAT_TO_SPECTRAL_SAMPLE(f) \
    ((spectral_sample_t)((f) >= 1.0f ? 32767 : (f) <= -1.0f ? -32768 : (int16_t)((f) * SPECTRAL_Q15_SCALE)))

#define SPECTRAL_SAMPLE_ADD(a, b) \
    ((spectral_sample_t)(((int32_t)(a) + (int32_t)(b)) > 32767 ? 32767 : \
                         (((int32_t)(a) + (int32_t)(b)) < -32768 ? -32768 : \
                          ((int32_t)(a) + (int32_t)(b)))))

#define SPECTRAL_SAMPLE_MUL(a, b) \
    ((spectral_sample_t)((((int32_t)(a) * (int32_t)(b)) >> 15) > 32767 ? 32767 : \
                         ((((int32_t)(a) * (int32_t)(b)) >> 15) < -32768 ? -32768 : \
                          (((int32_t)(a) * (int32_t)(b)) >> 15))))

#else

typedef float    spectral_sample_t;
typedef double   spectral_acc_t;

#define SPECTRAL_SAMPLE_MAX         1.0f
#define SPECTRAL_SAMPLE_MIN         (-1.0f)
#define SPECTRAL_SAMPLE_ZERO        0.0f
#define SPECTRAL_SAMPLE_HALF        0.5f

#define SPECTRAL_SAMPLE_TO_FLOAT(s) (s)
#define FLOAT_TO_SPECTRAL_SAMPLE(f) (f)
#define SPECTRAL_SAMPLE_ADD(a, b)   ((a) + (b))
#define SPECTRAL_SAMPLE_MUL(a, b)   ((a) * (b))

#endif

/* Timbre types */
typedef enum {
    TIMBRE_SINE     = 0,
    TIMBRE_SAW      = 1,
    TIMBRE_SQUARE   = 2,
    TIMBRE_TRIANGLE = 3,
    TIMBRE_ASIN     = 4,
    TIMBRE_PARABOLA = 5,
    TIMBRE_QUANTIZED= 6,
    TIMBRE_PWM      = 7,
    TIMBRE_COUNT    = 8
} SpectralTimbre;

#define TIMBRE_MIN  TIMBRE_SINE
#define TIMBRE_MAX  TIMBRE_PWM

#ifndef SPECTRAL_BACKEND_TIMBRE_MAX
#define SPECTRAL_BACKEND_TIMBRE_MAX     TIMBRE_MAX
#endif

/* Legacy error code aliases */
#define SPECTRAL_ERR_FILE      SPECTRAL_ERR_FILE_OPEN
#define SPECTRAL_ERR_FORMAT    SPECTRAL_ERR_FILE_FORMAT
#define SPECTRAL_ERR_IO        SPECTRAL_ERR_FILE_READ

/* Wavetable configuration */
#ifndef SPECTRAL_USE_WAVETABLE_LUT
#define SPECTRAL_USE_WAVETABLE_LUT      0
#endif
#ifndef SPECTRAL_WAVETABLE_SIZE
#define SPECTRAL_WAVETABLE_SIZE         (1<<11)
#endif
#define SPECTRAL_WAVETABLE_BITS         11
#define SPECTRAL_WAVETABLE_MASK         (SPECTRAL_WAVETABLE_SIZE - 1)
#ifndef SPECTRAL_MAX_WAVETABLES
#define SPECTRAL_MAX_WAVETABLES         8
#endif

/* LED timing (ms) */
#ifndef SPECTRAL_ERROR_BLINK_ON_MS
#define SPECTRAL_ERROR_BLINK_ON_MS      100
#endif
#ifndef SPECTRAL_ERROR_BLINK_OFF_MS
#define SPECTRAL_ERROR_BLINK_OFF_MS     100
#endif
#ifndef SPECTRAL_ERROR_BLINK_PAUSE_MS
#define SPECTRAL_ERROR_BLINK_PAUSE_MS   500
#endif
#ifndef SPECTRAL_LED_BLINK_PLAYING_MS
#define SPECTRAL_LED_BLINK_PLAYING_MS   250
#endif
#ifndef SPECTRAL_LED_BLINK_DONE_MS
#define SPECTRAL_LED_BLINK_DONE_MS      100
#endif

/* Platform detection */
#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
#define SPECTRAL_ARM_M7         1
#else
#define SPECTRAL_ARM_M7         0
#endif

#ifdef __APPLE__
#define SPECTRAL_USE_VDSP       1
#else
#define SPECTRAL_USE_VDSP       0
#endif

/* Compiler hints */
#if defined(__GNUC__) && !defined(__clang__)
#define SPECTRAL_UNROLL_4       _Pragma("GCC unroll 4")
#define SPECTRAL_UNROLL_2       _Pragma("GCC unroll 2")
#elif defined(__clang__)
#define SPECTRAL_UNROLL_4       _Pragma("clang loop unroll_count(4)")
#define SPECTRAL_UNROLL_2       _Pragma("clang loop unroll_count(2)")
#else
#define SPECTRAL_UNROLL_4
#define SPECTRAL_UNROLL_2
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SPECTRAL_LIKELY(x)      __builtin_expect(!!(x), 1)
#define SPECTRAL_UNLIKELY(x)    __builtin_expect(!!(x), 0)
#else
#define SPECTRAL_LIKELY(x)      (x)
#define SPECTRAL_UNLIKELY(x)    (x)
#endif

/* Math constants */
#define SPECTRAL_PI             3.14159265358979323846f
#define SPECTRAL_TWO_PI         6.283185307179586f
#define SPECTRAL_INV_PI         0.31830988618379067f
#define SPECTRAL_INV_TWO_PI     0.159154943091895f
#define SPECTRAL_TWO_INV_PI     0.6366197723675814f
#define SPECTRAL_PI_SQ          9.8696044f
#define SPECTRAL_INV_PI_SQ      0.10132118364233778f

/* Analysis defaults */
#if SPECTRAL_EMBEDDED
#define DEFAULT_N_FFT           1024
#define DEFAULT_HOP             256
#define DEFAULT_DB_THRESH       (-70.0f)
#define CACHE_ALIGN             32
#else
#define DEFAULT_N_FFT           4096
#define DEFAULT_HOP             128
#define DEFAULT_DB_THRESH       (-85.0f)
#define CACHE_ALIGN             64
#endif

/* Utility macros */
#ifndef MAX
#define MAX(a, b)               (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b)               (((a) < (b)) ? (a) : (b))
#endif
#ifndef CLAMP
#define CLAMP(x, lo, hi)        (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))
#endif

/* Runtime config validation */
typedef struct SpectralConfigParams {
    int sample_rate;
    float stretch;
    float pitch;
    int timbre;
    size_t buffer_size;
    size_t num_segments;
    int n_threads;
} SpectralConfigParams;

SpectralError spectral_config_validate(const SpectralConfigParams* cfg,
                                       char* error_msg, size_t error_msg_size);
int spectral_config_is_valid(const SpectralConfigParams* cfg);

#endif /* SPECTRAL_CONFIG_H */
