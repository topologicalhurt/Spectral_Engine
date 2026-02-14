/* spectral_config.h - Build Configuration and Platform Abstraction
 *
 * This header defines:
 *   - Build mode flags (SPECTRAL_EMBEDDED, SPECTRAL_RESTRICTED_MODE, etc.)
 *   - Sample type abstraction (Q15 vs float)
 *   - Timbre enumeration
 *   - Platform-specific memory/cache configuration
 *   - Feature detection (vDSP)
 *
 * For math constants, see spectral_consts.h
 * For compiler hints/macros, see spectral_macros.h
 * For SIMD detection, see oscillator_dispatch.h
 */
#ifndef SPECTRAL_CONFIG_H
#define SPECTRAL_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include "spectral_error.h"
#include "spectral_consts.h"
#include "spectral_macros.h"

/* Build Mode Flags */

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

/* Embedded float mode: use FPU instead of Q15 integer for synthesis.
 * Storage remains Q15 for memory efficiency, but synthesis uses float.
 * Requires FPU (Cortex-M4F/M7 with VFPv4). */
#ifndef SPECTRAL_EMBEDDED_FLOAT
#define SPECTRAL_EMBEDDED_FLOAT 0
#endif

/* DMA segment prefetch from SDRAM to DTCM */
#ifndef SPECTRAL_HAS_DMA
#define SPECTRAL_HAS_DMA        0
#endif
#ifndef SPECTRAL_DMA_BATCH
#define SPECTRAL_DMA_BATCH      32  /* Segments per DMA transfer (~512 bytes) */
#endif

/* SoA active segment layout (phase_acc[], freq_inc[] as separate arrays) */
#ifndef SPECTRAL_SOA_ACTIVE
#define SPECTRAL_SOA_ACTIVE     0
#endif

/* Emulator mode: desktop build simulating embedded target constraints */
#if defined(SPECTRAL_EMBEDDED_EMULATION) || defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#define SPECTRAL_IS_EMULATOR 1
#else
#define SPECTRAL_IS_EMULATOR 0
#endif

/* Q15 Segment Configuration */

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

/* Sample Type Abstraction */

#if SPECTRAL_EMBEDDED

typedef int16_t  spectral_sample_t;
typedef int32_t  spectral_acc_t;

#define SPECTRAL_SAMPLE_MAX         32767
#define SPECTRAL_SAMPLE_MIN         (-32768)
#define SPECTRAL_SAMPLE_ZERO        0
#define SPECTRAL_SAMPLE_HALF        16384

#define SPECTRAL_SAMPLE_TO_FLOAT(s) ((float)(s) * SPECTRAL_INV_Q15_SCALE)
#define FLOAT_TO_SPECTRAL_SAMPLE(f) \
    ((spectral_sample_t)(!((f) == (f)) ? 0 : (f) >= 1.0f ? 32767 : (f) <= -1.0f ? -32768 : (int16_t)((f) * SPECTRAL_Q15_SCALE)))

#define SPECTRAL_SAMPLE_ADD(a, b) \
    ((spectral_sample_t)(((int32_t)(a) + (int32_t)(b)) > 32767 ? 32767 : \
                         (((int32_t)(a) + (int32_t)(b)) < -32768 ? -32768 : \
                          ((int32_t)(a) + (int32_t)(b)))))

#define SPECTRAL_SAMPLE_MUL(a, b) \
    ((spectral_sample_t)((((int32_t)(a) * (int32_t)(b)) >> 15) > 32767 ? 32767 : \
                         ((((int32_t)(a) * (int32_t)(b)) >> 15) < -32768 ? -32768 : \
                          (((int32_t)(a) * (int32_t)(b)) >> 15))))

#else  /* Desktop: float samples */

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

/* Timbre Types */

typedef enum SpectralTimbre {
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

#define TIMBRE_MIN              TIMBRE_SINE
#define TIMBRE_MAX              TIMBRE_PWM
#define OSC_GPU_MAX_TIMBRE      TIMBRE_PARABOLA

#ifndef SPECTRAL_BACKEND_TIMBRE_MAX
#define SPECTRAL_BACKEND_TIMBRE_MAX     TIMBRE_MAX
#endif

/* Wavetable Configuration */

#ifndef SPECTRAL_WAVETABLE_SIZE
#define SPECTRAL_WAVETABLE_SIZE         (1<<11)
#endif
#define SPECTRAL_WAVETABLE_BITS         11
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(SPECTRAL_WAVETABLE_SIZE == (1 << SPECTRAL_WAVETABLE_BITS),
               "SPECTRAL_WAVETABLE_BITS must match SPECTRAL_WAVETABLE_SIZE");
#endif
#define SPECTRAL_WAVETABLE_MASK         (SPECTRAL_WAVETABLE_SIZE - 1)
#ifndef SPECTRAL_MAX_WAVETABLES
#define SPECTRAL_MAX_WAVETABLES         8
#endif

/* Synthesis Defaults */

/* GPU/compute block size for Metal/CUDA backends */
#ifndef SPECTRAL_GPU_TILE_SIZE
#define SPECTRAL_GPU_TILE_SIZE          512
#endif

/* Headroom factor for normalization (0.95 = -0.45dB) */
#ifndef SPECTRAL_NORMALIZE_HEADROOM
#define SPECTRAL_NORMALIZE_HEADROOM     0.95f
#endif

/* Emulator headroom for Q15 amplitude scaling */
#define SPECTRAL_EMULATOR_HEADROOM      0.99f

/* Metal segment cache size (threadgroup shared memory) */
#define SPECTRAL_METAL_SEG_CACHE_SIZE   128

/* Platform Detection */

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

/* ARM32 Embedded Configuration */

#if SPECTRAL_ARM_M7

/* STM32H7 Memory Regions:
 *   DTCM: 128KB @ 0x20000000 - Zero wait states
 *   ITCM: 64KB @ 0x00000000 - Instruction TCM
 *   AXI SRAM: 512KB - Cached */
#define SPECTRAL_DTCM_SIZE_KB   128
#define SPECTRAL_DTCM_SIZE      (SPECTRAL_DTCM_SIZE_KB * 1024)
#define SPECTRAL_CACHE_LINE     32

#ifndef SPECTRAL_OPT_LEVEL
#define SPECTRAL_OPT_LEVEL      1
#endif

#ifndef SPECTRAL_ARM32_MAX_ACTIVE
#define SPECTRAL_ARM32_MAX_ACTIVE    512
#endif

#endif /* SPECTRAL_ARM_M7 */

/* Linker Section Annotations
 * SPECTRAL_DTCM  — Zero wait-state data memory (128KB on STM32H7)
 * SPECTRAL_ITCM  — Zero wait-state instruction memory (64KB)
 * SPECTRAL_SDRAM — External SDRAM (large, higher latency, prefetchable)
 * On non-embedded targets these expand to nothing. */
#if SPECTRAL_ARM_M7 && SPECTRAL_EMBEDDED
#define SPECTRAL_DTCM   __attribute__((section(".dtcm_data")))
#define SPECTRAL_ITCM   __attribute__((section(".itcm_text")))
#define SPECTRAL_SDRAM  __attribute__((section(".sdram_data")))
#else
#define SPECTRAL_DTCM
#define SPECTRAL_ITCM
#define SPECTRAL_SDRAM
#endif

/* Fallback defaults for non-ARM platforms */
#ifndef SPECTRAL_CACHE_LINE
#define SPECTRAL_CACHE_LINE     64
#endif
#ifndef SPECTRAL_ARM32_MAX_ACTIVE
#define SPECTRAL_ARM32_MAX_ACTIVE    512
#endif

/* Analysis Defaults */

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

#endif /* SPECTRAL_CONFIG_H */
