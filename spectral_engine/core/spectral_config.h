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
/* Project-level unsafe fast-math profile (set by CMake in non-repro host builds). */
#ifndef SPECTRAL_CUSTOM_FAST_MATH_MODE
#define SPECTRAL_CUSTOM_FAST_MATH_MODE 0
#endif
/* Restricted profiling gate (single ownership).
 * Enabled only when restricted mode is active and restricted debug profiling
 * has been explicitly enabled by the build. */
#if SPECTRAL_RESTRICTED_MODE && defined(SPECTRAL_DEBUG_RESTRICTED) && SPECTRAL_DEBUG_RESTRICTED
#define SPECTRAL_RESTRICTED_PROFILE 1
#else
#define SPECTRAL_RESTRICTED_PROFILE 0
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
#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
#define SPECTRAL_SOA_ACTIVE     1
#else
#define SPECTRAL_SOA_ACTIVE     0
#endif
#endif

/* Embedded simulation mode: desktop build modeling embedded target constraints. */
#if defined(SPECTRAL_EMBEDDED_SIMULATION) || defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#define SPECTRAL_IS_EMBEDDED_SIM 1
#else
#define SPECTRAL_IS_EMBEDDED_SIM 0
#endif

/* Host file-system API availability.
 * Desktop builds have stdio/fopen; embedded and simulation builds do not.
 * Controls whether resource entries carry a path string (host) or an FNV-1a
 * file ID (embedded), and whether the hash digest is 64-bit or 32-bit. */
#ifndef SPECTRAL_HASH_HAS_HOST_FILE_API
#if !SPECTRAL_EMBEDDED && !SPECTRAL_IS_EMBEDDED_SIM
#define SPECTRAL_HASH_HAS_HOST_FILE_API 1
#else
#define SPECTRAL_HASH_HAS_HOST_FILE_API 0
#endif
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

/* Canonical resolution/fallback context labels */
#ifndef SPECTRAL_RESOLUTION_EVENT_FALLBACK
#define SPECTRAL_RESOLUTION_EVENT_FALLBACK "fallback"
#endif
#ifndef SPECTRAL_RESOLUTION_EVENT_RESOLUTION
#define SPECTRAL_RESOLUTION_EVENT_RESOLUTION "resolution"
#endif
#ifndef SPECTRAL_RESOLUTION_SCOPE_SELECTION
#define SPECTRAL_RESOLUTION_SCOPE_SELECTION "selection"
#endif
#ifndef SPECTRAL_RESOLUTION_SCOPE_BACKEND
#define SPECTRAL_RESOLUTION_SCOPE_BACKEND "backend"
#endif
#ifndef SPECTRAL_RESOLUTION_SCOPE_TIMBRE
#define SPECTRAL_RESOLUTION_SCOPE_TIMBRE "timbre"
#endif
#ifndef SPECTRAL_RESOLUTION_SCOPE_WAVETABLE
#define SPECTRAL_RESOLUTION_SCOPE_WAVETABLE "wavetable"
#endif
#ifndef SPECTRAL_RESOLUTION_LABEL_UNKNOWN
#define SPECTRAL_RESOLUTION_LABEL_UNKNOWN "unknown"
#endif
#ifndef SPECTRAL_RESOLUTION_BACKEND_DISPATCH
#define SPECTRAL_RESOLUTION_BACKEND_DISPATCH "dispatch"
#endif
#ifndef SPECTRAL_RESOLUTION_BACKEND_CPU
#define SPECTRAL_RESOLUTION_BACKEND_CPU "CPU"
#endif
#ifndef SPECTRAL_RESOLUTION_BACKEND_SIMULATION
#define SPECTRAL_RESOLUTION_BACKEND_SIMULATION "Simulation"
#endif

/* Canonical execution-mode labels for logging/diagnostics */
#ifndef SPECTRAL_EXEC_MODE_DESKTOP
#define SPECTRAL_EXEC_MODE_DESKTOP "desktop"
#endif
#ifndef SPECTRAL_EXEC_MODE_EMBEDDED_SIM
#define SPECTRAL_EXEC_MODE_EMBEDDED_SIM "embedded-simulation"
#endif
#ifndef SPECTRAL_EXEC_MODE_RESTRICTED
#define SPECTRAL_EXEC_MODE_RESTRICTED "restricted"
#endif
#ifndef SPECTRAL_EXEC_MODE_EMBEDDED
#define SPECTRAL_EXEC_MODE_EMBEDDED "embedded"
#endif

/* Canonical resolution/fallback reason strings */
#ifndef SPECTRAL_RESOLUTION_REASON_AUTO_BACKEND
#define SPECTRAL_RESOLUTION_REASON_AUTO_BACKEND \
    "automatic backend selection resolved request"
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_BACKEND_NOT_COMPILED
#define SPECTRAL_RESOLUTION_REASON_BACKEND_NOT_COMPILED \
    "requested backend is not compiled in this build"
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_BACKEND_UNAVAILABLE
#define SPECTRAL_RESOLUTION_REASON_BACKEND_UNAVAILABLE \
    "requested backend is unavailable at runtime"
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_WAVETABLE_CPU_ONLY
#define SPECTRAL_RESOLUTION_REASON_WAVETABLE_CPU_ONLY \
    "wavetable rendering requires CPU backend"
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_GPU_TIMBRE_LIMIT
#define SPECTRAL_RESOLUTION_REASON_GPU_TIMBRE_LIMIT \
    "GPU oscillators currently support sine..parabola (0..5); quantized/pwm require CPU path."
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_SIM_TIMBRE_LIMIT
#define SPECTRAL_RESOLUTION_REASON_SIM_TIMBRE_LIMIT \
    "Embedded simulation currently models sine-only oscillator behavior."
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_SIM_WAVETABLE_UNSUPPORTED
#define SPECTRAL_RESOLUTION_REASON_SIM_WAVETABLE_UNSUPPORTED \
    "embedded simulation does not support wavetable rendering; continuing with oscillator synthesis"
#endif
#ifndef SPECTRAL_RESOLUTION_REASON_INVALID_TIMBRE_ID
#define SPECTRAL_RESOLUTION_REASON_INVALID_TIMBRE_ID \
    "invalid timbre id outside supported enum range"
#endif

/* Canonical simulation environment keys */
#ifndef SPECTRAL_ENV_SIM_PERF_PROFILE
#define SPECTRAL_ENV_SIM_PERF_PROFILE "SPECTRAL_SIM_PERF_PROFILE"
#endif
#ifndef SPECTRAL_ENV_SIM_PESSIMISM
#define SPECTRAL_ENV_SIM_PESSIMISM "SPECTRAL_SIM_PESSIMISM"
#endif
#ifndef SPECTRAL_ENV_SIM_PERF_COLD
#define SPECTRAL_ENV_SIM_PERF_COLD "SPECTRAL_SIM_PERF_COLD"
#endif
/* Emit split analysis stage markers (fft/track) in addition to aggregate analysis markers. */
#ifndef SPECTRAL_ENV_STAGE_SPLIT_ANALYSIS
#define SPECTRAL_ENV_STAGE_SPLIT_ANALYSIS "SPECTRAL_STAGE_SPLIT_ANALYSIS"
#endif

/* Shared execution mode helper to avoid drift across modules. */
static inline const char* spectral_exec_mode_name(void) {
#if SPECTRAL_IS_EMBEDDED_SIM
    return SPECTRAL_EXEC_MODE_EMBEDDED_SIM;
#elif SPECTRAL_RESTRICTED_MODE
    return SPECTRAL_EXEC_MODE_RESTRICTED;
#elif SPECTRAL_EMBEDDED
    return SPECTRAL_EXEC_MODE_EMBEDDED;
#else
    return SPECTRAL_EXEC_MODE_DESKTOP;
#endif
}

/* Canonical runtime/CLI validation bounds */
#ifndef SPECTRAL_MIN_FFT_SIZE
#define SPECTRAL_MIN_FFT_SIZE           64
#endif
#ifndef SPECTRAL_MIN_SAMPLE_RATE
#define SPECTRAL_MIN_SAMPLE_RATE        8000
#endif
#ifndef SPECTRAL_MAX_SAMPLE_RATE
#define SPECTRAL_MAX_SAMPLE_RATE        192000
#endif
#ifndef SPECTRAL_MAX_THREADS
#define SPECTRAL_MAX_THREADS            256
#endif
#ifndef SPECTRAL_EMBEDDED_MAX_BUFFER_SIZE
#define SPECTRAL_EMBEDDED_MAX_BUFFER_SIZE 4096
#endif
#ifndef SPECTRAL_MAX_STRETCH
#define SPECTRAL_MAX_STRETCH            1000.0f
#endif
#ifndef SPECTRAL_MIN_PITCH
#define SPECTRAL_MIN_PITCH              (-48.0f)
#endif
#ifndef SPECTRAL_MAX_PITCH
#define SPECTRAL_MAX_PITCH              48.0f
#endif

/* Canonical output/pipeline defaults */
#ifndef SPECTRAL_PIPELINE_PATH_CAPACITY
#define SPECTRAL_PIPELINE_PATH_CAPACITY 1024
#endif
#ifndef SPECTRAL_OUTPUT_DIR_PRIMARY
#define SPECTRAL_OUTPUT_DIR_PRIMARY     "../output"
#endif
#ifndef SPECTRAL_OUTPUT_DIR_FALLBACK
#define SPECTRAL_OUTPUT_DIR_FALLBACK    "output"
#endif
#ifndef SPECTRAL_OUTPUT_WAV_NAME
#define SPECTRAL_OUTPUT_WAV_NAME        "out_c.wav"
#endif
#ifndef SPECTRAL_OUTPUT_SEGMENTS_NAME
#define SPECTRAL_OUTPUT_SEGMENTS_NAME   "segments.bin"
#endif
#ifndef SPECTRAL_OUTPUT_CACHE_SUBDIR
#define SPECTRAL_OUTPUT_CACHE_SUBDIR    "cache"
#endif

/* Canonical tool/runtime defaults */
#ifndef SPECTRAL_CONVERT_DEFAULT_POOL_MB
#define SPECTRAL_CONVERT_DEFAULT_POOL_MB 48u
#endif
#ifndef SPECTRAL_DEBUG_ONCE_MAX
#define SPECTRAL_DEBUG_ONCE_MAX         64
#endif

/* Canonical analysis/tracker defaults */

#ifndef SPECTRAL_TRACK_DEFAULT_WIDTH
#define SPECTRAL_TRACK_DEFAULT_WIDTH    0.5f
#endif
#ifndef SPECTRAL_TRACK_INTERP_LOG_DOMAIN
#if SPECTRAL_CUSTOM_FAST_MATH_MODE
#define SPECTRAL_TRACK_INTERP_LOG_DOMAIN 0
#else
#define SPECTRAL_TRACK_INTERP_LOG_DOMAIN 1
#endif
#endif
#ifndef SPECTRAL_TRACK_CANDIDATE_BATCH
#define SPECTRAL_TRACK_CANDIDATE_BATCH  128u
#endif
#ifndef SPECTRAL_TRACK_PREFETCH_LOOKAHEAD
#define SPECTRAL_TRACK_PREFETCH_LOOKAHEAD 12u
#endif
#ifndef SPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE
#define SPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE 48u
#endif
#ifndef SPECTRAL_TRACK_PREFETCH_READ_LOCALITY
#define SPECTRAL_TRACK_PREFETCH_READ_LOCALITY 2
#endif
#ifndef SPECTRAL_TRACK_PREFETCH_PHASE
#define SPECTRAL_TRACK_PREFETCH_PHASE 1
#endif
#ifndef SPECTRAL_TRACK_PREFETCH_WRITE_LOCALITY
#define SPECTRAL_TRACK_PREFETCH_WRITE_LOCALITY 2
#endif
#ifndef SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE
#define SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE 16u
#endif
#ifndef SPECTRAL_TRACK_PAIR_OMP_CHUNK
#define SPECTRAL_TRACK_PAIR_OMP_CHUNK 0u
#endif
#ifndef SPECTRAL_TRACK_SEG_PREFETCH_DISTANCE
#define SPECTRAL_TRACK_SEG_PREFETCH_DISTANCE 16u
#endif
#ifndef SPECTRAL_TRACK_DEBUG_TIMING
#define SPECTRAL_TRACK_DEBUG_TIMING 0
#endif
#ifndef SPECTRAL_STFT_CHUNK_FRAMES
#define SPECTRAL_STFT_CHUNK_FRAMES      512u
#endif
#ifndef SPECTRAL_STFT_CHUNK_THRESHOLD
#define SPECTRAL_STFT_CHUNK_THRESHOLD   (32ul * 1024ul * 1024ul)
#endif
#ifndef SPECTRAL_PRETOUCH_THRESHOLD
#define SPECTRAL_PRETOUCH_THRESHOLD     (64ul * 1024ul * 1024ul)
#endif
#ifndef SPECTRAL_PRETOUCH_PAGE_SIZE
#define SPECTRAL_PRETOUCH_PAGE_SIZE     4096u
#endif
#ifndef SPECTRAL_SEGMENT_POOL_BLOCK_SIZE
#define SPECTRAL_SEGMENT_POOL_BLOCK_SIZE 4096u
#endif

/* Canonical optional-processing policy flags */
#ifndef SPECTRAL_PROCESS_STRICT
#define SPECTRAL_PROCESS_STRICT         0
#endif

/* Canonical embedded debug LED timing defaults (milliseconds) */
#ifndef SPECTRAL_ERROR_BLINK_ON_MS
#define SPECTRAL_ERROR_BLINK_ON_MS      100u
#endif
#ifndef SPECTRAL_ERROR_BLINK_OFF_MS
#define SPECTRAL_ERROR_BLINK_OFF_MS     100u
#endif
#ifndef SPECTRAL_ERROR_BLINK_PAUSE_MS
#define SPECTRAL_ERROR_BLINK_PAUSE_MS   500u
#endif
#ifndef SPECTRAL_LED_BLINK_PLAYING_MS
#define SPECTRAL_LED_BLINK_PLAYING_MS   250u
#endif
#ifndef SPECTRAL_LED_BLINK_DONE_MS
#define SPECTRAL_LED_BLINK_DONE_MS      100u
#endif

/* Canonical embedded target defaults used by perf estimation */
#ifndef SPECTRAL_EMBEDDED_DEFAULT_CPU_MHZ
#define SPECTRAL_EMBEDDED_DEFAULT_CPU_MHZ      480u
#endif
#ifndef SPECTRAL_EMBEDDED_DEFAULT_SAMPLE_RATE
#define SPECTRAL_EMBEDDED_DEFAULT_SAMPLE_RATE  48000u
#endif
#ifndef SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE
#define SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE   256u
#endif
#ifndef SPECTRAL_EMBEDDED_SRAM_KB
#define SPECTRAL_EMBEDDED_SRAM_KB              512u
#endif
#ifndef SPECTRAL_EMBEDDED_SDRAM_KB
#define SPECTRAL_EMBEDDED_SDRAM_KB             65536u
#endif
#ifndef SPECTRAL_EMBEDDED_DEFAULT_MEMORY_KB
#define SPECTRAL_EMBEDDED_DEFAULT_MEMORY_KB \
    (SPECTRAL_EMBEDDED_SRAM_KB + SPECTRAL_EMBEDDED_SDRAM_KB)
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

/* Canonical fade lengths used by synthesis paths.
 * Desktop backends use a longer fade for smoother overlap,
 * embedded paths use a shorter fade to reduce CPU work. */
#ifndef SPECTRAL_FADE_SAMPLES_DESKTOP
#define SPECTRAL_FADE_SAMPLES_DESKTOP 64
#endif
#ifndef SPECTRAL_FADE_SAMPLES_EMBEDDED
#define SPECTRAL_FADE_SAMPLES_EMBEDDED 32
#endif
/* Optimization level: 0=safe, 1=balanced (default), 2=aggressive, 3=reserved. */
#ifndef SPECTRAL_OPT_LEVEL
#define SPECTRAL_OPT_LEVEL      1
#endif

/* GPU/compute block size for Metal/CUDA backends */
#ifndef SPECTRAL_GPU_TILE_SIZE
#define SPECTRAL_GPU_TILE_SIZE          512
#endif

/* Max deterministic work partitions for CPU synthesis reduction.
 * 0 keeps partition count aligned to the selected thread count. */
#ifndef SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS
#define SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS 0
#endif

/* Headroom factor for normalization (0.95 = -0.45dB) */
#ifndef SPECTRAL_NORMALIZE_HEADROOM
#define SPECTRAL_NORMALIZE_HEADROOM     0.95f
#endif

/* Embedded simulation headroom for Q15 amplitude scaling */
#define SPECTRAL_SIMULATION_HEADROOM    0.99f

/* GPU segment cache size (threadgroup / shared memory).
 * Both Metal and CUDA use 256 entries of SegmentGpu (32 bytes each),
 * fitting in the same 8 KB budget that formerly held 128 × 64-byte
 * Segments. */
#define SPECTRAL_GPU_SEG_CACHE_SIZE     256

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
/* CMSIS-DSP availability (arm_math.h expected when enabled). */
#ifndef SPECTRAL_USE_CMSIS
#if defined(ARM_MATH_CM7) || defined(ARM_MATH_CM4)
#define SPECTRAL_USE_CMSIS      1
#else
#define SPECTRAL_USE_CMSIS      0
#endif
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
#define SPECTRAL_CACHE_LINE_STRIDE (SPECTRAL_CACHE_LINE / sizeof(size_t))
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(SPECTRAL_CACHE_LINE % sizeof(size_t) == 0,
               "cache line must be multiple of size_t");
#endif
#ifndef SPECTRAL_ARM32_MAX_ACTIVE
#define SPECTRAL_ARM32_MAX_ACTIVE    512
#endif

/* Analysis Defaults */

#if SPECTRAL_EMBEDDED
#ifndef SPECTRAL_DEFAULT_N_FFT
#define SPECTRAL_DEFAULT_N_FFT      1024
#endif
#ifndef SPECTRAL_DEFAULT_HOP
#define SPECTRAL_DEFAULT_HOP        256
#endif
#ifndef SPECTRAL_DEFAULT_DB_THRESH
#define SPECTRAL_DEFAULT_DB_THRESH  (-70.0f)
#endif
#ifndef SPECTRAL_CACHE_ALIGN
#define SPECTRAL_CACHE_ALIGN        32
#endif
#else
#ifndef SPECTRAL_DEFAULT_N_FFT
#define SPECTRAL_DEFAULT_N_FFT      4096
#endif
#ifndef SPECTRAL_DEFAULT_HOP
#define SPECTRAL_DEFAULT_HOP        128
#endif
#ifndef SPECTRAL_DEFAULT_DB_THRESH
#define SPECTRAL_DEFAULT_DB_THRESH  (-85.0f)
#endif
#ifndef SPECTRAL_CACHE_ALIGN
#define SPECTRAL_CACHE_ALIGN        64
#endif
#endif

#endif /* SPECTRAL_CONFIG_H */
