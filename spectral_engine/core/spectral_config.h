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
 * For SIMD detection, see spectral_oscillator_dispatch.h
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

/* Correctness-first approximation gates.
 * Keep these disabled by default. Enable them only with dedicated error-bound
 * and perceptual-regression tests for the target backend.
 */
/* TRIG is the exception: the canonical sine is a degree-9 odd minimax fold
 * (spectral_fast_sin_inline), default-ON because it is both faster and
 * SIMD-vectorizable while staying within ~1.4 ULP of libm over the oscillator's
 * operating range (8.3e-7 worst case over [-4pi,4pi]); see the EXACT_TRIG
 * guarantee in spectral_guarantees.h. Set to 0 to
 * restore exact libm sinf (reproducible/parity builds). */
#ifndef SPECTRAL_ENABLE_APPROX_TRIG
#define SPECTRAL_ENABLE_APPROX_TRIG 1
#endif
#ifndef SPECTRAL_ENABLE_APPROX_ATAN2
#define SPECTRAL_ENABLE_APPROX_ATAN2 0
#endif
#ifndef SPECTRAL_ENABLE_APPROX_INV_SQRT
#define SPECTRAL_ENABLE_APPROX_INV_SQRT 0
#endif
#ifndef SPECTRAL_ENABLE_APPROX_PEAK_LOG
#define SPECTRAL_ENABLE_APPROX_PEAK_LOG 0
#endif
/* Q15 oscillator phase-corner approximation. Default 0 = the packed 8x SIMD Q15
 * triangle/parabola evaluators reproduce the scalar/float result exactly at the
 * pq == Q15_MIN (phase = -pi) corner. Set to 1 in a fast/approximate build to skip
 * the exact-corner correction and accept the cheap 1-LSB overflow-guard clamp there.
 * This is the canonical example of the "exact default, cheap approximation only when
 * gated" policy (see AI_CANON rule 3): the int16 SIMD ops overflow at -32768, so the
 * default does a tiny masked corner fix-up while the gate trades 1 LSB for two fewer ops. */
#ifndef SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY
#define SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY 0
#endif
#ifndef SPECTRAL_METAL_FAST_MATH
#define SPECTRAL_METAL_FAST_MATH 0
#endif
/* Precise (cubic) MQ phase reconstruction gate. EXPERIMENTAL — NOT a
 * supported path (policy: REVIEWER_HANDOFF_2 sec 4.3).
 * When 0 (default) the synthesis phase model is the canonical per-interval
 * quadratic phase (McAulay-Quatieri 1986 per-frame interpolation); the cubic
 * coefficients reduce to (c2=beta, c3=0) so the synth path is bit-identical.
 * When 1, the adaptive_track_density stage may annotate segments with
 * cross-frame cubic phase coefficients (smooth C1 phase across hops) which the
 * synth path consumes. No build target sets this flag: the cubic path is
 * unbuilt and untested end-to-end (only its Segment serialization is covered,
 * by segment_endian_roundtrip). Do not enable in production. Its fate is
 * decided when the F synthesis-method stream is engaged (OPTIMISATION_PLAN
 * F3 per-track cubic-phase interpolation): wire+golden-sign-off, or delete. */
#ifndef SPECTRAL_PRECISE_PHASE
#define SPECTRAL_PRECISE_PHASE 0
#endif
/* Restricted profiling gate (single ownership).
 * Enabled only when restricted mode is active and restricted debug profiling
 * has been explicitly enabled by the build. */
#if SPECTRAL_RESTRICTED_MODE && defined(SPECTRAL_DEBUG_RESTRICTED) && SPECTRAL_DEBUG_RESTRICTED
#define SPECTRAL_RESTRICTED_PROFILE 1
#else
#define SPECTRAL_RESTRICTED_PROFILE 0
#endif

/* Embedded float mode (RESERVED -- not yet implemented). Intended to use the FPU
 * (Cortex-M4F/M7, VFPv4) for synthesis instead of Q15 integer, with storage kept
 * in Q15 for memory efficiency. No code branches on this gate yet, so defining it
 * changes nothing: the embedded_arm_float target currently compiles identically to
 * embedded_arm. Wire an #if SPECTRAL_EMBEDDED_FLOAT synthesis path (with a
 * divergence test) before treating the two targets as distinct. */
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

/* SoA active segment layout (phase_acc[], phase_inc[] as separate arrays) */
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

static inline float spectral_sample_to_float(spectral_sample_t s) {
    return (float)s * SPECTRAL_INV_Q15_SCALE;
}
/* NaN -> silence; out-of-range saturates (the embedded contract everywhere). */
static inline spectral_sample_t float_to_spectral_sample(float f) {
    if (f != f) return 0;
    if (f >= 1.0f) return SPECTRAL_SAMPLE_MAX;
    if (f <= -1.0f) return SPECTRAL_SAMPLE_MIN;
    return (spectral_sample_t)(f * SPECTRAL_Q15_SCALE);
}
static inline spectral_sample_t spectral_sample_add(spectral_sample_t a, spectral_sample_t b) {
    int32_t sum = (int32_t)a + (int32_t)b;
    if (sum > SPECTRAL_SAMPLE_MAX) return SPECTRAL_SAMPLE_MAX;
    if (sum < SPECTRAL_SAMPLE_MIN) return SPECTRAL_SAMPLE_MIN;
    return (spectral_sample_t)sum;
}
static inline spectral_sample_t spectral_sample_mul(spectral_sample_t a, spectral_sample_t b) {
    int32_t prod = ((int32_t)a * (int32_t)b) >> 15;
    if (prod > SPECTRAL_SAMPLE_MAX) return SPECTRAL_SAMPLE_MAX;
    if (prod < SPECTRAL_SAMPLE_MIN) return SPECTRAL_SAMPLE_MIN;
    return (spectral_sample_t)prod;
}

#define SPECTRAL_SAMPLE_IS_FIXED    1
#define SPECTRAL_SAMPLE_SPAN_FINITE(s, n) ((void)(s), (void)(n), 1)

static inline spectral_sample_t spectral_sample_lerp_f(spectral_sample_t s0, spectral_sample_t s1, float frac) {
    int32_t fq8 = (int32_t)(frac * 256.0f);
    return (spectral_sample_t)(s0 + ((((int32_t)s1 - (int32_t)s0) * fq8) >> 8));
}
static inline spectral_sample_t spectral_sample_lerp_q8(spectral_sample_t s0, spectral_sample_t s1, uint32_t frac_q8) {
    return (spectral_sample_t)(s0 + ((((int32_t)s1 - (int32_t)s0) * (int32_t)frac_q8) >> 8));
}

#else  /* Desktop: float samples */

typedef float    spectral_sample_t;
typedef double   spectral_acc_t;

#define SPECTRAL_SAMPLE_MAX         1.0f
#define SPECTRAL_SAMPLE_MIN         (-1.0f)
#define SPECTRAL_SAMPLE_ZERO        0.0f

static inline float spectral_sample_to_float(spectral_sample_t s) { return s; }
static inline spectral_sample_t float_to_spectral_sample(float f) { return f; }
static inline spectral_sample_t spectral_sample_add(spectral_sample_t a, spectral_sample_t b) {
    return a + b;
}
static inline spectral_sample_t spectral_sample_mul(spectral_sample_t a, spectral_sample_t b) {
    return a * b;
}

#define SPECTRAL_SAMPLE_IS_FIXED    0
#define SPECTRAL_SAMPLE_SPAN_FINITE(s, n) spectral_f32_span_finite((s), (n))

static inline spectral_sample_t spectral_sample_lerp_f(spectral_sample_t s0, spectral_sample_t s1, float frac) {
    return s0 + (s1 - s0) * frac;
}
static inline spectral_sample_t spectral_sample_lerp_q8(spectral_sample_t s0, spectral_sample_t s1, uint32_t frac_q8) {
    float frac_f = (float)frac_q8 / 256.0f;
    return s0 + (s1 - s0) * frac_f;
}

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
/* Segment width domain bound, enforced by spectral_segment_payload_valid.
 * width is a waveform shape parameter (quantized step density, pwm duty); the
 * synth boundary computes rads*width with |rads| <= pi+ulp, which becomes
 * int-conversion UB once width reaches ~2^31/pi (the INT_MAX defect class
 * reproduced by test_segment_payload_bounds). 2^24 is the largest float with exactly
 * representable unit steps -- a step count above it cannot be represented
 * distinctly anyway -- and keeps pi*width ~40x under the UB envelope. The bound
 * is on |width| so no future consumer inherits the boundary class. */
#ifndef SPECTRAL_SEGMENT_WIDTH_MAX
#define SPECTRAL_SEGMENT_WIDTH_MAX      16777216.0f
#endif
#ifndef SPECTRAL_TRACK_INTERP_POWER_RATIONAL
#define SPECTRAL_TRACK_INTERP_POWER_RATIONAL 0
#endif
/* Per-frame candidate scratch batch for peak tracking.
 * [chosen: 128 entries keeps the candidate block cache-resident; the prefetch
 * benchmark profiles (benchmark_workflow) sweep 64-256 for re-evaluation.] */
#ifndef SPECTRAL_TRACK_CANDIDATE_BATCH
#define SPECTRAL_TRACK_CANDIDATE_BATCH  128u
#endif
#ifndef SPECTRAL_TRACK_INITIAL_SEG_CAP
#define SPECTRAL_TRACK_INITIAL_SEG_CAP  65536u
#endif
#ifndef SPECTRAL_TRACK_PREFETCH_LOOKAHEAD
#define SPECTRAL_TRACK_PREFETCH_LOOKAHEAD 12u
#endif
/* Software-prefetch lookahead for the SIMD peak pre-scan over the float magsq
 * spectrum (spectral_peak_track.c: prefetch row + f + 8/4 + DISTANCE).
 * [chosen: 48 floats = 192 B ~= 3 x 64 B cache lines of read-ahead, enough to
 *  cover the loadu/compare latency of the streaming scan.] */
#ifndef SPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE
#define SPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE 48u
#endif
/* `locality` argument of __builtin_prefetch(addr, 0, locality) (0..3 per the
 * GCC/Clang builtin: 0 = no temporal reuse/NTA, 3 = keep in all cache levels).
 * [chosen: 2 = moderate temporal locality; each magsq row is touched once in a
 *  streaming scan, so it should land in L2/L3 without evicting hot L1 lines.] */
#ifndef SPECTRAL_TRACK_PREFETCH_READ_LOCALITY
#define SPECTRAL_TRACK_PREFETCH_READ_LOCALITY 2
#endif
#ifndef SPECTRAL_TRACK_PREFETCH_PHASE
#define SPECTRAL_TRACK_PREFETCH_PHASE 1
#endif
#ifndef SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE
#define SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE 16u
#endif
#ifndef SPECTRAL_TRACK_PAIR_OMP_CHUNK
#define SPECTRAL_TRACK_PAIR_OMP_CHUNK 0u
#endif
#ifndef SPECTRAL_TRACK_DEBUG_TIMING
#define SPECTRAL_TRACK_DEBUG_TIMING 0
#endif
#ifndef SPECTRAL_PEAK_PHASE_CONSISTENCY_TOL_RADS
#define SPECTRAL_PEAK_PHASE_CONSISTENCY_TOL_RADS 0.39269908169872414f /* pi/8 */
#endif
#ifndef SPECTRAL_PEAK_PHASE_POLICY_DEFAULT
#define SPECTRAL_PEAK_PHASE_POLICY_DEFAULT SPECTRAL_PEAK_PHASE_POLICY_OBSERVE
#endif
#ifndef SPECTRAL_PEAK_AMP_POLICY_DEFAULT
#define SPECTRAL_PEAK_AMP_POLICY_DEFAULT SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED
#endif
#ifndef SPECTRAL_PEAK_AMP_MAX_GAIN
#define SPECTRAL_PEAK_AMP_MAX_GAIN 8.0f
#endif
/* STFT-size trip point for AUTO path selection (spectral_analysis.c:
 * use_fused_path = total_bins > THRESHOLD). Unit is BINS.
 * [derived: the full-matrix path materializes 2 floats/bin (magsq + phase) = 8 B,
 *  so 32 Mi bins x 8 B = 256 MiB STFT; above that the fused SPSC path is used to
 *  avoid the full-matrix footprint.] */
#ifndef SPECTRAL_STFT_CHUNK_THRESHOLD
#define SPECTRAL_STFT_CHUNK_THRESHOLD   (32ul * 1024ul * 1024ul)
#endif
/* Segment-merge size above which the contiguous merge buffer is page-aligned and
 * kernel pre-faulted (spectral_peak_track.c, compared to merge_bytes). Unit is
 * BYTES. [chosen: 64 MiB; below it ordinary malloc is fine, above it the
 * ~63K minor page faults of the parallel copy are worth the posix_memalign +
 * MADV_POPULATE_WRITE / pre-touch path.] */
#ifndef SPECTRAL_PRETOUCH_THRESHOLD
#define SPECTRAL_PRETOUCH_THRESHOLD     (64ul * 1024ul * 1024ul)
#endif
/* Fallback page size for the pre-touch path. The merge code prefers the runtime
 * sysconf(_SC_PAGESIZE) and only falls back to this macro if sysconf fails;
 * 4096 is the classic 4 KiB page. [chosen: portable default for hosts without a
 * usable sysconf; NOTE Apple Silicon uses 16 KiB pages, which is exactly why the
 * runtime query is preferred over this literal.] */
#ifndef SPECTRAL_PRETOUCH_PAGE_SIZE
#define SPECTRAL_PRETOUCH_PAGE_SIZE     4096u
#endif


/* Canonical embedded debug LED timing defaults (milliseconds) */
#ifndef SPECTRAL_ERROR_BLINK_ON_MS
#define SPECTRAL_ERROR_BLINK_ON_MS      100u
#endif
#ifndef SPECTRAL_ERROR_BLINK_OFF_MS
#define SPECTRAL_ERROR_BLINK_OFF_MS     100u
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
/* Active fade length for the current build profile. Profile selection lives here
 * (the canonical profile header), not in each synthesis source, so the synthesis
 * files stay free of SPECTRAL_EMBEDDED branching. */
#ifndef SPECTRAL_FADE_SAMPLES_ACTIVE
#if SPECTRAL_EMBEDDED
#define SPECTRAL_FADE_SAMPLES_ACTIVE SPECTRAL_FADE_SAMPLES_EMBEDDED
#else
#define SPECTRAL_FADE_SAMPLES_ACTIVE SPECTRAL_FADE_SAMPLES_DESKTOP
#endif
#endif
/* Optimization level: 0=safe, 1=balanced (default), 2=aggressive, 3=reserved. */
#ifndef SPECTRAL_OPT_LEVEL
#define SPECTRAL_OPT_LEVEL      1
#endif

/* GPU/compute block size for Metal/CUDA backends.
 * [chosen: a multiple of the 32-wide SIMD-group/warp on both backends, at
 * half the common 1024 threads/group ceiling for register headroom; not yet
 * swept on hardware — revisit if a GPU profile shows occupancy pressure.] */
#ifndef SPECTRAL_GPU_TILE_SIZE
#define SPECTRAL_GPU_TILE_SIZE          512
#endif

/* Max deterministic work partitions for CPU synthesis reduction.
 * 0 keeps partition count aligned to the selected thread count. */
#ifndef SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS
#define SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS 0
#endif

/* Headroom factor for normalization (0.95 = -0.45 dB).
 * [chosen: keeps reconstructed inter-sample peaks from clipping after
 * normalization to the sample-domain max.] */
#ifndef SPECTRAL_NORMALIZE_HEADROOM
#define SPECTRAL_NORMALIZE_HEADROOM     0.95f
#endif

/* Embedded simulation headroom for Q15 amplitude scaling.
 * [chosen: scales the peak segment amplitude to just under Q15 full scale so
 * float_to_spectral_sample saturation never engages on the nominal peak; the
 * 1% margin absorbs envelope/rounding overshoot.] */
#define SPECTRAL_SIMULATION_HEADROOM    0.99f

/* GPU segment cache size (threadgroup / shared memory).
 * Both Metal and CUDA use 256 entries of SegmentGpu (32 bytes each),
 * fitting in the same 8 KB budget that formerly held 128 × 64-byte
 * Segments. */
#define SPECTRAL_GPU_SEG_CACHE_SIZE     256

/* Platform Detection */

/* Overridable so a host build can force the M7 codepath for host-sim/testing
 * (-DSPECTRAL_ARM_M7=1). Defaults to real-hardware detection otherwise. */
#ifndef SPECTRAL_ARM_M7
#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
#define SPECTRAL_ARM_M7         1
#else
#define SPECTRAL_ARM_M7         0
#endif
#endif

/* Default: Accelerate/vDSP on Apple, portable elsewhere. A build may force the
 * portable path by pre-defining SPECTRAL_USE_VDSP=0 (e.g. to exercise the
 * source-visible window/FFT formulas without linking Accelerate). */
#ifndef SPECTRAL_USE_VDSP
#ifdef __APPLE__
#define SPECTRAL_USE_VDSP       1
#else
#define SPECTRAL_USE_VDSP       0
#endif
#endif
/* CMSIS-DSP availability (arm_math.h expected when enabled). */
#ifndef SPECTRAL_USE_CMSIS
#if defined(ARM_MATH_CM7) || defined(ARM_MATH_CM4)
#define SPECTRAL_USE_CMSIS      1
#else
#define SPECTRAL_USE_CMSIS      0
#endif
#endif

/* ---- Hardware capability flags --------------------------------------------------------
 * Core kernels gate on CAPABILITIES, never on a specific CPU. The platform detection above
 * is the ONLY place that maps a device/arch to capabilities; downstream code asks "do I have
 * capability X?". This is the Linux-kernel model: optimize the bulk target first (here the
 * low-level ARM A/M profile), then add a per-arch hierarchy of overrides. A new arch (another
 * ARM profile, later RISC-V) is supported by extending THIS mapping, not by editing kernels.
 * Define a capability =1 only where the operation has a MEASURED benefit on that arch.
 *
 * Capability candidates as the embedded surface grows (add when there is a real benefit, and
 * route the kernels through them rather than through CPU macros): SPECTRAL_HAS_DUAL_MAC (done),
 * a packed-SIMD capability (NEON / MVE / RVV) for the float and 8xQ15 kernels, a hardware-FPU
 * capability, and a fast-tightly-coupled-memory capability (DTCM/ITCM presence). Until verified,
 * those stay expressed by their existing CPU/feature macros; do not add speculative ones. */

/* Dual 16-bit MAC: two Q15 products into one accumulate (ARM DSP-extension SMLAD/SMLALD).
 * Verified benefit on Cortex-M7: halves the additive-synth MAC instructions (smlald codegen-
 * confirmed). Derived from the ARM ACLE feature macro; another arch would set it from its own
 * detection. Kernels gate the voice-pairing dual-MAC path on this, not on SPECTRAL_ARM_M7. */
#ifndef SPECTRAL_HAS_DUAL_MAC
#if defined(__ARM_FEATURE_DSP) && __ARM_FEATURE_DSP
#define SPECTRAL_HAS_DUAL_MAC   1
#else
#define SPECTRAL_HAS_DUAL_MAC   0
#endif
#endif

/* ARM32 embedded configuration */

/* Memory-class placement (device-agnostic intent; the concrete device binding
 * lives in a BSP, not here). Defines SPECTRAL_MEM_FAST / SPECTRAL_MEM_FAST_CODE /
 * SPECTRAL_MEM_BULK / SPECTRAL_CACHE_LINE. */
#include "spectral_mem.h"
/* Static bound on the arm32 kernel's active-voice state arrays. The REAL-TIME
 * cap is the WCET-gated DAISY_MAX_ACTIVE (128, capacity table in
 * M7_PERF_MODEL_PLAN.md); this only sizes storage. [chosen: 4x the runtime
 * cap so batch/offline use is not storage-bound.] */
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
#endif

/* Allocation alignment follows the cache-line SSOT (spectral_mem.h, included
 * above): one quantity, so the simulate host (cache-line 64) cannot disagree with
 * a 32-byte alloc alignment. A BSP that overrides SPECTRAL_CACHE_LINE moves both. */
#ifndef SPECTRAL_CACHE_ALIGN
#define SPECTRAL_CACHE_ALIGN        SPECTRAL_CACHE_LINE
#endif

#endif /* SPECTRAL_CONFIG_H */
