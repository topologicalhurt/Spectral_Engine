/* spectral_utils.h - Generic utility helpers
 *
 * This module provides reusable helpers for:
 *   - Human-readable formatting for bytes, durations, percentages
 *   - Argument/path parsing and finite/overflow checks
 *   - Timbre name lookup
 *   - Debug logging helpers
 *
 * Console/table/progress/status APIs are defined in spectral_console.h.
 */
#ifndef SPECTRAL_UTILS_H
#define SPECTRAL_UTILS_H

#include "spectral_config.h"
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Locale-independent ASCII lowercase: maps A-Z to a-z and leaves every other
 * byte (digits, punctuation, UTF-8 continuation bytes) untouched. Use this for
 * case-insensitive matching of ASCII tokens (option names, file suffixes);
 * libc tolower() depends on the active locale and can reclassify bytes outside
 * the C locale, which would make token matching locale-sensitive. */
static inline char spectral_ascii_tolower(char c) {
    return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
}

/*
 * Byte-size conversion macros
 */

#define SPECTRAL_BYTES_PER_KIB ((size_t)1024u)
#define SPECTRAL_BYTES_PER_MIB (SPECTRAL_BYTES_PER_KIB * SPECTRAL_BYTES_PER_KIB)
#define SPECTRAL_BYTES_PER_GIB (SPECTRAL_BYTES_PER_MIB * SPECTRAL_BYTES_PER_KIB)

#define BYTES_TO_KB(b)  ((double)(b) / (double)SPECTRAL_BYTES_PER_KIB)
#define BYTES_TO_MB(b)  ((double)(b) / (double)SPECTRAL_BYTES_PER_MIB)
#define BYTES_TO_GB(b)  ((double)(b) / (double)SPECTRAL_BYTES_PER_GIB)

static inline double spectral_bandwidth_gibps(size_t bytes, double elapsed_sec) {
    if (elapsed_sec <= 0.0) return 0.0;
    return BYTES_TO_GB(bytes) / elapsed_sec;
}

#define SPECTRAL_MILLIS_PER_SECOND_F 1000.0f
#define SPECTRAL_MILLIS_PER_SECOND_D 1000.0
#define SPECTRAL_MICROS_PER_MILLI_D  1000.0

/*
 * String Formatting
 */

const char* format_bytes(size_t bytes, char* buf, size_t buf_size);
const char* format_duration_ms(double ms, char* buf, size_t buf_size);
const char* format_percent(double pct, char* buf, size_t buf_size);

typedef struct SpectralResolutionContext {
    const char* event;             /* e.g. fallback, resolution */
    const char* backend;           /* optional backend name */
    const char* scope;             /* e.g. backend, timbre */
    const char* requested_label;   /* requested symbolic value */
    int         requested_id;      /* <0 to omit */
    const char* effective_label;   /* effective symbolic value */
    int         effective_id;      /* <0 to omit */
    int         max_supported_id;  /* <0 to omit */
    const char* mode;              /* optional execution mode */
    const char* reason;            /* optional explanatory note */
} SpectralResolutionContext;

const char* spectral_format_resolution_context(
    char* buf,
    size_t buf_size,
    const SpectralResolutionContext* ctx);

const char* spectral_format_expected_actual(char* buf, size_t buf_size,
                                            const char* field,
                                            const char* expected,
                                            const char* actual);
int         spectral_is_empty_string(const char* s);
int         spectral_path_join(char* out, size_t out_size, const char* a, const char* b);
const char* spectral_basename_no_ext(const char* path, char* out, size_t out_size);
int         parse_u32_arg(const char* s, uint32_t* out);
int         parse_size_arg(const char* s, size_t* out);
int         parse_i32_arg(const char* s, int* out);
int         parse_f32_arg(const char* s, float* out);

const char* spectral_getenv_nonempty(const char* key);
int         spectral_getenv_bool(const char* key, int* out);
int         spectral_getenv_f64(const char* key, double* out);
int         spectral_getenv_f64_positive(const char* key, double* out);

/* Inlined finiteness via the exponent bits (all-ones exponent == Inf or NaN). This is a real
 * check that READS the bit pattern, so unlike isfinite() it (a) inlines to a couple of integer
 * ops instead of a libm __isfinitef/__isfinited call — the hottest single cost in the analysis
 * profile — and (b) is not elided to `true` under -ffinite-math-only (-ffast-math), so the
 * validation guards keep working. */
static inline int spectral_is_finite_f32(float v) {
    uint32_t u;
    __builtin_memcpy(&u, &v, sizeof u);
    return (u & 0x7F800000u) != 0x7F800000u;
}

static inline int spectral_is_finite_f64(double v) {
    uint64_t u;
    __builtin_memcpy(&u, &v, sizeof u);
    return (u & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

static inline int spectral_is_finite_positive_f32(float v) {
    return spectral_is_finite_f32(v) && v > 0.0f;
}

static inline int spectral_is_finite_positive_f64(double v) {
    return spectral_is_finite_f64(v) && v > 0.0;
}

/* Type-dispatched, inlined drop-in for libm isfinite() in hot paths: routes float/double to
 * the bit-trick helpers above so the validation-heavy estimator/tracker never emit a libm
 * __isfinitef/__isfinited call. The _Generic controlling expression is unevaluated, so x is
 * evaluated exactly once. */
#define SPECTRAL_ISFINITE(x) (_Generic((x), \
    float:   spectral_is_finite_f32, \
    double:  spectral_is_finite_f64, \
    default: spectral_is_finite_f64)(x))

static inline float spectral_clamp_f32(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline double spectral_clamp_f64(double v, double lo, double hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float spectral_finite_or_fallback_f32(float v, float fallback, int* used_fallback) {
    if (!spectral_is_finite_f32(v)) {
        if (used_fallback) *used_fallback = 1;
        return fallback;
    }
    return v;
}

static inline float spectral_sanitize_finite_clamp_f32(
    float v, float lo, float hi, float fallback, int* used_fallback)
{
    return spectral_clamp_f32(spectral_finite_or_fallback_f32(v, fallback, used_fallback), lo, hi);
}

static inline int spectral_size_add(size_t a, size_t b, size_t* out) {
    if (!out) return 0;
#if defined(__has_builtin)
#if __has_builtin(__builtin_add_overflow)
    return !__builtin_add_overflow(a, b, out);
#endif
#endif
    if (a > SIZE_MAX - b) return 0;
    *out = a + b;
    return 1;
}

static inline int spectral_size_mul(size_t a, size_t b, size_t* out) {
    if (!out) return 0;
#if defined(__has_builtin)
#if __has_builtin(__builtin_mul_overflow)
    return !__builtin_mul_overflow(a, b, out);
#endif
#endif
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *out = a * b;
    return 1;
}

static inline int spectral_array_bytes(size_t count, size_t element_size, size_t* out) {
    return spectral_size_mul(count, element_size, out);
}

/* Canonical growth policy for grow-only caches/buffers: cap <- required * 1.5 */
static inline int spectral_next_capacity_3_over_2(size_t required, size_t* out_capacity) {
    size_t headroom = required / 2u;
    if (!out_capacity) return 0;
    return spectral_size_add(required, headroom, out_capacity);
}

static inline void* spectral_malloc_array(size_t count, size_t element_size) {
    size_t bytes = 0;
    if (!spectral_array_bytes(count, element_size, &bytes)) return NULL;
    return malloc(bytes);
}

static inline void* spectral_calloc_array(size_t count, size_t element_size) {
    size_t bytes = 0;
    if (!spectral_array_bytes(count, element_size, &bytes)) return NULL;
    return calloc(1, bytes);
}

static inline void* spectral_realloc_array(void* ptr, size_t count, size_t element_size) {
    size_t bytes = 0;
    if (!spectral_array_bytes(count, element_size, &bytes)) return NULL;
    return realloc(ptr, bytes);
}

#define SPECTRAL_RETURN_IF(cond) do { if (cond) return; } while (0)
#define SPECTRAL_RETURN_VAL_IF(cond, value) do { if (cond) return (value); } while (0)

/* 
 * Timbre Names
 */

const char* timbre_name(SpectralTimbre timbre);

/*
 * Debug Printing
 * 
 * - spectral_debug(): Prints when SPECTRAL_DEBUG=1 and NDEBUG not defined
 * - spectral_warn(): Always prints to stderr (errors/warnings)
 * - spectral_debug_once(): Prints once per unique id
 * 
 * Build flags:
 *   -DSPECTRAL_DEBUG=1  Enable debug output
 *   -DNDEBUG            Disable all debug output (release builds)
 */

#ifndef SPECTRAL_DEBUG
#define SPECTRAL_DEBUG 0
#endif

/* Debug levels for filtering output */
typedef enum {
    DEBUG_LEVEL_ERROR = 0,
    DEBUG_LEVEL_WARN  = 1,
    DEBUG_LEVEL_INFO  = 2,
    DEBUG_LEVEL_TRACE = 3
} SpectralDebugLevel;

void spectral_debug(SpectralDebugLevel level, const char* fmt, ...);
void spectral_warn(const char* fmt, ...);
int  spectral_debug_once(int id, SpectralDebugLevel level, const char* fmt, ...);

/* Convenience macros with automatic level */
#if SPECTRAL_DEBUG && !defined(NDEBUG)
#define SPECTRAL_DBG(...)       spectral_debug(DEBUG_LEVEL_INFO, __VA_ARGS__)
#define SPECTRAL_DBG_TRACE(...) spectral_debug(DEBUG_LEVEL_TRACE, __VA_ARGS__)
#else
#define SPECTRAL_DBG(...)       ((void)0)
#define SPECTRAL_DBG_TRACE(...) ((void)0)
#endif

#define SPECTRAL_WARN(...)      spectral_warn(__VA_ARGS__)
#define SPECTRAL_WARN_ONCE(id, ...) spectral_debug_once((id), DEBUG_LEVEL_WARN, __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_UTILS_H */
