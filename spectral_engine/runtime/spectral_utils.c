/* spectral_utils.c - Generic utility helpers
 *
 * Implements formatting, parsing/path helpers, timbre naming,
 * and debug output utilities (non-console module).
 */
#include "spectral_utils.h"
#include "spectral_log.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>

/* 
 * String Formatting
 */

const char* format_bytes(size_t bytes, char* buf, size_t buf_size) {
    if (bytes >= SPECTRAL_BYTES_PER_GIB) {
        snprintf(buf, buf_size, "%.1f GB", BYTES_TO_GB(bytes));
    } else if (bytes >= SPECTRAL_BYTES_PER_MIB) {
        snprintf(buf, buf_size, "%.1f MB", BYTES_TO_MB(bytes));
    } else if (bytes >= SPECTRAL_BYTES_PER_KIB) {
        snprintf(buf, buf_size, "%.1f KB", BYTES_TO_KB(bytes));
    } else {
        snprintf(buf, buf_size, "%zu B", bytes);
    }
    return buf;
}

const char* format_duration_ms(double ms, char* buf, size_t buf_size) {
    if (ms >= SPECTRAL_MILLIS_PER_SECOND_D) {
        snprintf(buf, buf_size, "%.2f s", ms / SPECTRAL_MILLIS_PER_SECOND_D);
    } else if (ms >= 1.0) {
        snprintf(buf, buf_size, "%.1f ms", ms);
    } else {
        snprintf(buf, buf_size, "%.0f us", ms * SPECTRAL_MICROS_PER_MILLI_D);
    }
    return buf;
}

const char* format_percent(double pct, char* buf, size_t buf_size) {
    snprintf(buf, buf_size, "%.1f%%", pct);
    return buf;
}

static const char* spectral_nonempty_or_default(const char* s, const char* fallback) {
    return (s && s[0]) ? s : fallback;
}

static size_t spectral_appendf(char* buf, size_t buf_size, size_t used, const char* fmt, ...) {
    int w = 0;
    va_list args;
    if (!buf || buf_size == 0 || used >= buf_size || !fmt) return used;
    va_start(args, fmt);
    w = vsnprintf(buf + used, buf_size - used, fmt, args);
    va_end(args);
    if (w < 0) return used;
    if ((size_t)w >= buf_size - used) return buf_size - 1;
    return used + (size_t)w;
}

const char* spectral_format_resolution_context(
    char* buf,
    size_t buf_size,
    const SpectralResolutionContext* ctx)
{
    SpectralResolutionContext empty = {0};
    size_t used = 0;

    if (!buf || buf_size == 0) return NULL;
    if (!ctx) ctx = &empty;

    used = spectral_appendf(buf, buf_size, used, "event=\"%s\"",
                            spectral_nonempty_or_default(ctx->event, SPECTRAL_RESOLUTION_EVENT_RESOLUTION));
    used = spectral_appendf(buf, buf_size, used, " scope=\"%s\"",
                            spectral_nonempty_or_default(ctx->scope, SPECTRAL_RESOLUTION_SCOPE_SELECTION));
    if (!spectral_is_empty_string(ctx->backend)) {
        used = spectral_appendf(buf, buf_size, used, " backend=\"%s\"", ctx->backend);
    }
    used = spectral_appendf(buf, buf_size, used, " requested=\"%s\"",
                            spectral_nonempty_or_default(ctx->requested_label, SPECTRAL_RESOLUTION_LABEL_UNKNOWN));
    if (ctx->requested_id >= 0) {
        used = spectral_appendf(buf, buf_size, used, " requested_id=%d", ctx->requested_id);
    }
    used = spectral_appendf(buf, buf_size, used, " effective=\"%s\"",
                            spectral_nonempty_or_default(ctx->effective_label, SPECTRAL_RESOLUTION_LABEL_UNKNOWN));
    if (ctx->effective_id >= 0) {
        used = spectral_appendf(buf, buf_size, used, " effective_id=%d", ctx->effective_id);
    }
    if (ctx->max_supported_id >= 0) {
        used = spectral_appendf(buf, buf_size, used, " max_supported_id=%d",
                                ctx->max_supported_id);
    }
    if (!spectral_is_empty_string(ctx->mode)) {
        used = spectral_appendf(buf, buf_size, used, " mode=\"%s\"", ctx->mode);
    }
    if (!spectral_is_empty_string(ctx->reason)) {
        used = spectral_appendf(buf, buf_size, used, " reason=\"%s\"", ctx->reason);
    }
    if (used >= buf_size) {
        buf[buf_size - 1] = '\0';
    }
    return buf;
}

const char* spectral_format_expected_actual(char* buf, size_t buf_size,
                                            const char* field,
                                            const char* expected,
                                            const char* actual) {
    if (!buf || buf_size == 0) return NULL;
    snprintf(buf, buf_size, "%s: expected %s, got %s",
             spectral_nonempty_or_default(field, "value"),
             spectral_nonempty_or_default(expected, "valid value"),
             spectral_nonempty_or_default(actual, "unknown"));
    return buf;
}

int spectral_is_empty_string(const char* s) {
    return (!s || !s[0]);
}

int spectral_path_join(char* out, size_t out_size, const char* a, const char* b) {
    if (!out || out_size == 0 || !a || !b) return 0;
    {
        int w = snprintf(out, out_size, "%s/%s", a, b);
        return (w > 0 && (size_t)w < out_size);
    }
}

const char* spectral_basename_no_ext(const char* path, char* out, size_t out_size) {
    const char* base;
    const char* dot;
    size_t len;
    if (!path || !out || out_size == 0) return NULL;
    base = strrchr(path, '/');
    base = base ? (base + 1) : path;
    dot = strrchr(base, '.');
    len = (dot && dot > base) ? (size_t)(dot - base) : strlen(base);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, base, len);
    out[len] = '\0';
    return out;
}

int parse_u32_arg(const char* s, uint32_t* out) {
    char* end = NULL;
    unsigned long v;
    if (!s || !out) return 0;
    errno = 0;
    v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || (end && *end != '\0') || v > UINT32_MAX) {
        return 0;
    }
    *out = (uint32_t)v;
    return 1;
}

int parse_size_arg(const char* s, size_t* out) {
    char* end = NULL;
    unsigned long long v;
    if (!s || !out) return 0;
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || (end && *end != '\0') || v > (unsigned long long)SIZE_MAX) {
        return 0;
    }
    *out = (size_t)v;
    return 1;
}

int parse_i32_arg(const char* s, int* out) {
    char* end = NULL;
    long v;
    if (!s || !out) return 0;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || (end && *end != '\0') || v < INT_MIN || v > INT_MAX) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

int parse_f32_arg(const char* s, float* out) {
    char* end = NULL;
    float v;
    if (!s || !out) return 0;
    errno = 0;
    v = strtof(s, &end);
    if (errno != 0 || end == s || (end && *end != '\0') || !spectral_is_finite_f32(v)) {
        return 0;
    }
    *out = v;
    return 1;
}

const char* spectral_getenv_nonempty(const char* key) {
    const char* value;
    if (!key || key[0] == '\0') return NULL;
    value = getenv(key);
    return spectral_is_empty_string(value) ? NULL : value;
}

int spectral_getenv_f64(const char* key, double* out) {
    const char* value = spectral_getenv_nonempty(key);
    char* end = NULL;
    double parsed;
    if (!out || !value) return 0;
    errno = 0;
    parsed = strtod(value, &end);
    if (errno != 0 || end == value || (end && *end != '\0') || !spectral_is_finite_f64(parsed)) {
        return 0;
    }
    *out = parsed;
    return 1;
}

int spectral_getenv_f64_positive(const char* key, double* out) {
    double value = 0.0;
    if (!out || !spectral_getenv_f64(key, &value)) return 0;
    if (!spectral_is_finite_positive_f64(value)) return 0;
    *out = value;
    return 1;
}

static int spectral_parse_bool_str(const char* s, int* out) {
    char lowered[8] = {0};
    size_t n = 0;
    if (!s || !out) return 0;
    if (strcmp(s, "1") == 0) { *out = 1; return 1; }
    if (strcmp(s, "0") == 0) { *out = 0; return 1; }

    while (s[n] != '\0' && n < sizeof(lowered) - 1) {
        lowered[n] = (char)tolower((unsigned char)s[n]);
        n++;
    }
    lowered[n] = '\0';
    if (s[n] != '\0') return 0;

    if (strcmp(lowered, "true") == 0 || strcmp(lowered, "yes") == 0 || strcmp(lowered, "on") == 0) {
        *out = 1;
        return 1;
    }
    if (strcmp(lowered, "false") == 0 || strcmp(lowered, "no") == 0 || strcmp(lowered, "off") == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

int spectral_getenv_bool(const char* key, int* out) {
    const char* value = spectral_getenv_nonempty(key);
    if (!value || !out) return 0;
    return spectral_parse_bool_str(value, out);
}

/* 
 * Timbre Names
 */

const char* timbre_name(SpectralTimbre timbre) {
    static const char* names[TIMBRE_COUNT] = {
        "sine", "saw", "square", "triangle",
        "asin", "parabola", "quantized", "pwm"
    };
    unsigned int idx = (unsigned int)timbre;
    return (idx < TIMBRE_COUNT) ? names[idx] : "unknown";
}

/*
 * Debug Printing Implementation
 */

#ifndef NDEBUG
static uint8_t debug_once_flags[SPECTRAL_DEBUG_ONCE_MAX] = {0};
#endif

#ifndef NDEBUG
static SpectralLogLevel spectral_log_level_from_debug_level(SpectralDebugLevel level) {
    switch (level) {
    case DEBUG_LEVEL_ERROR: return SPECTRAL_LOG_LEVEL_ERROR;
    case DEBUG_LEVEL_WARN:  return SPECTRAL_LOG_LEVEL_WARN;
    case DEBUG_LEVEL_INFO:  return SPECTRAL_LOG_LEVEL_DEBUG;
    case DEBUG_LEVEL_TRACE: return SPECTRAL_LOG_LEVEL_TRACE;
    }
    return SPECTRAL_LOG_LEVEL_DEBUG;
}
#endif

void spectral_debug(SpectralDebugLevel level, const char* fmt, ...) {
#if SPECTRAL_DEBUG && !defined(NDEBUG)
    va_list args;
    va_start(args, fmt);
    spectral_log_v(spectral_log_level_from_debug_level(level), stderr, 1, 1, fmt, args);
    va_end(args);
#else
    (void)level;
    (void)fmt;
#endif
}

void spectral_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    spectral_log_v(SPECTRAL_LOG_LEVEL_WARN, stderr, 1, 1, fmt, args);
    va_end(args);
}

int spectral_debug_once(int id, SpectralDebugLevel level, const char* fmt, ...) {
#ifndef NDEBUG
    if (id < 0 || id >= SPECTRAL_DEBUG_ONCE_MAX) return 0;
    if (debug_once_flags[id]) return 0;
    
    debug_once_flags[id] = 1;

    va_list args;
    va_start(args, fmt);
    spectral_log_v(spectral_log_level_from_debug_level(level), stderr, 1, 1, fmt, args);
    va_end(args);
    return 1;
#else
    (void)id;
    (void)level;
    (void)fmt;
    return 0;
#endif
}
