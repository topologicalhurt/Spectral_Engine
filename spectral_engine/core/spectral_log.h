/* spectral_log.h - Canonical logging interface
 *
 * Centralized logging backend used across modules to keep formatting
 * and stream routing consistent.
 */
#ifndef SPECTRAL_LOG_H
#define SPECTRAL_LOG_H

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPECTRAL_LOG_LEVEL_ERROR = 0,
    SPECTRAL_LOG_LEVEL_WARN  = 1,
    SPECTRAL_LOG_LEVEL_INFO  = 2,
    SPECTRAL_LOG_LEVEL_DEBUG = 3,
    SPECTRAL_LOG_LEVEL_TRACE = 4
} SpectralLogLevel;

const char* spectral_log_level_name(SpectralLogLevel level);

void spectral_log_v(SpectralLogLevel level,
                    FILE* stream,
                    int include_level_prefix,
                    int append_newline,
                    const char* fmt,
                    va_list args);

void spectral_log(SpectralLogLevel level,
                  FILE* stream,
                  int include_level_prefix,
                  int append_newline,
                  const char* fmt,
                  ...);

#define SPECTRAL_LOG_INFO(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_INFO, stdout, 0, 1, __VA_ARGS__)
#define SPECTRAL_LOG_WARN(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_WARN, stdout, 0, 1, __VA_ARGS__)
#define SPECTRAL_LOG_ERROR(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_ERROR, stdout, 0, 1, __VA_ARGS__)
#define SPECTRAL_LOG_DEBUG(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_DEBUG, stdout, 0, 1, __VA_ARGS__)
#define SPECTRAL_LOG_TRACE(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_TRACE, stdout, 0, 1, __VA_ARGS__)

#define SPECTRAL_LOG_WARN_STDERR(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_WARN, stderr, 0, 1, __VA_ARGS__)
#define SPECTRAL_LOG_ERROR_STDERR(...) \
    spectral_log(SPECTRAL_LOG_LEVEL_ERROR, stderr, 0, 1, __VA_ARGS__)

static inline uint64_t spectral_log_get_monotonic_ns(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
#endif
    return 0;
}

static inline void spectral_log_emit_stage_marker(const char* kind, const char* stage_name, uint64_t mono_ns) {
    fprintf(stderr, "SPECTRAL_STAGE_%s %s %llu\n",
            kind, stage_name, (unsigned long long)mono_ns);
    fflush(stderr);
}

#define SPECTRAL_STAGE_BEGIN_AT(stage_name, mono_ns) \
    spectral_log_emit_stage_marker("BEGIN", (stage_name), (uint64_t)(mono_ns))

#define SPECTRAL_STAGE_END_AT(stage_name, mono_ns) \
    spectral_log_emit_stage_marker("END", (stage_name), (uint64_t)(mono_ns))

#define SPECTRAL_STAGE_BEGIN(stage_name) \
    SPECTRAL_STAGE_BEGIN_AT((stage_name), spectral_log_get_monotonic_ns())

#define SPECTRAL_STAGE_END(stage_name) \
    SPECTRAL_STAGE_END_AT((stage_name), spectral_log_get_monotonic_ns())

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_LOG_H */
