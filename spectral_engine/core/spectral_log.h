/* spectral_log.h - Canonical logging interface
 *
 * Centralized logging backend used across modules to keep formatting
 * and stream routing consistent.
 */
#ifndef SPECTRAL_LOG_H
#define SPECTRAL_LOG_H

#include <stdio.h>
#include <stdarg.h>

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

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_LOG_H */
