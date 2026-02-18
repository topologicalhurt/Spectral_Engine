/* spectral_log.c - Canonical logging interface implementation */
#include "spectral_log.h"

const char* spectral_log_level_name(SpectralLogLevel level) {
    switch (level) {
    case SPECTRAL_LOG_LEVEL_ERROR: return "ERROR";
    case SPECTRAL_LOG_LEVEL_WARN:  return "WARNING";
    case SPECTRAL_LOG_LEVEL_INFO:  return "INFO";
    case SPECTRAL_LOG_LEVEL_DEBUG: return "DEBUG";
    case SPECTRAL_LOG_LEVEL_TRACE: return "TRACE";
    }
    return "LOG";
}

void spectral_log_v(SpectralLogLevel level,
                    FILE* stream,
                    int include_level_prefix,
                    int append_newline,
                    const char* fmt,
                    va_list args)
{
    FILE* out = stream ? stream : stdout;
    if (!fmt) return;

    if (include_level_prefix) {
        fprintf(out, "[%s] ", spectral_log_level_name(level));
    }
    vfprintf(out, fmt, args);
    if (append_newline) {
        fputc('\n', out);
    }
}

void spectral_log(SpectralLogLevel level,
                  FILE* stream,
                  int include_level_prefix,
                  int append_newline,
                  const char* fmt,
                  ...)
{
    va_list args;
    va_start(args, fmt);
    spectral_log_v(level, stream, include_level_prefix, append_newline, fmt, args);
    va_end(args);
}
