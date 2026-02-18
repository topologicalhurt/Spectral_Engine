/* spectral_error.c - Error code string conversion */
#include "spectral_error.h"
#include "spectral_log.h"
#include <stdarg.h>
#include <stdio.h>

static const char* spectral_core_error_message(int code)
{
    switch ((SpectralError)code) {
        case SPECTRAL_OK:                  return "OK";
        case SPECTRAL_ERR_PARAM:           return "invalid parameter";
        case SPECTRAL_ERR_MEMORY:          return "memory allocation failed";
        case SPECTRAL_ERR_OVERFLOW:        return "buffer overflow";
        case SPECTRAL_ERR_NOTINIT:         return "not initialized";
        case SPECTRAL_ERR_BUSY:            return "resource busy";
        case SPECTRAL_ERR_TIMEOUT:         return "timeout";
        case SPECTRAL_ERR_FILE_OPEN:       return "file open failed";
        case SPECTRAL_ERR_FILE_READ:       return "file read failed";
        case SPECTRAL_ERR_FILE_WRITE:      return "file write failed";
        case SPECTRAL_ERR_FILE_FORMAT:     return "invalid file format";
        case SPECTRAL_ERR_FILE_VERSION:    return "unsupported version";
        case SPECTRAL_ERR_FILE_CORRUPT:    return "file corrupt";
        case SPECTRAL_ERR_BACKEND_UNAVAIL: return "backend unavailable";
        case SPECTRAL_ERR_TIMBRE_UNSUP:    return "timbre unsupported";
        case SPECTRAL_ERR_WAVETABLE_UNSUP: return "wavetable unsupported";
        case SPECTRAL_ERR_FFT_INIT:        return "FFT init failed";
        case SPECTRAL_ERR_GPU_INIT:        return "GPU init failed";
        case SPECTRAL_ERR_SD_MOUNT:        return "SD mount failed";
        case SPECTRAL_ERR_SD_OPEN:         return "SD file open failed";
        case SPECTRAL_ERR_SD_READ:         return "SD read failed";
        case SPECTRAL_ERR_SD_WRITE:        return "SD write failed";
        case SPECTRAL_ERR_PROTO_CHECKSUM:  return "checksum error";
        case SPECTRAL_ERR_PROTO_CMD:       return "invalid command";
        case SPECTRAL_ERR_PROTO_OVERFLOW:  return "protocol overflow";
        case SPECTRAL_ERR_PROTO_TIMEOUT:   return "protocol timeout";
        case SPECTRAL_ERR_SIM_UNAVAIL:     return "simulation unavailable";
        case SPECTRAL_ERR_SIM_SEG_FAIL:    return "simulation segment failed";
        case SPECTRAL_ERR_SIM_ACCUM_FAIL:  return "simulation accumulator failed";
        default:                           return "unknown error";
    }
}

static const char* spectral_wavetable_error_message(int code)
{
    switch ((WavetableError)code) {
        case WAVETABLE_OK:            return "OK";
        case WAVETABLE_ERR_PARAM:     return "invalid parameter";
        case WAVETABLE_ERR_FILE:      return "file open failed";
        case WAVETABLE_ERR_FORMAT:    return "invalid format";
        case WAVETABLE_ERR_SIZE:      return "size mismatch";
        case WAVETABLE_ERR_FULL:      return "wavetable bank full";
        case WAVETABLE_ERR_NOT_FOUND: return "wavetable not found";
        case WAVETABLE_ERR_VERSION:   return "unsupported version";
        case WAVETABLE_ERR_MEMORY:    return "memory allocation failed";
        case WAVETABLE_ERR_UNSUPPORTED: return "unsupported on this build target";
        default:                      return "unknown wavetable error";
    }
}

static const char* spectral_pipeline_error_message(int code)
{
    switch ((PipelineError)code) {
        case PIPELINE_OK:            return "OK";
        case PIPELINE_ERR_INPUT:     return "input error";
        case PIPELINE_ERR_ANALYSIS:  return "analysis failed";
        case PIPELINE_ERR_SYNTHESIS: return "synthesis failed";
        case PIPELINE_ERR_OUTPUT:    return "output error";
        case PIPELINE_ERR_WAVETABLE: return "wavetable error";
        case PIPELINE_ERR_MEMORY:    return "memory allocation failed";
        default:                     return "unknown pipeline error";
    }
}

const char* spectral_error_message(SpectralErrorDomain domain, int code)
{
    switch (domain) {
        case SPECTRAL_ERROR_DOMAIN_CORE:
            return spectral_core_error_message(code);
        case SPECTRAL_ERROR_DOMAIN_WAVETABLE:
            return spectral_wavetable_error_message(code);
        case SPECTRAL_ERROR_DOMAIN_PIPELINE:
            return spectral_pipeline_error_message(code);
        default:
            return "unknown error domain";
    }
}

static void spectral_log_codef(SpectralLogLevel level,
                               SpectralErrorDomain domain,
                               int code,
                               const char* context_fmt,
                               va_list args)
{
    const char* message = spectral_error_message(domain, code);
    char context[512];

    if (context_fmt && context_fmt[0]) {
        (void)vsnprintf(context, sizeof(context), context_fmt, args);
        spectral_log(level, stdout, 0, 1, "%s (%s)", context, message);
        return;
    }

    spectral_log(level, stdout, 0, 1, "%s", message);
}

void spectral_log_error_codef(SpectralErrorDomain domain, int code, const char* context_fmt, ...)
{
    va_list args;
    va_start(args, context_fmt);
    spectral_log_codef(SPECTRAL_LOG_LEVEL_ERROR, domain, code, context_fmt, args);
    va_end(args);
}

void spectral_log_warn_codef(SpectralErrorDomain domain, int code, const char* context_fmt, ...)
{
    va_list args;
    va_start(args, context_fmt);
    spectral_log_codef(SPECTRAL_LOG_LEVEL_WARN, domain, code, context_fmt, args);
    va_end(args);
}
