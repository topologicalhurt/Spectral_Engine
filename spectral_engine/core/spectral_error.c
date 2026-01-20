/* spectral_error.c - Error code string conversion */
#include "spectral_error.h"
#include "spectral_wavetable.h"

const char* spectral_strerror(SpectralError err)
{
    switch (err) {
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
        case SPECTRAL_ERR_EMU_UNAVAIL:     return "emulator unavailable";
        case SPECTRAL_ERR_EMU_SEG_FAIL:    return "emulator segment failed";
        case SPECTRAL_ERR_EMU_ACCUM_FAIL:  return "emulator accumulator failed";
        default:                           return "unknown error";
    }
}

const char* wavetable_strerror(WavetableError err)
{
    switch (err) {
        case WAVETABLE_OK:            return "OK";
        case WAVETABLE_ERR_PARAM:     return "invalid parameter";
        case WAVETABLE_ERR_FILE:      return "file open failed";
        case WAVETABLE_ERR_FORMAT:    return "invalid format";
        case WAVETABLE_ERR_SIZE:      return "size mismatch";
        case WAVETABLE_ERR_FULL:      return "wavetable bank full";
        case WAVETABLE_ERR_NOT_FOUND: return "wavetable not found";
        case WAVETABLE_ERR_VERSION:   return "unsupported version";
        case WAVETABLE_ERR_MEMORY:    return "memory allocation failed";
        default:                      return "unknown wavetable error";
    }
}

const char* pipeline_strerror(PipelineError err)
{
    switch (err) {
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
