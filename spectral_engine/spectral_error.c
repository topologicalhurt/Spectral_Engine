/* spectral_error.c - Error code string conversion */
#include "spectral_error.h"

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
        default:                           return "unknown error";
    }
}
