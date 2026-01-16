/* spectral_error.h - Unified error codes */
#ifndef SPECTRAL_ERROR_H
#define SPECTRAL_ERROR_H

typedef enum {
    SPECTRAL_OK                  =    0,
    
    /* Core (-1 to -49) */
    SPECTRAL_ERR_PARAM           =   -1,
    SPECTRAL_ERR_MEMORY          =   -2,
    SPECTRAL_ERR_OVERFLOW        =   -3,
    SPECTRAL_ERR_NOTINIT         =   -4,
    SPECTRAL_ERR_BUSY            =   -5,
    SPECTRAL_ERR_TIMEOUT         =   -6,
    
    /* File (-50 to -99) */
    SPECTRAL_ERR_FILE_OPEN       =  -50,
    SPECTRAL_ERR_FILE_READ       =  -51,
    SPECTRAL_ERR_FILE_WRITE      =  -52,
    SPECTRAL_ERR_FILE_FORMAT     =  -53,
    SPECTRAL_ERR_FILE_VERSION    =  -54,
    SPECTRAL_ERR_FILE_CORRUPT    =  -55,
    
    /* Backend (-100 to -149) */
    SPECTRAL_ERR_BACKEND_UNAVAIL = -100,
    SPECTRAL_ERR_TIMBRE_UNSUP    = -101,
    SPECTRAL_ERR_WAVETABLE_UNSUP = -102,
    SPECTRAL_ERR_FFT_INIT        = -103,
    SPECTRAL_ERR_GPU_INIT        = -104,
    
    /* SD card (-150 to -199) */
    SPECTRAL_ERR_SD_MOUNT        = -150,
    SPECTRAL_ERR_SD_OPEN         = -151,
    SPECTRAL_ERR_SD_READ         = -152,
    SPECTRAL_ERR_SD_WRITE        = -153,
    
    /* Protocol (-200 to -249) */
    SPECTRAL_ERR_PROTO_CHECKSUM  = -200,
    SPECTRAL_ERR_PROTO_CMD       = -201,
    SPECTRAL_ERR_PROTO_OVERFLOW  = -202,
    SPECTRAL_ERR_PROTO_TIMEOUT   = -203
} SpectralError;

const char* spectral_strerror(SpectralError err);

static inline int spectral_is_ok(SpectralError err) { return err == SPECTRAL_OK; }
static inline int spectral_is_error(SpectralError err) { return err != SPECTRAL_OK; }

#endif
