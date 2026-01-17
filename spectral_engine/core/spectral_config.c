/* spectral_config.c - Config validation */
#include "spectral_config.h"
#include <stdio.h>
#include <math.h>

SpectralError spectral_config_validate(const SpectralConfigParams* cfg,
                                       char* error_msg, size_t error_msg_size)
{
    if (!cfg) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "NULL config");
        return SPECTRAL_ERR_PARAM;
    }
    
    if (cfg->sample_rate < 8000 || cfg->sample_rate > 192000) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "sample_rate %d out of range", cfg->sample_rate);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (!isfinite(cfg->stretch) || cfg->stretch <= 0.0f || cfg->stretch > 1000.0f) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "stretch %.2f invalid", cfg->stretch);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (!isfinite(cfg->pitch) || cfg->pitch < -48.0f || cfg->pitch > 48.0f) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "pitch %.1f invalid", cfg->pitch);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (cfg->timbre < TIMBRE_MIN || cfg->timbre > TIMBRE_MAX) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "timbre %d invalid", cfg->timbre);
        return SPECTRAL_ERR_PARAM;
    }
    
    if (cfg->buffer_size == 0) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "buffer_size is zero");
        return SPECTRAL_ERR_PARAM;
    }
    
#if SPECTRAL_EMBEDDED
    if (cfg->buffer_size > 4096) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "buffer_size %zu too large", cfg->buffer_size);
        return SPECTRAL_ERR_OVERFLOW;
    }
#endif
    
#if !SPECTRAL_EMBEDDED
    if (cfg->n_threads < 1 || cfg->n_threads > 256) {
        if (error_msg && error_msg_size > 0)
            snprintf(error_msg, error_msg_size, "n_threads %d out of range", cfg->n_threads);
        return SPECTRAL_ERR_PARAM;
    }
#endif
    
    return SPECTRAL_OK;
}

int spectral_config_is_valid(const SpectralConfigParams* cfg)
{
    return spectral_config_validate(cfg, NULL, 0) == SPECTRAL_OK;
}
