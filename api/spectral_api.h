/* spectral_api.h - Draft platform-agnostic API surface (v0.1.0)
 *
 * STATUS: DRAFT — NOT YET IMPLEMENTED
 *
 * This header defines a forward-looking, platform-agnostic API contract for
 * the Spectral Engine.  None of the functions declared here have backing
 * implementations in this repository yet.
 *
 * For production code paths use:
 *   - Desktop:    spectral_engine/synth/api/spectral_synth.h
 *   - Daisy Seed: api/daisy_seed/daisy_seed_spectral.h
 *
 * Including this header outside of API design work will produce link errors.
 * Guard your usage with SPECTRAL_API_DRAFT to acknowledge the draft status:
 *
 *   #define SPECTRAL_API_DRAFT 1
 *   #include "spectral_api.h"
 */

#ifndef SPECTRAL_API_H
#define SPECTRAL_API_H

#ifndef SPECTRAL_API_DRAFT
#error "spectral_api.h is a draft header with no implementation. " \
       "Define SPECTRAL_API_DRAFT=1 before including to acknowledge this."
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPECTRAL_API_VERSION_MAJOR  0
#define SPECTRAL_API_VERSION_MINOR  1
#define SPECTRAL_API_VERSION_PATCH  0

typedef enum {
    SPECTRAL_API_OK          =  0,
    SPECTRAL_API_ERR_MEMORY  = -1,
    SPECTRAL_API_ERR_PARAM   = -2,
    SPECTRAL_API_ERR_FORMAT  = -3,
    SPECTRAL_API_ERR_IO      = -4,
} SpectralApiResult;

typedef enum SpectralApiBackend {
    SPECTRAL_BACKEND_AUTO   = 0,
    SPECTRAL_BACKEND_CPU    = 1,
    SPECTRAL_BACKEND_SIMD   = 2,
    SPECTRAL_BACKEND_GPU    = 3
} SpectralApiBackend;

typedef struct SpectralApiParams {
    float stretch;
    float pitch;
    float amplitude;
    uint8_t timbre;
    uint8_t quality;
    uint8_t mono;
    uint8_t _reserved;
} SpectralApiParams;

typedef struct SpectralApiStreamCtx {
    void* segments;
    uint32_t num_segments;
    uint32_t current_segment;
    uint32_t sample_position;
    uint32_t sample_rate;
    SpectralApiParams params;
    void* user_data;
} SpectralApiStreamCtx;

typedef struct SpectralApiMemStats {
    uint32_t segments_loaded;
    uint32_t segments_cached;
    uint32_t bytes_used;
    uint32_t bytes_available;
} SpectralApiMemStats;

/* Initialization */
SpectralApiResult spectral_api_init(uint32_t sample_rate, SpectralApiBackend backend);
void spectral_api_deinit(void);
const char* spectral_api_version(void);

/* Segment loading */
SpectralApiResult spectral_api_load_file(SpectralApiStreamCtx* ctx, const char* path);
SpectralApiResult spectral_api_load_buffer(SpectralApiStreamCtx* ctx,
                                           const void* data, size_t size);
void spectral_api_unload(SpectralApiStreamCtx* ctx);

/* Synthesis */
void spectral_api_set_params(SpectralApiStreamCtx* ctx, const SpectralApiParams* params);
uint32_t spectral_api_process_float(SpectralApiStreamCtx* ctx,
                                    float* output, uint32_t num_samples);
void spectral_api_reset(SpectralApiStreamCtx* ctx);
void spectral_api_seek(SpectralApiStreamCtx* ctx, uint32_t sample_pos);

/* Status */
uint32_t spectral_api_get_position(const SpectralApiStreamCtx* ctx);
uint32_t spectral_api_get_duration(const SpectralApiStreamCtx* ctx);
int spectral_api_is_complete(const SpectralApiStreamCtx* ctx);
SpectralApiMemStats spectral_api_get_mem_stats(const SpectralApiStreamCtx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_API_H */
