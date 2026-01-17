/* spectral_api.h - Platform-agnostic API for spectral engine */

#ifndef SPECTRAL_API_H
#define SPECTRAL_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPECTRAL_API_VERSION_MAJOR  0
#define SPECTRAL_API_VERSION_MINOR  1
#define SPECTRAL_API_VERSION_PATCH  0

/* Platform detection */
#if defined(SPECTRAL_PLATFORM_DAISY)
    #define SPECTRAL_EMBEDDED       1
    #define SPECTRAL_MAX_SEGMENTS   4096
    #define SPECTRAL_SAMPLE_TYPE    int16_t
    #define SPECTRAL_USE_FIXED      0
#elif defined(SPECTRAL_PLATFORM_DESKTOP)
    #define SPECTRAL_EMBEDDED       0
    #define SPECTRAL_MAX_SEGMENTS   0
    #define SPECTRAL_SAMPLE_TYPE    float
    #define SPECTRAL_USE_FIXED      0
#else
    #define SPECTRAL_EMBEDDED       0
    #define SPECTRAL_MAX_SEGMENTS   0
    #define SPECTRAL_SAMPLE_TYPE    float
    #define SPECTRAL_USE_FIXED      0
#endif

/* Core types */

typedef struct SpectralSegmentCompact {
    float start, length, phase, omega, df, amp, da, width;
} SpectralSegmentCompact;

typedef struct SpectralSegment {
    float start, length, phase, omega, df, amp, da, width;
    float _pad[8];
} SpectralSegment;

typedef struct SpectralParams {
    float stretch;
    float pitch;
    float amplitude;
    uint8_t timbre;
    uint8_t quality;
    uint8_t mono;
    uint8_t _reserved;
} SpectralParams;

typedef struct SpectralStreamCtx {
    void* segments;
    uint32_t num_segments;
    uint32_t current_segment;
    uint32_t sample_position;
    uint32_t sample_rate;
    SpectralParams params;
    void* user_data;
} SpectralStreamCtx;

typedef enum SpectralResult {
    SPECTRAL_OK             =  0,
    SPECTRAL_ERR_MEMORY     = -1,
    SPECTRAL_ERR_PARAM      = -2,
    SPECTRAL_ERR_FORMAT     = -3,
    SPECTRAL_ERR_IO         = -4,
    SPECTRAL_ERR_OVERFLOW   = -5,
    SPECTRAL_ERR_NOTINIT    = -6
} SpectralResult;

typedef enum SpectralBackend {
    SPECTRAL_BACKEND_AUTO   = 0,
    SPECTRAL_BACKEND_CPU    = 1,
    SPECTRAL_BACKEND_SIMD   = 2,
    SPECTRAL_BACKEND_GPU    = 3
} SpectralBackend;

/* Initialization */
SpectralResult spectral_init(uint32_t sample_rate, SpectralBackend backend);
void spectral_deinit(void);
const char* spectral_version(void);

/* Segment loading */
SpectralResult spectral_load_file(SpectralStreamCtx* ctx, const char* path);
SpectralResult spectral_load_buffer(SpectralStreamCtx* ctx, 
                                    const void* data, size_t size);
SpectralResult spectral_load_streaming(SpectralStreamCtx* ctx, 
                                       const char* path, 
                                       uint32_t max_cached);
void spectral_unload(SpectralStreamCtx* ctx);

/* Synthesis */
void spectral_set_params(SpectralStreamCtx* ctx, const SpectralParams* params);
uint32_t spectral_process(SpectralStreamCtx* ctx, 
                          SPECTRAL_SAMPLE_TYPE* output,
                          uint32_t num_samples);
uint32_t spectral_process_float(SpectralStreamCtx* ctx,
                                float* output,
                                uint32_t num_samples);
void spectral_reset(SpectralStreamCtx* ctx);
void spectral_seek(SpectralStreamCtx* ctx, uint32_t sample_pos);

/* Utility */
uint32_t spectral_get_position(const SpectralStreamCtx* ctx);
uint32_t spectral_get_duration(const SpectralStreamCtx* ctx);
int spectral_is_complete(const SpectralStreamCtx* ctx);

typedef struct SpectralMemStats {
    uint32_t segments_loaded;
    uint32_t segments_cached;
    uint32_t bytes_used;
    uint32_t bytes_available;
} SpectralMemStats;

SpectralMemStats spectral_get_mem_stats(const SpectralStreamCtx* ctx);

/* Embedded-specific */
#if SPECTRAL_EMBEDDED

void spectral_segment_compact(const SpectralSegment* full, 
                              SpectralSegmentCompact* compact);
void spectral_segment_expand(const SpectralSegmentCompact* compact,
                             SpectralSegment* full);

#if SPECTRAL_USE_FIXED
typedef int32_t spectral_fixed_t;
#define SPECTRAL_FIXED_SHIFT    16
#define SPECTRAL_FLOAT_TO_FIXED(x)  ((spectral_fixed_t)((x) * (1 << SPECTRAL_FIXED_SHIFT)))
#define SPECTRAL_FIXED_TO_FLOAT(x)  ((float)(x) / (1 << SPECTRAL_FIXED_SHIFT))
#endif

#endif /* SPECTRAL_EMBEDDED */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_API_H */
