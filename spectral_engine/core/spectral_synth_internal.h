/* spectral_synth_internal.h - Shared synthesis helpers (CPU, Metal, CUDA, embedded) */
#ifndef SPECTRAL_SYNTH_INTERNAL_H
#define SPECTRAL_SYNTH_INTERNAL_H

#include "spectral_common.h"
#include "spectral_error.h"
#include "spectral_segment_math.h"
#include "spectral_utils.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Precomputed segment parameters for inner synthesis loop */
typedef struct SegmentLoopParams {
    size_t start_idx;
    size_t length;
    float alpha;    /* phase increment per sample */
    float beta;     /* chirp rate (phase acceleration) */
    float c2;       /* cubic phase coeff of offset^2 (== beta when no linkage) */
    float c3;       /* cubic phase coeff of offset^3 (0 when no linkage) */
    float d_amp;    /* amplitude delta per sample */
    float phase;
    float amp;
    float width;
    int valid;
} SegmentLoopParams;

SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs);
SpectralError synth_validate_params(float stretch, float pitch);
SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len);

/* Shared preflight: validate + params + timing in one call.
 * ok==0 means early exit was already handled (zero-filled or dummy timing). */
typedef struct {
    SynthParams   params;
    double        start_time;   /* omp_get_wtime() captured after validation */
    SpectralError error;        /* SPECTRAL_OK for valid early exits */
    int           ok;           /* 0 = early exit/error, 1 = proceed */
} SynthPreflight;

SynthPreflight synth_preflight_float(
    float* out_buffer, size_t out_len, SegmentArray sa,
    float stretch, float pitch, double** t_synth);

SynthPreflight synth_preflight_native(
    spectral_sample_t* out_buffer, size_t out_len, SegmentArray sa,
    float stretch, float pitch, double** t_synth);

/* Per-dispatch effective timbre tracking for logging/reporting. */
void synth_effective_timbre_reset(SpectralTimbre requested_timbre);
void synth_effective_timbre_set(SpectralTimbre effective_timbre);
SpectralTimbre synth_effective_timbre_get(void);

/* Canonical unsupported-timbre fallback hook used by synthesis backends. */
typedef SpectralError (*SpectralUnsupportedTimbreFallbackFn)(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    int n_threads, double* t_synth, void* user_data);

SpectralError spectral_handle_unsupported_timbre(
    const char* backend_name,
    int max_supported_timbre,
    SpectralTimbre requested_timbre,
    SpectralTimbre fallback_timbre,
    const char* backend_constraint_note,
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, int n_threads, double* t_synth,
    SpectralUnsupportedTimbreFallbackFn fallback_fn, void* user_data,
    int* out_continue_backend);

/* GPU timbre support: timbres 0-5 only (6-7 need width param, CPU only). */

SpectralError gpu_check_timbre_or_fallback(const char* backend_name,
                                           SegmentArray sa, float* out_buffer, size_t out_len,
                                           float stretch, float pitch, SpectralTimbre timbre,
                                           int n_threads, double* t_synth,
                                           int* out_continue_backend);

static inline int gpu_timbre_supported(SpectralTimbre timbre) {
    return (int)timbre <= TIMBRE_PARABOLA;
}

/* GPU tile preprocessing - shared between Metal and CUDA backends */
typedef struct { uint32_t start; uint32_t count; } TileRange;

typedef struct {
    TileRange* ranges;
    uint32_t*  segment_ids;
    uint32_t   num_tiles;
    uint32_t   total_refs;
} GpuTileData;

static inline void gpu_tile_data_free(GpuTileData* td) {
    free(td->ranges);
    free(td->segment_ids);
    *td = (GpuTileData){0};
}

SpectralError gpu_tile_preprocess(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    GpuTileData* out);

/* Fetches pre-computed tile layout from cache if available, otherwise allocates
 * and executes gpu_tile_preprocess. out_owns_data is set to 1 only if a new
 * allocation occurred (must call gpu_tile_data_free later). */
SpectralError gpu_tile_preprocess_cached(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    GpuTileData* out_td, int* out_owns_data);

/* Global GPU tile cache — pre-computed tiles from the segment cache
 * bypass tile preprocessing in GPU backends. */
void gpu_tile_cache_set(const void* ranges, const uint32_t* segment_ids,
                        uint32_t num_tiles, uint32_t total_refs,
                        float stretch, size_t out_len);
void gpu_tile_cache_clear(void);

/* Global GPU segment cache — pre-packed SegmentGpu data from the segment
 * cache (mmap'd).  Lets GPU backends skip the Segment→SegmentGpu pack loop. */
void gpu_seg_cache_set(const SegmentGpu* segs, uint32_t count);
void gpu_seg_cache_clear(void);

/* GPU synthesis params — layout must match Metal shader SynthParams struct */
typedef struct {
    float    stretch, inv_stretch, inv_stretch_sq, pitch_factor;
    uint32_t out_len, num_segments, tile_size, timbre;
} GpuSynthParams;

/* Checked GPU params pack: GPU kernels expose 32-bit lengths/counts even though
 * shared SynthParams stores out_len as size_t.  This helper is the only valid
 * boundary-crossing pack for Metal/CUDA dispatch. */
static inline SpectralError gpu_synth_params_pack_checked(
    const SynthParams* sp, uint32_t tile_size, SpectralTimbre timbre, GpuSynthParams* out)
{
    if (!sp || !out || tile_size == 0u) return SPECTRAL_ERR_PARAM;
    *out = (GpuSynthParams){0};

    if (sp->out_len > (size_t)UINT32_MAX ||
        sp->num_segments > UINT32_MAX ||
        (int)timbre < TIMBRE_SINE ||
        (int)timbre > TIMBRE_PWM) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    if (!spectral_is_finite_positive_f32(sp->stretch) ||
        !spectral_is_finite_positive_f32(sp->inv_stretch) ||
        !spectral_is_finite_positive_f32(sp->inv_stretch_sq) ||
        !spectral_is_finite_positive_f32(sp->pitch_factor)) {
        return SPECTRAL_ERR_PARAM;
    }

    *out = (GpuSynthParams){
        sp->stretch, sp->inv_stretch, sp->inv_stretch_sq, sp->pitch_factor,
        (uint32_t)sp->out_len, sp->num_segments, tile_size, (uint32_t)timbre
    };
    return SPECTRAL_OK;
}

typedef struct SpectralGpuDispatchPlan {
    const SegmentGpu* segment_source;   /* NULL means backend must pack from SegmentArray */
    size_t            segment_bytes;
    GpuTileData       tiles;
    int               owns_tile_data;
    int               zero_output;
    GpuSynthParams    params;
    size_t            tile_ids_bytes;
    size_t            tile_ranges_bytes;
} SpectralGpuDispatchPlan;

SpectralError spectral_gpu_dispatch_plan_init(SpectralGpuDispatchPlan* plan,
                                              SegmentArray sa,
                                              const SynthParams* params,
                                              float stretch,
                                              SpectralTimbre timbre,
                                              size_t out_len);
void spectral_gpu_dispatch_plan_free(SpectralGpuDispatchPlan* plan);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SYNTH_INTERNAL_H */
