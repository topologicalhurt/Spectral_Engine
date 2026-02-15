/* spectral_synth_internal.h - Shared synthesis helpers (CPU, Metal, CUDA, embedded) */
#ifndef SPECTRAL_SYNTH_INTERNAL_H
#define SPECTRAL_SYNTH_INTERNAL_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_error.h"
#include "oscillator.h"
#include <stdlib.h>
#include <string.h>

/* Precomputed segment parameters for inner synthesis loop */
typedef struct SegmentLoopParams {
    size_t start_idx;
    size_t length;
    float alpha;    /* phase increment per sample */
    float beta;     /* chirp rate (phase acceleration) */
    float d_amp;    /* amplitude delta per sample */
    float phase;
    float amp;
    float width;
    int valid;
} SegmentLoopParams;

SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs);
SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len);

/* Compute instantaneous phase at sample j (quadratic phase model) */
static inline float compute_phase(float phase0, float alpha, float beta, size_t j) {
    float jf = (float)j;
    return phase0 + jf * (alpha + beta * jf);
}

/* Validate synth inputs; call at start of every synth function */

typedef enum {
    SYNTH_VALIDATE_OK = 1,
    SYNTH_VALIDATE_EARLY_EXIT = 0
} SynthValidateResult;

SynthValidateResult synth_validate_inputs(void* out_buffer, size_t out_len, size_t elem_size,
                                          SegmentArray sa, double** t_synth_ptr);

#define SYNTH_VALIDATE_FLOAT(buf, len, sa, t_ptr) \
    synth_validate_inputs((buf), (len), sizeof(float), (sa), (t_ptr))

#define SYNTH_VALIDATE_NATIVE(buf, len, sa, t_ptr) \
    synth_validate_inputs((buf), (len), sizeof(spectral_sample_t), (sa), (t_ptr))

/* Shared preflight: validate + params + timing in one call.
 * ok==0 means early exit was already handled (zero-filled or dummy timing). */
typedef struct {
    SynthParams params;
    double      start_time;   /* omp_get_wtime() captured after validation */
    int         ok;           /* 0 = early exit, 1 = proceed */
} SynthPreflight;

SynthPreflight synth_preflight_float(
    float* out_buffer, size_t out_len, SegmentArray sa,
    float stretch, float pitch, double** t_synth);

/* GPU timbre support: timbres 0-5 only (6-7 need width param, CPU only) */

int gpu_check_timbre_or_fallback(const char* backend_name,
                                  SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch, SpectralTimbre timbre, 
                                  double* t_synth);

static inline int gpu_timbre_supported(SpectralTimbre timbre) {
    return (int)timbre <= OSC_GPU_MAX_TIMBRE;
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

/* GPU synthesis params — layout must match Metal shader SynthParams struct */
typedef struct {
    float    stretch, inv_stretch, inv_stretch_sq, pitch_factor;
    uint32_t out_len, num_segments, tile_size, timbre;
} GpuSynthParams;

static inline GpuSynthParams gpu_synth_params_pack(
    const SynthParams* sp, uint32_t tile_size, SpectralTimbre timbre) {
    return (GpuSynthParams){
        sp->stretch, sp->inv_stretch, sp->inv_stretch_sq, sp->pitch_factor,
        (uint32_t)sp->out_len, sp->num_segments, tile_size, (uint32_t)timbre
    };
}

#endif /* SPECTRAL_SYNTH_INTERNAL_H */
