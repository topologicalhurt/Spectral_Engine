/* spectral_synth_internal.h - Shared synthesis helpers (CPU, Metal, CUDA, embedded) */
#ifndef SPECTRAL_SYNTH_INTERNAL_H
#define SPECTRAL_SYNTH_INTERNAL_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_error.h"
#include "oscillator.h"
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

SpectralError gpu_tile_preprocess(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    TileRange** out_ranges, uint32_t** out_segment_ids,
    uint32_t* out_num_tiles, uint32_t* out_total_refs
);
void gpu_tile_preprocess_free(TileRange* ranges, uint32_t* segment_ids);

#endif /* SPECTRAL_SYNTH_INTERNAL_H */
