/* spectral_synth_internal.c - Shared Synthesis Helpers */

#include "spectral_synth_internal.h"
#include "spectral_synth.h"
#include "spectral_utils.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs) {
    return (SynthParams){
        .stretch = stretch,
        .inv_stretch = 1.0f / stretch,
        .inv_stretch_sq = 1.0f / (stretch * stretch),
        .pitch_factor = SPECTRAL_PITCH_FACTOR(pitch),
        .out_len = out_len,
        .num_segments = (uint32_t)num_segs
    };
}

static double g_synth_timing_dummy = 0;

/* Validate synth inputs. All backends call this first. */
SynthValidateResult synth_validate_inputs(void* out_buffer, size_t out_len, size_t elem_size,
                                          SegmentArray sa, double** t_synth_ptr) {
    if (!*t_synth_ptr) {
        *t_synth_ptr = &g_synth_timing_dummy;
    }
    
    if (!out_buffer || out_len == 0) {
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    if (sa.count == 0 || !sa.segs) {
        memset(out_buffer, 0, out_len * elem_size);
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    return SYNTH_VALIDATE_OK;
}

int gpu_check_timbre_or_fallback(const char* backend_name,
                                  SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch, SpectralTimbre timbre, 
                                  double* t_synth) {
    (void)backend_name;  /* Used only in debug builds */
    
    if (!gpu_timbre_supported(timbre)) {
        SPECTRAL_DBG("%s: Timbre %d not supported on GPU (max %d), using CPU synthesis",
                     backend_name, (int)timbre, OSC_GPU_MAX_TIMBRE);
        synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, 1, t_synth);
        return 0;  /* Used CPU fallback */
    }
    return 1;  /* GPU can handle this timbre */
}

SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len) {
    SegmentLoopParams lp;
    
    lp.start_idx = (size_t)(s->start * p->stretch);
    lp.length = (size_t)(s->length * p->stretch);
    
    if (lp.start_idx >= out_len) {
        lp.valid = 0;
        return lp;
    }
    if (lp.start_idx + lp.length > out_len) {
        lp.length = out_len - lp.start_idx;
    }
    
    lp.alpha = s->omega * p->pitch_factor * p->inv_stretch;
    lp.beta = s->df * p->pitch_factor * p->inv_stretch_sq;
    lp.d_amp = s->da * p->inv_stretch;
    lp.phase = s->phase;
    lp.amp = s->amp;
    lp.width = s->width;
    lp.valid = 1;

    return lp;
}

/* GPU tile preprocessing - maps segments to tiles for Metal/CUDA dispatch */
#if !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE

SpectralError gpu_tile_preprocess(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    TileRange** out_ranges, uint32_t** out_segment_ids,
    uint32_t* out_num_tiles, uint32_t* out_total_refs
) {
    uint32_t num_tiles = ((uint32_t)out_len + tile_size - 1) / tile_size;

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif

    uint32_t** thread_counts = malloc(n_threads * sizeof(uint32_t*));
    if (!thread_counts) return SPECTRAL_ERR_MEMORY;
    for (int t = 0; t < n_threads; t++) {
        thread_counts[t] = calloc(num_tiles, sizeof(uint32_t));
        if (!thread_counts[t]) {
            for (int i = 0; i < t; i++) free(thread_counts[i]);
            free(thread_counts);
            return SPECTRAL_ERR_MEMORY;
        }
    }

    #pragma omp parallel
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        uint32_t* my_counts = thread_counts[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            float start = sa.segs[i].start * stretch;
            float end = start + sa.segs[i].length * stretch;

            int start_tile = (int)(start / tile_size);
            int end_tile = (int)(end / tile_size);
            if (start_tile < 0) start_tile = 0;
            if (start_tile >= (int)num_tiles) continue;
            if (end_tile >= (int)num_tiles) end_tile = num_tiles - 1;

            for (int tt = start_tile; tt <= end_tile; tt++) {
                my_counts[tt]++;
            }
        }
    }

    uint32_t* tile_counts = calloc(num_tiles, sizeof(uint32_t));
    if (!tile_counts) {
        for (int t = 0; t < n_threads; t++) free(thread_counts[t]);
        free(thread_counts);
        return SPECTRAL_ERR_MEMORY;
    }
    for (int t = 0; t < n_threads; t++) {
        for (uint32_t i = 0; i < num_tiles; i++) {
            tile_counts[i] += thread_counts[t][i];
        }
        free(thread_counts[t]);
    }
    free(thread_counts);

    TileRange* tile_ranges = malloc(num_tiles * sizeof(TileRange));
    if (!tile_ranges) {
        free(tile_counts);
        return SPECTRAL_ERR_MEMORY;
    }
    uint32_t total_refs = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_ranges[t].start = total_refs;
        tile_ranges[t].count = tile_counts[t];
        total_refs += tile_counts[t];
    }

    uint32_t* tile_segment_ids = malloc(total_refs * sizeof(uint32_t));
    uint32_t* tile_cursors = calloc(num_tiles, sizeof(uint32_t));
    if (!tile_segment_ids || !tile_cursors) {
        free(tile_counts);
        free(tile_ranges);
        free(tile_segment_ids);
        free(tile_cursors);
        return SPECTRAL_ERR_MEMORY;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sa.count; i++) {
        float start = sa.segs[i].start * stretch;
        float end = start + sa.segs[i].length * stretch;

        int start_tile = (int)(start / tile_size);
        int end_tile = (int)(end / tile_size);
        if (start_tile < 0) start_tile = 0;
        if (start_tile >= (int)num_tiles) continue;
        if (end_tile >= (int)num_tiles) end_tile = num_tiles - 1;

        for (int tt = start_tile; tt <= end_tile; tt++) {
            uint32_t pos;
            #pragma omp atomic capture
            pos = tile_cursors[tt]++;
            tile_segment_ids[tile_ranges[tt].start + pos] = (uint32_t)i;
        }
    }

    free(tile_counts);
    free(tile_cursors);

    *out_ranges = tile_ranges;
    *out_segment_ids = tile_segment_ids;
    *out_num_tiles = num_tiles;
    *out_total_refs = total_refs;
    return SPECTRAL_OK;
}

void gpu_tile_preprocess_free(TileRange* ranges, uint32_t* segment_ids) {
    free(ranges);
    free(segment_ids);
}

#endif /* !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE */
