/* spectral_synth_internal.c - Shared Synthesis Helpers */

#include "spectral_synth_internal.h"
#include "spectral_synth.h"
#include "spectral_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "spectral_omp.h"

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
    if (!t_synth_ptr) {
        return SYNTH_VALIDATE_EARLY_EXIT;
    }

    if (!*t_synth_ptr) {
        *t_synth_ptr = &g_synth_timing_dummy;
    }

    if (elem_size == 0 || out_len > SIZE_MAX / elem_size) {
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
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

SynthPreflight synth_preflight_float(
    float* out_buffer, size_t out_len, SegmentArray sa,
    float stretch, float pitch, double** t_synth)
{
    SynthPreflight pf = {0};
    if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, t_synth)) return pf;
    pf.params = make_synth_params(stretch, pitch, out_len, sa.count);
    pf.start_time = omp_get_wtime();
    pf.ok = 1;
    return pf;
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
    SegmentLoopParams lp = {0};
    if (!s || !p || out_len == 0) return lp;

    /* Overflow-safe: clamp negative/huge float products before casting to size_t */
    double start_d = (double)s->start * (double)p->stretch;
    double length_d = (double)s->length * (double)p->stretch;
    if (!isfinite(start_d) || start_d < 0.0 || start_d >= (double)out_len || start_d > (double)SIZE_MAX) {
        lp.valid = 0;
        return lp;
    }
    if (!isfinite(length_d) || length_d < 0.0 || length_d > (double)SIZE_MAX) {
        lp.valid = 0;
        return lp;
    }

    lp.start_idx = (size_t)start_d;
    lp.length = (size_t)length_d;

    if (lp.start_idx >= out_len) {
        lp.valid = 0;
        return lp;
    }
    /* Overflow-safe comparison: rearranged to avoid size_t addition overflow */
    if (lp.length > out_len - lp.start_idx) {
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
    GpuTileData* out
) {
    if (!out || tile_size == 0 || out_len == 0) return SPECTRAL_ERR_PARAM;
    if (sa.count > 0 && !sa.segs) return SPECTRAL_ERR_PARAM;

    *out = (GpuTileData){0};
    if (out_len > UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;
    uint32_t num_tiles = ((uint32_t)out_len + tile_size - 1) / tile_size;

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif
    if (n_threads < 1) n_threads = 1;

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
            if (!isfinite(start) || !isfinite(end) || end <= start) continue;

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
    int tile_overflow = 0;
    for (int t = 0; t < n_threads; t++) {
        for (uint32_t i = 0; i < num_tiles; i++) {
            uint32_t sum = tile_counts[i] + thread_counts[t][i];
            if (sum < tile_counts[i]) { tile_overflow = 1; break; }
            tile_counts[i] = sum;
        }
        free(thread_counts[t]);
        if (tile_overflow) {
            for (int j = t + 1; j < n_threads; j++) free(thread_counts[j]);
            break;
        }
    }
    free(thread_counts);
    if (tile_overflow) {
        free(tile_counts);
        return SPECTRAL_ERR_OVERFLOW;
    }

    TileRange* tile_ranges = malloc(num_tiles * sizeof(TileRange));
    if (!tile_ranges) {
        free(tile_counts);
        return SPECTRAL_ERR_MEMORY;
    }
    uint32_t total_refs = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_ranges[t].start = total_refs;
        tile_ranges[t].count = tile_counts[t];
        if (total_refs > UINT32_MAX - tile_counts[t]) {
            free(tile_counts);
            free(tile_ranges);
            return SPECTRAL_ERR_OVERFLOW;
        }
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
        if (!isfinite(start) || !isfinite(end) || end <= start) continue;

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

    out->ranges = tile_ranges;
    out->segment_ids = tile_segment_ids;
    out->num_tiles = num_tiles;
    out->total_refs = total_refs;
    return SPECTRAL_OK;
}

#endif /* !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE */
