/* spectral_gpu_tile.c (host profile) - GPU tile preprocessing.
 *
 * Build-selected for the host/desktop profile, where a GPU backend
 * (Metal/CUDA) actually exists. Maps segments to output tiles for GPU
 * dispatch. This is the real implementation extracted from
 * core/spectral_synth_internal.c (its former #if !SPECTRAL_EMBEDDED &&
 * !SPECTRAL_RESTRICTED_MODE body); the embedded/simulation profile, which has
 * no GPU backend, build-selects the SPECTRAL_ERR_BACKEND_UNAVAIL stub in
 * core/port/embedded/spectral_gpu_tile.c instead. The shared declaration lives
 * in spectral_synth_internal.h, so callers are profile-agnostic.
 */

#include "spectral_synth_internal.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include "spectral_omp.h"
#include <stdlib.h>
#include <math.h>

typedef struct SpectralGpuTileSpan {
    uint32_t start_tile;
    uint32_t end_tile;
    int valid;
} SpectralGpuTileSpan;

static int spectral_gpu_segment_tile_span(const Segment* seg,
                                          float stretch,
                                          uint32_t tile_size,
                                          uint32_t num_tiles,
                                          SpectralGpuTileSpan* out)
{
    double start = 0.0;
    double end = 0.0;
    double tile_size_d = 0.0;
    double output_span = 0.0;
    double start_tile_d = 0.0;
    double end_tile_d = 0.0;

    if (!out) return 0;
    *out = (SpectralGpuTileSpan){0};
    if (!seg || tile_size == 0u || num_tiles == 0u) return 0;
    if (!spectral_is_finite_positive_f32(stretch) || stretch > SPECTRAL_MAX_STRETCH) return 0;
    if (!spectral_is_finite_f32(seg->start) || !spectral_is_finite_f32(seg->length)) return 0;
    if (seg->length <= 0.0f) return 0;

    start = (double)seg->start * (double)stretch;
    end = start + (double)seg->length * (double)stretch;
    if (!spectral_is_finite_f64(start) || !spectral_is_finite_f64(end) || end <= start) return 0;

    tile_size_d = (double)tile_size;
    output_span = (double)num_tiles * tile_size_d;
    if (end <= 0.0 || start >= output_span) return 0;

    if (start < 0.0) start = 0.0;
    if (end > output_span) end = output_span;

    start_tile_d = floor(start / tile_size_d);
    end_tile_d = ceil(end / tile_size_d) - 1.0;
    if (!spectral_is_finite_f64(start_tile_d) || !spectral_is_finite_f64(end_tile_d)) return 0;
    if (start_tile_d < 0.0) start_tile_d = 0.0;
    if (end_tile_d < start_tile_d) return 0;
    if (start_tile_d >= (double)num_tiles) return 0;
    if (end_tile_d >= (double)num_tiles) end_tile_d = (double)(num_tiles - 1u);

    out->start_tile = (uint32_t)start_tile_d;
    out->end_tile = (uint32_t)end_tile_d;
    out->valid = 1;
    return 1;
}

static void gpu_tile_preprocess_scratch_free(uint32_t* tile_counts,
                                             uint32_t* tile_cursors,
                                             uint32_t** thread_counts,
                                             int n_threads)
{
    free(tile_counts);
    free(tile_cursors);
    if (thread_counts) {
        for (int t = 0; t < n_threads; t++) {
            free(thread_counts[t]);
        }
        free(thread_counts);
    }
}

SpectralError gpu_tile_preprocess(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    GpuTileData* out
) {
    size_t bytes = 0;
    size_t counts_bytes = 0;
    size_t ids_bytes = 0;
    size_t ranges_bytes = 0;
    size_t cursors_bytes = 0;

    if (!out || tile_size == 0 || out_len == 0) return SPECTRAL_ERR_PARAM;
    if (!spectral_is_finite_positive_f32(stretch) || stretch > SPECTRAL_MAX_STRETCH) {
        return SPECTRAL_ERR_PARAM;
    }
    if (sa.count > 0 && !sa.segs) return SPECTRAL_ERR_PARAM;
    if (sa.count > (size_t)UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;

    *out = (GpuTileData){0};
    if (out_len > UINT32_MAX) return SPECTRAL_ERR_OVERFLOW;
    uint32_t out_len_u32 = (uint32_t)out_len;
    uint32_t num_tiles = out_len_u32 / tile_size + ((out_len_u32 % tile_size) ? 1u : 0u);
    SpectralError return_err = SPECTRAL_OK;
    int tile_overflow = 0;
    int fill_overflow = 0;
    uint32_t total_refs = 0;

    uint32_t** thread_counts = NULL;
    uint32_t* tile_counts = NULL;
    TileRange* tile_ranges = NULL;
    uint32_t* tile_segment_ids = NULL;
    uint32_t* tile_cursors = NULL;

    int n_threads = spectral_omp_effective_thread_count();

    if (!spectral_size_mul((size_t)n_threads, sizeof(uint32_t*), &bytes)) {
        return_err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }
    thread_counts = spectral_calloc_array((size_t)n_threads, sizeof(uint32_t*));
    if (!thread_counts) {
        return_err = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }
    if (!spectral_size_mul((size_t)num_tiles, sizeof(uint32_t), &counts_bytes)) {
        return_err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }
    for (int t = 0; t < n_threads; t++) {
        thread_counts[t] = spectral_calloc_array((size_t)num_tiles, sizeof(uint32_t));
        if (!thread_counts[t]) {
            return_err = SPECTRAL_ERR_MEMORY;
            goto cleanup;
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
            SpectralGpuTileSpan span = {0};
            if (!spectral_gpu_segment_tile_span(&sa.segs[i], stretch, tile_size, num_tiles, &span)) {
                continue;
            }

            for (uint32_t tt = span.start_tile; tt <= span.end_tile; tt++) {
                my_counts[tt]++;
            }
        }
    }

    tile_counts = spectral_calloc_array((size_t)num_tiles, sizeof(uint32_t));
    if (!tile_counts) {
        return_err = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }
    for (int t = 0; t < n_threads; t++) {
        for (uint32_t i = 0; i < num_tiles; i++) {
            uint32_t sum = tile_counts[i] + thread_counts[t][i];
            if (sum < tile_counts[i]) { tile_overflow = 1; break; }
            tile_counts[i] = sum;
        }
        if (tile_overflow) break;
    }
    if (tile_overflow) {
        return_err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }

    if (!spectral_size_mul((size_t)num_tiles, sizeof(TileRange), &ranges_bytes)) {
        return_err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }
    tile_ranges = spectral_malloc_array((size_t)num_tiles, sizeof(TileRange));
    if (!tile_ranges) {
        return_err = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }
    total_refs = 0;
    for (uint32_t t = 0; t < num_tiles; t++) {
        tile_ranges[t].start = total_refs;
        tile_ranges[t].count = tile_counts[t];
        if (total_refs > UINT32_MAX - tile_counts[t]) {
            return_err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }
        total_refs += tile_counts[t];
    }

    if (!spectral_size_mul((size_t)num_tiles, sizeof(uint32_t), &cursors_bytes)) {
        return_err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }
    tile_cursors = spectral_calloc_array((size_t)num_tiles, sizeof(uint32_t));
    if (!tile_cursors) {
        return_err = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }

    if (total_refs > 0u) {
        if (!spectral_size_mul((size_t)total_refs, sizeof(uint32_t), &ids_bytes)) {
            return_err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }
        tile_segment_ids = spectral_malloc_array((size_t)total_refs, sizeof(uint32_t));
        if (!tile_segment_ids) {
            return_err = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            SpectralGpuTileSpan span = {0};
            if (!spectral_gpu_segment_tile_span(&sa.segs[i], stretch, tile_size, num_tiles, &span)) {
                continue;
            }

            for (uint32_t tt = span.start_tile; tt <= span.end_tile; tt++) {
                uint32_t pos;
                uint32_t write_index = 0;
                #pragma omp atomic capture
                pos = tile_cursors[tt]++;

                if (pos >= tile_ranges[tt].count ||
                    tile_ranges[tt].start > total_refs ||
                    pos > total_refs - tile_ranges[tt].start) {
                    #pragma omp atomic write
                    fill_overflow = 1;
                    continue;
                }

                write_index = tile_ranges[tt].start + pos;
                if (write_index >= total_refs) {
                    #pragma omp atomic write
                    fill_overflow = 1;
                    continue;
                }

                tile_segment_ids[write_index] = (uint32_t)i;
            }
        }
    }

    if (fill_overflow) {
        return_err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }
    for (uint32_t t = 0; t < num_tiles; t++) {
        if (tile_cursors[t] != tile_counts[t]) {
            return_err = SPECTRAL_ERR_FILE_CORRUPT;
            goto cleanup;
        }
    }

    gpu_tile_preprocess_scratch_free(tile_counts, tile_cursors, thread_counts, n_threads);

    out->ranges = tile_ranges;
    out->segment_ids = tile_segment_ids;
    out->num_tiles = num_tiles;
    out->total_refs = total_refs;
    return SPECTRAL_OK;

cleanup:
    gpu_tile_preprocess_scratch_free(tile_counts, tile_cursors, thread_counts, n_threads);
    free(tile_ranges);
    free(tile_segment_ids);
    return return_err;
}
