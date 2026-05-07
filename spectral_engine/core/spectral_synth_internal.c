/* spectral_synth_internal.c - Shared Synthesis Helpers */

#include "spectral_synth_internal.h"
#include "spectral_synth.h"
#include "spectral_utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "spectral_omp.h"

static SpectralError synth_derive_param_scalars(float stretch, float pitch,
                                               float* out_inv_stretch,
                                               float* out_inv_stretch_sq,
                                               float* out_pitch_factor)
{
    float stretch_sq = 0.0f;
    float inv_stretch = 0.0f;
    float inv_stretch_sq = 0.0f;
    float pitch_factor = 0.0f;

    if (!spectral_is_finite_positive_f32(stretch) || stretch > SPECTRAL_MAX_STRETCH) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!spectral_is_finite_f32(pitch) ||
        pitch < SPECTRAL_MIN_PITCH || pitch > SPECTRAL_MAX_PITCH) {
        return SPECTRAL_ERR_PARAM;
    }

    /* The public domain is not just "stretch is positive".  Backends consume
     * these derived scalars directly; tiny positive stretch values can make
     * stretch * stretch underflow to zero and expose Inf in inv_stretch_sq. */
    stretch_sq = stretch * stretch;
    if (!spectral_is_finite_f32(stretch_sq) || stretch_sq <= 0.0f) {
        return SPECTRAL_ERR_PARAM;
    }

    inv_stretch = 1.0f / stretch;
    inv_stretch_sq = 1.0f / stretch_sq;
    pitch_factor = SPECTRAL_PITCH_FACTOR(pitch);

    if (!spectral_is_finite_positive_f32(inv_stretch) ||
        !spectral_is_finite_positive_f32(inv_stretch_sq) ||
        !spectral_is_finite_positive_f32(pitch_factor)) {
        return SPECTRAL_ERR_PARAM;
    }

    if (out_inv_stretch) *out_inv_stretch = inv_stretch;
    if (out_inv_stretch_sq) *out_inv_stretch_sq = inv_stretch_sq;
    if (out_pitch_factor) *out_pitch_factor = pitch_factor;
    return SPECTRAL_OK;
}

SpectralError synth_validate_params(float stretch, float pitch) {
    return synth_derive_param_scalars(stretch, pitch, NULL, NULL, NULL);
}


SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs) {
    float inv_stretch = 0.0f;
    float inv_stretch_sq = 0.0f;
    float pitch_factor = 0.0f;

    if (num_segs > (size_t)UINT32_MAX ||
        synth_derive_param_scalars(stretch, pitch,
                                   &inv_stretch,
                                   &inv_stretch_sq,
                                   &pitch_factor) != SPECTRAL_OK) {
        return (SynthParams){0};
    }

    return (SynthParams){
        .stretch = stretch,
        .inv_stretch = inv_stretch,
        .inv_stretch_sq = inv_stretch_sq,
        .pitch_factor = pitch_factor,
        .out_len = out_len,
        .num_segments = (uint32_t)num_segs
    };
}


static void synth_zero_output_if_valid(void* out_buffer, size_t out_len, size_t elem_size) {
    size_t out_bytes = 0;
    if (out_buffer && elem_size != 0 && spectral_size_mul(out_len, elem_size, &out_bytes)) {
        memset(out_buffer, 0, out_bytes);
    }
}

static double g_synth_timing_dummy = 0;
static SPECTRAL_THREAD_LOCAL int g_effective_timbre_tls = TIMBRE_SINE;

void synth_effective_timbre_reset(SpectralTimbre requested_timbre) {
    g_effective_timbre_tls = (int)requested_timbre;
}

void synth_effective_timbre_set(SpectralTimbre effective_timbre) {
    g_effective_timbre_tls = (int)effective_timbre;
}

SpectralTimbre synth_effective_timbre_get(void) {
    int t = g_effective_timbre_tls;
    if (t < TIMBRE_SINE || t > TIMBRE_PWM) return TIMBRE_SINE;
    return (SpectralTimbre)t;
}

/* Validate synth inputs. All backends call this first. */
SynthValidateResult synth_validate_inputs(void* out_buffer, size_t out_len, size_t elem_size,
                                          SegmentArray sa, double** t_synth_ptr) {
    size_t out_bytes = 0;
    if (!t_synth_ptr) {
        return SYNTH_VALIDATE_EARLY_EXIT;
    }

    if (!*t_synth_ptr) {
        *t_synth_ptr = &g_synth_timing_dummy;
    }

    if (elem_size == 0 || !spectral_size_mul(out_len, elem_size, &out_bytes)) {
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    if (!out_buffer || out_len == 0) {
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    if (sa.count == 0 || !sa.segs) {
        memset(out_buffer, 0, out_bytes);
        **t_synth_ptr = 0;
        return SYNTH_VALIDATE_EARLY_EXIT;
    }
    
    return SYNTH_VALIDATE_OK;
}

static SynthPreflight synth_preflight_common(
    void* out_buffer, size_t out_len, size_t elem_size, SegmentArray sa,
    float stretch, float pitch, double** t_synth)
{
    SynthPreflight pf = {0};
    size_t preflight_out_bytes = 0;
    pf.error = SPECTRAL_OK;

    /* Error cases must not be collapsed into the benign early-exit path.
     * In particular, out_len * elem_size overflow is a contract failure that
     * every backend must see as SPECTRAL_ERR_OVERFLOW, not SPECTRAL_OK. */
    if (!t_synth) {
        pf.error = SPECTRAL_ERR_PARAM;
        return pf;
    }
    if (!*t_synth) {
        *t_synth = &g_synth_timing_dummy;
    }
    if (elem_size == 0) {
        **t_synth = 0;
        pf.error = SPECTRAL_ERR_PARAM;
        return pf;
    }
    if (!spectral_size_mul(out_len, elem_size, &preflight_out_bytes)) {
        **t_synth = 0;
        pf.error = SPECTRAL_ERR_OVERFLOW;
        return pf;
    }

    if (!synth_validate_inputs(out_buffer, out_len, elem_size, sa, t_synth)) {
        return pf;
    }

    pf.error = synth_validate_params(stretch, pitch);
    if (pf.error != SPECTRAL_OK) {
        synth_zero_output_if_valid(out_buffer, out_len, elem_size);
        if (t_synth && *t_synth) **t_synth = 0;
        return pf;
    }

    if (sa.count > UINT32_MAX) {
        pf.error = SPECTRAL_ERR_OVERFLOW;
        synth_zero_output_if_valid(out_buffer, out_len, elem_size);
        if (t_synth && *t_synth) **t_synth = 0;
        return pf;
    }

    pf.params = make_synth_params(stretch, pitch, out_len, sa.count);
    pf.start_time = omp_get_wtime();
    pf.ok = 1;
    return pf;
}

SynthPreflight synth_preflight_float(
    float* out_buffer, size_t out_len, SegmentArray sa,
    float stretch, float pitch, double** t_synth)
{
    return synth_preflight_common(out_buffer, out_len, sizeof(float),
                                  sa, stretch, pitch, t_synth);
}

SynthPreflight synth_preflight_native(
    spectral_sample_t* out_buffer, size_t out_len, SegmentArray sa,
    float stretch, float pitch, double** t_synth)
{
    return synth_preflight_common(out_buffer, out_len, sizeof(spectral_sample_t),
                                  sa, stretch, pitch, t_synth);
}

SpectralError spectral_handle_unsupported_timbre(
    const char* backend_name,
    int max_supported_timbre,
    SpectralTimbre requested_timbre,
    SpectralTimbre fallback_timbre,
    const char* backend_constraint_note,
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, int n_threads, double* t_synth,
    SpectralUnsupportedTimbreFallbackFn fallback_fn, void* user_data,
    int* out_continue_backend)
{
    char resolution[384] = {0};
    SpectralResolutionContext context = {0};

    if (out_continue_backend) {
        *out_continue_backend = 1;
    }

    if ((int)requested_timbre >= TIMBRE_SINE && (int)requested_timbre <= max_supported_timbre) {
        return SPECTRAL_OK;
    }

    context.event = SPECTRAL_RESOLUTION_EVENT_FALLBACK;
    context.backend = backend_name;
    context.scope = SPECTRAL_RESOLUTION_SCOPE_TIMBRE;
    context.requested_label = timbre_name(requested_timbre);
    context.requested_id = (int)requested_timbre;
    context.effective_label = timbre_name(fallback_timbre);
    context.effective_id = (int)fallback_timbre;
    context.max_supported_id = max_supported_timbre;
    context.mode = spectral_exec_mode_name();
    context.reason = backend_constraint_note;
    spectral_format_resolution_context(resolution, sizeof(resolution), &context);

    SPECTRAL_WARN("%s", resolution);
    synth_effective_timbre_set(fallback_timbre);

    if (fallback_fn) {
        SpectralError fallback_err =
            fallback_fn(sa, out_buffer, out_len, stretch, pitch, fallback_timbre,
                        n_threads, t_synth, user_data);
        if (out_continue_backend) {
            *out_continue_backend = 0;
        }
        return fallback_err;
    }
    if (out_continue_backend) {
        *out_continue_backend = 0;
    }
    return SPECTRAL_ERR_TIMBRE_UNSUP;
}

static SpectralError synth_cpu_fallback_invoke(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    int n_threads, double* t_synth, void* user_data)
{
    (void)user_data;
    return synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
}

SpectralError gpu_check_timbre_or_fallback(const char* backend_name,
                                           SegmentArray sa, float* out_buffer, size_t out_len,
                                           float stretch, float pitch, SpectralTimbre timbre,
                                           int n_threads, double* t_synth,
                                           int* out_continue_backend) {
    return spectral_handle_unsupported_timbre(
        backend_name,
        TIMBRE_PARABOLA,
        timbre,
        timbre,
        SPECTRAL_RESOLUTION_REASON_GPU_TIMBRE_LIMIT,
        sa, out_buffer, out_len,
        stretch, pitch, n_threads, t_synth,
        synth_cpu_fallback_invoke, NULL, out_continue_backend);
}

SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len) {
    SegmentLoopParams lp = {0};
    float alpha = 0.0f;
    float beta = 0.0f;
    float d_amp = 0.0f;
    double last_offset_d = 0.0;

    if (!s || !p || out_len == 0) return lp;

    if (!spectral_is_finite_f32(s->start) ||
        !spectral_is_finite_f32(s->length) ||
        !spectral_is_finite_f32(s->phase) ||
        !spectral_is_finite_f32(s->omega) ||
        !spectral_is_finite_f32(s->df) ||
        !spectral_is_finite_f32(s->amp) ||
        !spectral_is_finite_f32(s->da) ||
        !spectral_is_finite_f32(s->width)) {
        lp.valid = 0;
        return lp;
    }

    /* Overflow-safe: clamp negative/huge float products before casting to size_t */
    double start_d = (double)s->start * (double)p->stretch;
    double length_d = (double)s->length * (double)p->stretch;
    if (!spectral_is_finite_f64(start_d) || start_d < 0.0 || start_d >= (double)out_len || start_d > (double)SIZE_MAX) {
        lp.valid = 0;
        return lp;
    }
    if (!spectral_is_finite_f64(length_d) || length_d < 0.0 || length_d > (double)SIZE_MAX) {
        lp.valid = 0;
        return lp;
    }

    lp.start_idx = (size_t)start_d;
    lp.length = (size_t)length_d;

    if (lp.start_idx >= out_len || lp.length == 0) {
        lp.valid = 0;
        return lp;
    }
    /* Overflow-safe comparison: rearranged to avoid size_t addition overflow */
    if (lp.length > out_len - lp.start_idx) {
        lp.length = out_len - lp.start_idx;
    }
    if (lp.length == 0) {
        lp.valid = 0;
        return lp;
    }

    /* Hot loops cast sample offsets to float.  Prove the final offset is
     * representable before later `(float)j` conversions in CPU/native callbacks
     * and before endpoint validation below. */
    last_offset_d = (double)(lp.length - 1u);
    if (!spectral_is_finite_f64(last_offset_d) || last_offset_d > (double)FLT_MAX) {
        lp.valid = 0;
        return lp;
    }

    alpha = spectral_segment_alpha_f32(s->omega, p->pitch_factor, p->inv_stretch);
    beta = spectral_segment_beta_f32(s->df, p->pitch_factor, p->inv_stretch_sq);
    d_amp = spectral_segment_d_amp_f32(s->da, p->inv_stretch);

    /* Segment fields and SynthParams can each be finite while their derived
     * hot-loop scalars overflow.  Reject before the backend loop consumes
     * alpha/beta/d_amp or endpoint phase/amplitude state. */
    if (!spectral_is_finite_f32(alpha) ||
        !spectral_is_finite_f32(beta) ||
        !spectral_is_finite_f32(d_amp)) {
        lp.valid = 0;
        return lp;
    }

    {
        const float last = (float)last_offset_d;
        const float phase0 = spectral_segment_phase_at_f32(s->phase, alpha, beta, 0.0f);
        const float phase1 = spectral_segment_phase_at_f32(s->phase, alpha, beta, last);
        const float amp0 = spectral_segment_amp_at_f32(s->amp, d_amp, 0.0f);
        const float amp1 = spectral_segment_amp_at_f32(s->amp, d_amp, last);

        if (!spectral_is_finite_f32(phase0) ||
            !spectral_is_finite_f32(phase1) ||
            !spectral_is_finite_f32(amp0) ||
            !spectral_is_finite_f32(amp1)) {
            lp.valid = 0;
            return lp;
        }
    }

    lp.alpha = alpha;
    lp.beta = beta;
    lp.d_amp = d_amp;
    lp.phase = s->phase;
    lp.amp = s->amp;
    lp.width = s->width;
    lp.valid = 1;

    return lp;
}



/* GPU tile preprocessing - maps segments to tiles for Metal/CUDA dispatch */
#if !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE

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

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif
    if (n_threads < 1) n_threads = 1;

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

#endif /* !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE */

/* --- GPU tile cache (process-global, single-slot) -----------------------
 * Always compiled (trivial, no GPU dependency) so the pipeline can call
 * these even in simulation/embedded builds where gpu_tile_preprocess is
 * unavailable — the cache simply stays empty. */

static SPECTRAL_THREAD_LOCAL struct {
    TileRange*  ranges;
    uint32_t*   segment_ids;
    uint32_t    num_tiles;
    uint32_t    total_refs;
    uint32_t    tile_size;
    float       stretch;
    size_t      out_len;
    int         valid;
} g_gpu_tile_cache;

void gpu_tile_cache_set(const void* ranges, const uint32_t* segment_ids,
                        uint32_t num_tiles, uint32_t total_refs,
                        float stretch, size_t out_len)
{
    g_gpu_tile_cache.ranges      = (TileRange*)ranges;
    g_gpu_tile_cache.segment_ids = (uint32_t*)segment_ids;
    g_gpu_tile_cache.num_tiles   = num_tiles;
    g_gpu_tile_cache.total_refs  = total_refs;
    g_gpu_tile_cache.tile_size   = (uint32_t)SPECTRAL_GPU_TILE_SIZE;
    g_gpu_tile_cache.stretch     = stretch;
    g_gpu_tile_cache.out_len     = out_len;
    g_gpu_tile_cache.valid       = 1;
}

int gpu_tile_cache_try_get(float stretch, size_t out_len, GpuTileData* out)
{
    if (!g_gpu_tile_cache.valid || !out) return 0;
    if (g_gpu_tile_cache.tile_size != (uint32_t)SPECTRAL_GPU_TILE_SIZE) return 0;
    if (g_gpu_tile_cache.stretch != stretch || g_gpu_tile_cache.out_len != out_len) return 0;
    out->ranges      = g_gpu_tile_cache.ranges;
    out->segment_ids = g_gpu_tile_cache.segment_ids;
    out->num_tiles   = g_gpu_tile_cache.num_tiles;
    out->total_refs  = g_gpu_tile_cache.total_refs;
    return 1;
}

void gpu_tile_cache_clear(void)
{
    g_gpu_tile_cache.valid = 0;
    g_gpu_tile_cache.ranges = NULL;
    g_gpu_tile_cache.segment_ids = NULL;
    g_gpu_tile_cache.num_tiles = 0;
    g_gpu_tile_cache.total_refs = 0;
    g_gpu_tile_cache.tile_size = 0;
}

SpectralError gpu_tile_preprocess_cached(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    GpuTileData* out_td, int* out_owns_data)
{
    if (!out_td || !out_owns_data) return SPECTRAL_ERR_PARAM;
    *out_owns_data = 0;
    if (tile_size == (uint32_t)SPECTRAL_GPU_TILE_SIZE &&
        gpu_tile_cache_try_get(stretch, out_len, out_td)) {
        return SPECTRAL_OK;
    }
#if !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
    SpectralError tile_err = gpu_tile_preprocess(sa, stretch, tile_size, out_len, out_td);
    if (tile_err == SPECTRAL_OK) {
        *out_owns_data = 1;
    }
    return tile_err;
#else
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
#endif
}

/* ---------- GPU segment cache (pre-packed SegmentGpu from seg cache) ----- */

static SPECTRAL_THREAD_LOCAL struct {
    const SegmentGpu* segs;
    uint32_t          count;
    int               valid;
} g_gpu_seg_cache;

void gpu_seg_cache_set(const SegmentGpu* segs, uint32_t count)
{
    g_gpu_seg_cache.segs  = segs;
    g_gpu_seg_cache.count = count;
    g_gpu_seg_cache.valid = (segs != NULL);
}

int gpu_seg_cache_try_get(uint32_t count, const SegmentGpu** out)
{
    if (!g_gpu_seg_cache.valid || !out) return 0;
    if (g_gpu_seg_cache.count != count) return 0;
    *out = g_gpu_seg_cache.segs;
    return 1;
}

void gpu_seg_cache_clear(void)
{
    g_gpu_seg_cache.valid = 0;
    g_gpu_seg_cache.segs  = NULL;
    g_gpu_seg_cache.count = 0;
}
