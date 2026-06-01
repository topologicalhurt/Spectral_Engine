/* spectral_synth_internal.c - Shared Synthesis Helpers */

#include "spectral_synth_internal.h"
#include "spectral_synth.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include <stdlib.h>
#include <string.h>
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
static SynthPreflight synth_preflight_common(
    void* out_buffer, size_t out_len, size_t elem_size, SegmentArray sa,
    float stretch, float pitch, double** t_synth)
{
    SynthPreflight pf = {0};
    size_t preflight_out_bytes = 0;

    pf.error = SPECTRAL_OK;

    if (!t_synth) {
        pf.error = SPECTRAL_ERR_PARAM;
        return pf;
    }
    if (!*t_synth) {
        *t_synth = &g_synth_timing_dummy;
    }

    if (elem_size == 0u) {
        **t_synth = 0;
        pf.error = SPECTRAL_ERR_PARAM;
        return pf;
    }
    if (!spectral_size_mul(out_len, elem_size, &preflight_out_bytes)) {
        **t_synth = 0;
        pf.error = SPECTRAL_ERR_OVERFLOW;
        return pf;
    }

    if (!out_buffer || out_len == 0u) {
        **t_synth = 0;
        return pf;
    }

    if (sa.count == 0u || !sa.segs) {
        memset(out_buffer, 0, preflight_out_bytes);
        **t_synth = 0;
        return pf;
    }

    pf.error = synth_validate_params(stretch, pitch);
    if (pf.error != SPECTRAL_OK) {
        memset(out_buffer, 0, preflight_out_bytes);
        **t_synth = 0;
        return pf;
    }

    if (sa.count > UINT32_MAX) {
        pf.error = SPECTRAL_ERR_OVERFLOW;
        memset(out_buffer, 0, preflight_out_bytes);
        **t_synth = 0;
        return pf;
    }

    if (!spectral_segment_array_valid_for_synth(&sa)) {
        pf.error = SPECTRAL_ERR_PARAM;
        memset(out_buffer, 0, preflight_out_bytes);
        **t_synth = 0;
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

    if (!spectral_segment_valid_for_synth(s)) {
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



/* GPU tile preprocessing (gpu_tile_preprocess) is profile-divergent and lives
 * in build-selected port files behind the spectral_synth_internal.h interface:
 * core/port/host/spectral_gpu_tile.c (real Metal/CUDA-oriented mapping) and
 * core/port/embedded/spectral_gpu_tile.c (SPECTRAL_ERR_BACKEND_UNAVAIL stub,
 * no GPU backend). The cache below is device-agnostic and always compiled. */

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
    if (num_tiles == 0u || (total_refs > 0u && (!ranges || !segment_ids))) {
        gpu_tile_cache_clear();
        return;
    }
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
    GpuTileData td = {0};

    if (!g_gpu_tile_cache.valid || !out) return 0;
    if (g_gpu_tile_cache.tile_size != (uint32_t)SPECTRAL_GPU_TILE_SIZE ||
        g_gpu_tile_cache.stretch != stretch ||
        g_gpu_tile_cache.out_len != out_len) {
        gpu_tile_cache_clear();
        return 0;
    }

    td.ranges      = g_gpu_tile_cache.ranges;
    td.segment_ids = g_gpu_tile_cache.segment_ids;
    td.num_tiles   = g_gpu_tile_cache.num_tiles;
    td.total_refs  = g_gpu_tile_cache.total_refs;

    /* Tile layout depends on segment start/length as well as output shape.
     * Since this cache key does not encode segment identity, it is a one-shot
     * handoff from the analysis/cache path to the immediately following GPU
     * synthesis call. */
    gpu_tile_cache_clear();

    *out = td;
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
        if (sa.count <= (size_t)UINT32_MAX &&
            spectral_gpu_tile_layout_words_valid((const void*)out_td->ranges,
                                             out_td->segment_ids,
                                             out_td->num_tiles,
                                             out_td->total_refs,
                                             (uint32_t)sa.count)) {
            return SPECTRAL_OK;
        }
        gpu_tile_cache_clear();
        *out_td = (GpuTileData){0};
    }
    SpectralError tile_err = gpu_tile_preprocess(sa, stretch, tile_size, out_len, out_td);
    if (tile_err == SPECTRAL_OK) {
        *out_owns_data = 1;
    }
    return tile_err;
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
    const SegmentGpu* segs = NULL;

    if (!g_gpu_seg_cache.valid || !out) return 0;
    if (g_gpu_seg_cache.count != count || !g_gpu_seg_cache.segs) {
        gpu_seg_cache_clear();
        return 0;
    }

    segs = g_gpu_seg_cache.segs;
    *out = segs;

    /* Process-local cache entries are scoped to the immediately following
     * backend invocation.  A count-only cache key is not a segment identity, so
     * never allow a previous call's SegmentGpu pointer to be reused implicitly. */
    gpu_seg_cache_clear();
    return 1;
}


void gpu_seg_cache_clear(void)
{
    g_gpu_seg_cache.valid = 0;
    g_gpu_seg_cache.segs  = NULL;
    g_gpu_seg_cache.count = 0;
}

SpectralError spectral_gpu_dispatch_plan_init(SpectralGpuDispatchPlan* plan,
                                              SegmentArray sa,
                                              const SynthParams* params,
                                              float stretch,
                                              SpectralTimbre timbre,
                                              size_t out_len)
{
    SpectralError err = SPECTRAL_OK;
    const SegmentGpu* cached_gpu_segs = NULL;

    if (!plan || !params) return SPECTRAL_ERR_PARAM;
    *plan = (SpectralGpuDispatchPlan){0};

    if (!spectral_array_bytes((size_t)sa.count, sizeof(SegmentGpu), &plan->segment_bytes)) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    if (gpu_seg_cache_try_get(sa.count, &cached_gpu_segs)) {
        plan->segment_source = cached_gpu_segs;
    }

    err = gpu_tile_preprocess_cached(sa, stretch, SPECTRAL_GPU_TILE_SIZE,
                                     out_len, &plan->tiles, &plan->owns_tile_data);
    if (err != SPECTRAL_OK) {
        spectral_gpu_dispatch_plan_free(plan);
        return err;
    }

    err = gpu_synth_params_pack_checked(params, SPECTRAL_GPU_TILE_SIZE, timbre, &plan->params);
    if (err != SPECTRAL_OK) {
        spectral_gpu_dispatch_plan_free(plan);
        return err;
    }

    if (plan->tiles.total_refs == 0u) {
        plan->zero_output = 1;
        return SPECTRAL_OK;
    }

    if (!spectral_array_bytes((size_t)plan->tiles.total_refs, sizeof(uint32_t), &plan->tile_ids_bytes) ||
        !spectral_array_bytes((size_t)plan->tiles.num_tiles, sizeof(TileRange), &plan->tile_ranges_bytes)) {
        spectral_gpu_dispatch_plan_free(plan);
        return SPECTRAL_ERR_OVERFLOW;
    }

    return SPECTRAL_OK;
}

void spectral_gpu_dispatch_plan_free(SpectralGpuDispatchPlan* plan)
{
    if (!plan) return;
    if (plan->owns_tile_data) {
        gpu_tile_data_free(&plan->tiles);
    }
    *plan = (SpectralGpuDispatchPlan){0};
}

