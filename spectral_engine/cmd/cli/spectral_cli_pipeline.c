/* spectral_cli_pipeline.c - Pipeline orchestration */
#include "spectral_cli_pipeline.h"
#include "spectral_cli.h"
#include "spectral_synth.h"
#include "oscillator.h"
#include "spectral_segment_parser.h"
#include "spectral_seg_cache.h"
#include "spectral_hash_xx32_xx3.h"
#include "spectral_synth_internal.h"
#include "spectral_io.h"
#include "spectral_utils.h"
#include "spectral_log.h"
#include "spectral_wavetable.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spectral_omp.h"

/* Perf tracking (includes spectral_get_time_sec) */
#if !SPECTRAL_NO_PERF
#include "spectral_perf.h"
#if SPECTRAL_IS_EMBEDDED_SIM
#include "spectral_synth_arm32.h"
#endif
static inline void SPECTRAL_MAYBE_UNUSED pipeline_perf_free_input_bytes(size_t bytes) {
    if (bytes > 0) {
        perf_track_free(bytes);
    }
}
#else
static inline void SPECTRAL_MAYBE_UNUSED pipeline_perf_free_input_bytes(size_t bytes) { (void)bytes; }
#endif

typedef struct {
    int resolved;
    char output_dir[SPECTRAL_PIPELINE_PATH_CAPACITY];
    char output_wav[SPECTRAL_PIPELINE_PATH_CAPACITY];
    char output_cache_dir[SPECTRAL_PIPELINE_PATH_CAPACITY];
} PipelineOutputPaths;

static PipelineOutputPaths g_paths = {0};

static char pipeline_ascii_tolower(char c) {
    if (c >= 'A' && c <= 'Z') return (char)(c - 'A' + 'a');
    return c;
}

static int pipeline_path_has_suffix_ci(const char* path, const char* suffix) {
    size_t path_len = 0;
    size_t suffix_len = 0;
    size_t i = 0;
    const char* tail = NULL;

    if (!path || !suffix) return 0;
    path_len = strlen(path);
    suffix_len = strlen(suffix);
    if (suffix_len == 0 || path_len < suffix_len) return 0;

    tail = path + (path_len - suffix_len);
    for (i = 0; i < suffix_len; i++) {
        if (pipeline_ascii_tolower(tail[i]) != pipeline_ascii_tolower(suffix[i])) {
            return 0;
        }
    }
    return 1;
}

static int SPECTRAL_MAYBE_UNUSED pipeline_peek_first_nonspace_char(const char* path, int* out_ch) {
    FILE* f = NULL;
    int ch = 0;

    if (!path || !out_ch) return 0;
    f = fopen(path, "rb");
    if (!f) return 0;

    do {
        ch = fgetc(f);
    } while (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n');

    fclose(f);
    if (ch == EOF) return 0;
    *out_ch = ch;
    return 1;
}

typedef struct {
    SegmentArray segments;
    float* mono;
    float* output;
#if !SPECTRAL_RESTRICTED_MODE && !SPECTRAL_NO_PERF
    size_t tracked_input_bytes;
#endif
#if !SPECTRAL_NO_PERF
    size_t tracked_segment_bytes;
    size_t tracked_output_bytes;
#endif
} PipelineRunResources;

static void SPECTRAL_MAYBE_UNUSED
pipeline_resources_track_input(PipelineRunResources* resources, size_t bytes) {
#if !SPECTRAL_RESTRICTED_MODE && !SPECTRAL_NO_PERF
    if (!resources || bytes == 0) return;
    resources->tracked_input_bytes = bytes;
    perf_track_alloc(bytes);
#else
    (void)resources;
    (void)bytes;
#endif
}

static void SPECTRAL_MAYBE_UNUSED
pipeline_resources_track_render(PipelineRunResources* resources,
                                uint32_t segment_count,
                                size_t output_bytes) {
#if !SPECTRAL_NO_PERF
    if (!resources) return;
    if (!spectral_size_mul((size_t)segment_count, sizeof(Segment), &resources->tracked_segment_bytes)) {
        resources->tracked_segment_bytes = 0;
    }
    resources->tracked_output_bytes = output_bytes;
    if (resources->tracked_segment_bytes > 0) {
        perf_track_alloc(resources->tracked_segment_bytes);
    }
    if (resources->tracked_output_bytes > 0) {
        perf_track_alloc(resources->tracked_output_bytes);
    }
#else
    (void)resources;
    (void)segment_count;
    (void)output_bytes;
#endif
}

static void pipeline_reset_segment_array(SegmentArray* segments) {
    if (!segments) return;
    if (segments->capacity > 0) free(segments->segs);
    *segments = (SegmentArray){0};
}

static SegmentArray SPECTRAL_MAYBE_UNUSED
pipeline_take_segment_array(SegmentArray* segments) {
    SegmentArray taken = (SegmentArray){0};
    if (!segments) return taken;
    taken = *segments;
    *segments = (SegmentArray){0};
    return taken;
}

static void pipeline_release_resources(PipelineRunResources* resources) {
    if (!resources) return;
#if !SPECTRAL_NO_PERF
    if (resources->tracked_segment_bytes > 0) {
        perf_track_free(resources->tracked_segment_bytes);
        resources->tracked_segment_bytes = 0;
    }
    if (resources->tracked_output_bytes > 0) {
        perf_track_free(resources->tracked_output_bytes);
        resources->tracked_output_bytes = 0;
    }
#endif
#if !SPECTRAL_RESTRICTED_MODE && !SPECTRAL_NO_PERF
    if (resources->tracked_input_bytes > 0) {
        pipeline_perf_free_input_bytes(resources->tracked_input_bytes);
        resources->tracked_input_bytes = 0;
    }
#endif
    free(resources->mono);
    resources->mono = NULL;
    free(resources->output);
    resources->output = NULL;
    pipeline_reset_segment_array(&resources->segments);
}

static PipelineError ensure_dir_exists(const char* path) {
    if (spectral_is_empty_string(path)) return PIPELINE_ERR_INPUT;
    if (mkdir(path, 0775) == 0) return PIPELINE_OK;
    if (errno == EEXIST) {
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return PIPELINE_OK;
    }
    return PIPELINE_ERR_OUTPUT;
}

static PipelineError resolve_output_paths(void) {
    if (g_paths.resolved) return PIPELINE_OK;

    int in_engine_dir = (access("core", F_OK) == 0 && access("cmd/cli/spectral_cli_pipeline.c", F_OK) == 0);
    int has_repo_parent = (access("../spectral_engine", F_OK) == 0);
    int prefer_parent_output = (in_engine_dir || has_repo_parent);
    const char* candidates[] = {
        prefer_parent_output ? SPECTRAL_OUTPUT_DIR_PRIMARY : SPECTRAL_OUTPUT_DIR_FALLBACK,
        prefer_parent_output ? SPECTRAL_OUTPUT_DIR_FALLBACK : SPECTRAL_OUTPUT_DIR_PRIMARY
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        const char* base = candidates[i];
        if (ensure_dir_exists(base) != PIPELINE_OK) continue;

        if (!spectral_path_join(g_paths.output_wav, sizeof(g_paths.output_wav), base, SPECTRAL_OUTPUT_WAV_NAME)) {
            return PIPELINE_ERR_OUTPUT;
        }
        if (!spectral_path_join(g_paths.output_cache_dir, sizeof(g_paths.output_cache_dir), base, SPECTRAL_OUTPUT_CACHE_SUBDIR)) {
            return PIPELINE_ERR_OUTPUT;
        }

        size_t base_len = strlen(base);
        if (base_len >= sizeof(g_paths.output_dir)) return PIPELINE_ERR_OUTPUT;
        memcpy(g_paths.output_dir, base, base_len + 1);
        g_paths.resolved = 1;
        return PIPELINE_OK;
    }

    return PIPELINE_ERR_OUTPUT;
}

static PipelineError ensure_output_dir_exists(void) {
    return resolve_output_paths();
}

#if !SPECTRAL_RESTRICTED_MODE
static int pipeline_hash_input_file(const char* path, SpectralHashDigest* out_digest)
{
    FILE* f = NULL;
    SpectralHashFileMethod method = SPECTRAL_HASH_FILE_METHOD_ZERO_INIT;
    SpectralError err = SPECTRAL_OK;

    if (spectral_is_empty_string(path) || !out_digest) return 0;
    *out_digest = 0;

    err = spectral_hash_file_method_init(&method, SPECTRAL_HASH_FILE_STREAM);
    if (err != SPECTRAL_OK) {
        return 0;
    }

    f = fopen(path, "rb");
    if (!f) {
        spectral_hash_file_method_destroy(&method);
        return 0;
    }

    err = spectral_hash_file_method_consume_file(&method, f);
    if (fclose(f) != 0 && err == SPECTRAL_OK) {
        err = SPECTRAL_ERR_FILE_READ;
    }
    if (err == SPECTRAL_OK) {
        err = spectral_hash_file_method_digest(&method, out_digest);
    }

    spectral_hash_file_method_destroy(&method);
    return err == SPECTRAL_OK;
}

static uint64_t build_cache_key(const SpectralCliOptions* opts) {
    char stem[192] = {0};
    char input_id[256] = {0};
    SpectralHashDigest input_digest = 0;
    int len = 0;

    if (!opts || !spectral_basename_no_ext(opts->input_path, stem, sizeof(stem))
        || spectral_is_empty_string(stem)) {
        return 0;
    }
    if (!pipeline_hash_input_file(opts->input_path, &input_digest)) {
        return 0;
    }

    /* Basename is retained only as a readable namespace hint.  The digest is
     * the cache identity.  Without content identity, two different inputs named
     * "song.wav" collide when their analysis parameters match. */
    len = snprintf(input_id, sizeof(input_id), "%s|xxh=%016llx",
                   stem, (unsigned long long)input_digest);
    if (len <= 0 || (size_t)len >= sizeof(input_id)) {
        return 0;
    }

    return spectral_seg_cache_key(input_id, opts->n_fft, opts->hop,
                                  opts->db_thresh, opts->start_sec, opts->end_sec,
                                  opts->stretch, SPECTRAL_GPU_TILE_SIZE);
}
#endif

static size_t segment_array_output_length(const SegmentArray* sa) {
    size_t max_end = 0;
    if (!sa || !sa->segs) return 0;
    for (size_t i = 0; i < sa->count; i++) {
        double seg_end_f = (double)sa->segs[i].start + (double)sa->segs[i].length;
        if (!spectral_is_finite_f64(seg_end_f) || seg_end_f <= 0.0 || seg_end_f > (double)SIZE_MAX) continue;
        size_t seg_end = (size_t)seg_end_f;
        if (seg_end > max_end) max_end = seg_end;
    }
    return max_end;
}

#if SPECTRAL_RESTRICTED_MODE
static SpectralError pipeline_load_segments_with_metadata(
    const char* path,
    SegmentArray* segments,
    int* out_sample_rate)
{
    int sample_rate_loaded = 0;
    float stretch_loaded = 0.0f;
    float pitch_loaded = 0.0f;
    SpectralError load_err = SPECTRAL_OK;

    if (spectral_is_empty_string(path) || !segments || !out_sample_rate) {
        return SPECTRAL_ERR_PARAM;
    }

    load_err = segments_load(path, segments, &sample_rate_loaded, &stretch_loaded, &pitch_loaded);
    if (load_err != SPECTRAL_OK) return load_err;

    *out_sample_rate = sample_rate_loaded;
    return SPECTRAL_OK;
}
#endif /* SPECTRAL_RESTRICTED_MODE */

static void print_processing_mask_report(const SpectralProcessReport* report) {
    char requested_buf[192] = {0};
    char pending_buf[192] = {0};

    if (!report) return;
    spectral_process_mask_to_string(report->requested, requested_buf, sizeof(requested_buf));
    spectral_process_mask_to_string(report->pending, pending_buf, sizeof(pending_buf));

    if (report->pending) {
        SPECTRAL_LOG_INFO("Processing mask: %s (pending: %s)", requested_buf, pending_buf);
    } else {
        SPECTRAL_LOG_INFO("Processing mask: %s", requested_buf);
    }
}

static void prewarm_backend_if_requested(const SpectralCliOptions* opts) {
    if (!opts || !opts->enable_cache) return;
    SynthBackend selected = opts->backend;
    if (selected == BACKEND_AUTO || selected == BACKEND_EXPORT) {
        selected = spectral_backend_select_for_timbre((int)opts->timbre, 1);
    }
    if (selected == BACKEND_METAL || selected == BACKEND_CUDA) {
        const SpectralBackendVTable* vt = spectral_backend_vtable(selected);
        if (vt && vt->init) {
            vt->init();
            SPECTRAL_LOG_INFO("Cache prewarm: initialized %s backend", spectral_backend_name(selected));
        }
    }
}

static PipelineError load_wavetable(const char* path, SpectralWavetableBank* bank,
                                    SpectralTimbre timbre);
static SpectralError run_synthesis(const SpectralCliOptions* opts, SegmentArray sa,
                                   float* out_buf, size_t out_len,
                                   SpectralWavetableBank* wt_bank, double* t_synth);

/* Analysis is only available when NOT in restricted mode */
#if !SPECTRAL_RESTRICTED_MODE
#include "spectral_analysis.h"
#endif

static PipelineError pipeline_render_and_write(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    int sample_rate,
    size_t n_samples,
    SpectralWavetableBank* wt_bank,
    SpectralTimingResults* timing)
{
    double out_len_f = 0.0;
    size_t out_len = 0;
    size_t out_bytes = 0;

    if (!opts || !resources || !resources->segments.segs || !timing) {
        return PIPELINE_ERR_INPUT;
    }

    out_len_f = (double)n_samples * (double)opts->stretch;
    if (out_len_f <= 0.0 || out_len_f > (double)SIZE_MAX) {
        return PIPELINE_ERR_INPUT;
    }
    out_len = (size_t)out_len_f;
    if (out_len == 0 || !spectral_size_mul(out_len, sizeof(float), &out_bytes)) {
        return PIPELINE_ERR_INPUT;
    }

    resources->output = spectral_calloc_array(out_len, sizeof(float));
    if (!resources->output) {
        return PIPELINE_ERR_MEMORY;
    }

#if !SPECTRAL_NO_PERF
    pipeline_resources_track_render(resources, resources->segments.count, out_bytes);
#endif

    SPECTRAL_STAGE_BEGIN("synth");
    {
        SpectralError synth_err = run_synthesis(opts, resources->segments, resources->output,
                                                out_len, wt_bank, &timing->t_synth);
        if (synth_err != SPECTRAL_OK) {
            spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_CORE, synth_err,
                                     "Synthesis failed (backend=%s/%d, timbre=%s/%d, segments=%u, out_len=%zu, stretch=%.6g, pitch=%.6g)",
                                     spectral_backend_name(opts->backend), (int)opts->backend,
                                     timbre_name(opts->timbre), (int)opts->timbre,
                                     resources->segments.count, out_len,
                                     (double)opts->stretch, (double)opts->pitch);
            return PIPELINE_ERR_SYNTHESIS;
        }
    }
    SPECTRAL_STAGE_END("synth");

    SPECTRAL_STAGE_BEGIN("norm");
    {
        double norm_start = omp_get_wtime();
        spectral_normalize_float(resources->output, out_len, SPECTRAL_NORMALIZE_HEADROOM);
        timing->t_norm = omp_get_wtime() - norm_start;
    }
    SPECTRAL_STAGE_END("norm");

    if (ensure_output_dir_exists() != PIPELINE_OK) {
        SPECTRAL_LOG_ERROR("Failed to create output directory (path=%s)",
                           g_paths.resolved ? g_paths.output_dir : SPECTRAL_OUTPUT_DIR_PRIMARY);
        return PIPELINE_ERR_OUTPUT;
    }

    SPECTRAL_STAGE_BEGIN("write");
    {
        double write_start = omp_get_wtime();
        SpectralError write_err = spectral_audio_write(
            g_paths.output_wav, resources->output, out_len, sample_rate, 1);
        timing->t_write = omp_get_wtime() - write_start;
        if (write_err != SPECTRAL_OK) {
            spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_CORE, write_err,
                                     "Output write failed (path=%s, frames=%zu, sample_rate=%d)",
                                     g_paths.output_wav, out_len, sample_rate);
            return PIPELINE_ERR_OUTPUT;
        }
    }
    SPECTRAL_STAGE_END("write");
    SPECTRAL_LOG_INFO("Wrote output: %s", g_paths.output_wav);

    return PIPELINE_OK;
}

#if !SPECTRAL_RESTRICTED_MODE
typedef struct {
    int use_cache_labels;
    int print_window_range;
    int release_input_buffer_after_analysis;
} PipelineAnalyzeRequest;

static PipelineAnalyzeRequest pipeline_make_default_analyze_request(void) {
    PipelineAnalyzeRequest request = {0};
    request.print_window_range = 1;
    return request;
}

static PipelineAnalyzeRequest pipeline_make_cache_analyze_request(void)
{
    PipelineAnalyzeRequest request = {0};
    request.use_cache_labels = 1;
    request.print_window_range = 0;
    request.release_input_buffer_after_analysis = 1;
    return request;
}

static int pipeline_split_analysis_markers_enabled(void) {
    static int initialized = 0;
    static int enabled = 0;

    if (!initialized) {
        int parsed = 0;
        if (spectral_getenv_bool(SPECTRAL_ENV_STAGE_SPLIT_ANALYSIS, &parsed)) {
            enabled = parsed ? 1 : 0;
        } else {
            enabled = 0;
        }
        initialized = 1;
    }

    return enabled;
}

static void pipeline_emit_split_analysis_markers(
    uint64_t analysis_begin_ns,
    uint64_t analysis_end_ns,
    const SpectralTimingResults* timing)
{
    uint64_t analysis_span_ns = 0;
    double fft_sec = 0.0;
    double track_sec = 0.0;
    double total_sec = 0.0;
    uint64_t fft_span_ns = 0;
    uint64_t fft_end_ns = 0;

    if (!timing || analysis_end_ns <= analysis_begin_ns) return;

    analysis_span_ns = analysis_end_ns - analysis_begin_ns;
    fft_sec = timing->t_fft;
    track_sec = timing->t_track;

    if (!spectral_is_finite_f64(fft_sec) || fft_sec < 0.0) fft_sec = 0.0;
    if (!spectral_is_finite_f64(track_sec) || track_sec < 0.0) track_sec = 0.0;

    total_sec = fft_sec + track_sec;
    if (total_sec <= 0.0) return;

    fft_span_ns = (uint64_t)((double)analysis_span_ns * (fft_sec / total_sec));
    if (fft_span_ns > analysis_span_ns) fft_span_ns = analysis_span_ns;
    fft_end_ns = analysis_begin_ns + fft_span_ns;

    SPECTRAL_STAGE_BEGIN_AT("fft", analysis_begin_ns);
    SPECTRAL_STAGE_END_AT("fft", fft_end_ns);
    SPECTRAL_STAGE_BEGIN_AT("track", fft_end_ns);
    SPECTRAL_STAGE_END_AT("track", analysis_end_ns);
}

static PipelineError pipeline_analyze_input_to_segments(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    SpectralTimingResults* timing,
    size_t* out_n_samples,
    int* out_sample_rate,
    const PipelineAnalyzeRequest* request)
{
    SpectralAudioInfo audio_info = {0};
    float* windowed_audio = NULL;
    int use_cache_labels = 0;
    int print_window_range = 0;
    int release_input_buffer_after_analysis = 0;
    uint64_t analysis_begin_ns = 0;
    uint64_t analysis_end_ns = 0;
    int emit_split_analysis_markers = 0;

    if (request) {
        use_cache_labels = request->use_cache_labels;
        print_window_range = request->print_window_range;
        release_input_buffer_after_analysis = request->release_input_buffer_after_analysis;
    }

    if (!opts || !resources || !timing || !out_n_samples || !out_sample_rate) {
        return PIPELINE_ERR_INPUT;
    }

    {
        SpectralError read_err = spectral_audio_read(opts->input_path, &audio_info, &resources->mono);
        if (read_err != SPECTRAL_OK) {
            spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_CORE, read_err,
                                     "Input read failed (path=%s)", opts->input_path);
            return PIPELINE_ERR_INPUT;
        }
    }
    *out_sample_rate = audio_info.sample_rate;
    {
        size_t input_bytes = 0;
        if (!spectral_array_bytes(audio_info.frames, sizeof(float), &input_bytes)) {
            return PIPELINE_ERR_MEMORY;
        }
        pipeline_resources_track_input(resources, input_bytes);
    }

    {
        SpectralError window_err = spectral_audio_window(
            resources->mono, audio_info.frames, opts->start_sec, opts->end_sec,
            *out_sample_rate, &windowed_audio, out_n_samples);
        if (window_err != SPECTRAL_OK) {
            spectral_log_error_codef(
                SPECTRAL_ERROR_DOMAIN_CORE,
                window_err,
                "Invalid time window (start=%.6g, end=%.6g, sample_rate=%d, input_frames=%zu)",
                (double)opts->start_sec,
                (double)opts->end_sec,
                *out_sample_rate,
                audio_info.frames);
            return PIPELINE_ERR_INPUT;
        }
    }

    if (print_window_range && (opts->start_sec > 0 || opts->end_sec > 0)) {
        SPECTRAL_LOG_INFO("Window: %.3f-%.3fs (%zu frames)", opts->start_sec,
                          (opts->end_sec < 0) ? (float)audio_info.frames / (*out_sample_rate) : opts->end_sec,
                          *out_n_samples);
    }

    if (use_cache_labels) {
        SPECTRAL_LOG_INFO("Cache build: analyzing %zu frames (fft=%d hop=%d thresh=%.1f threads=%d)...",
                          *out_n_samples, opts->n_fft, opts->hop, opts->db_thresh, opts->n_threads);
    } else {
        SPECTRAL_LOG_INFO("Analyzing %zu frames (fft=%d hop=%d thresh=%.1f threads=%d)...",
                          *out_n_samples, opts->n_fft, opts->hop, opts->db_thresh, opts->n_threads);
    }

    emit_split_analysis_markers = pipeline_split_analysis_markers_enabled();

    analysis_begin_ns = spectral_log_get_monotonic_ns();
    SPECTRAL_STAGE_BEGIN_AT("analysis", analysis_begin_ns);
    resources->segments = analyze_audio(windowed_audio, *out_n_samples, *out_sample_rate,
                                        opts->n_fft, opts->hop, opts->db_thresh,
                                        &timing->t_fft, &timing->t_track);
    analysis_end_ns = spectral_log_get_monotonic_ns();
    SPECTRAL_STAGE_END_AT("analysis", analysis_end_ns);
    if (emit_split_analysis_markers) {
        pipeline_emit_split_analysis_markers(analysis_begin_ns, analysis_end_ns, timing);
    }
    if (use_cache_labels) {
        SPECTRAL_LOG_INFO("Cache build: found %u segments", resources->segments.count);
    } else {
        SPECTRAL_LOG_INFO("Found %u segments", resources->segments.count);
    }

    if (release_input_buffer_after_analysis) {
        free(resources->mono);
        resources->mono = NULL;
    }

    return PIPELINE_OK;
}

static PipelineError pipeline_analyze_from_input(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    SpectralTimingResults* timing,
    size_t* out_n_samples,
    int* out_sample_rate,
    const PipelineAnalyzeRequest* request)
{
    if (!opts || !resources || !timing || !out_n_samples || !out_sample_rate) {
        return PIPELINE_ERR_INPUT;
    }

    pipeline_reset_segment_array(&resources->segments);
    return pipeline_analyze_input_to_segments(
        opts, resources, timing, out_n_samples, out_sample_rate, request);
}
#endif

static PipelineError pipeline_apply_processing_mask(
    SegmentArray* segments,
    int sample_rate,
    int hop,
    int n_fft,
    uint32_t processing_mask)
{
    SpectralProcessReport proc_report = {0};
    SpectralError proc_err = SPECTRAL_OK;

    if (!segments) return PIPELINE_ERR_INPUT;

    proc_err = spectral_process_chain_apply(segments, sample_rate, hop, n_fft, processing_mask, &proc_report);
    if (proc_err != SPECTRAL_OK) {
        spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_CORE, proc_err,
                                 "Processing chain failed (sample_rate=%d, mask=0x%08x)",
                                 sample_rate, (unsigned)processing_mask);
        return PIPELINE_ERR_ANALYSIS;
    }
    print_processing_mask_report(&proc_report);
    return PIPELINE_OK;
}

static PipelineError pipeline_maybe_load_wavetable(
    const SpectralCliOptions* opts,
    SpectralWavetableBank* wt_bank_storage,
    SpectralWavetableBank** out_wt_bank)
{
    PipelineError wt_result = PIPELINE_OK;

    if (!opts || !wt_bank_storage || !out_wt_bank) return PIPELINE_ERR_INPUT;

    *out_wt_bank = NULL;
    if (!opts->use_wavetable || !opts->wavetable_path) return PIPELINE_OK;

    wt_result = load_wavetable(opts->wavetable_path, wt_bank_storage, opts->timbre);
    if (wt_result != PIPELINE_OK) return wt_result;

    *out_wt_bank = wt_bank_storage;
    return PIPELINE_OK;
}

static PipelineError pipeline_run_synthesis_stage(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    int sample_rate,
    size_t n_samples,
    SpectralTimingResults* timing)
{
    PipelineError result = PIPELINE_OK;
    SpectralWavetableBank wt_bank;
    SpectralWavetableBank* wt_bank_ptr = NULL;

    if (!opts || !resources || !timing) return PIPELINE_ERR_INPUT;

    result = pipeline_maybe_load_wavetable(opts, &wt_bank, &wt_bank_ptr);
    if (result != PIPELINE_OK) return result;

    result = pipeline_apply_processing_mask(&resources->segments, sample_rate, opts->hop, opts->n_fft, opts->processing_mask);
    if (result != PIPELINE_OK) return result;

    if (n_samples == 0) {
        n_samples = segment_array_output_length(&resources->segments);
    }
    return pipeline_render_and_write(opts, resources, sample_rate, n_samples, wt_bank_ptr, timing);
}

#if !SPECTRAL_RESTRICTED_MODE
static PipelineError pipeline_export_segments(
    const SpectralCliOptions* opts,
    const SegmentArray* segments,
    int sample_rate)
{
    char segments_path[SPECTRAL_PIPELINE_PATH_CAPACITY] = {0};

    if (!opts || !segments) return PIPELINE_ERR_INPUT;
    if (ensure_output_dir_exists() != PIPELINE_OK ||
        !spectral_path_join(segments_path, sizeof(segments_path),
                            g_paths.output_dir, SPECTRAL_OUTPUT_SEGMENTS_NAME)) {
        SPECTRAL_LOG_ERROR("Failed to resolve segments export path (output_dir=%s, file=%s)",
                           g_paths.resolved ? g_paths.output_dir : "<unresolved>",
                           SPECTRAL_OUTPUT_SEGMENTS_NAME);
        return PIPELINE_ERR_OUTPUT;
    }

    SPECTRAL_LOG_INFO("Exporting to %s...", segments_path);
    {
        SpectralError save_err = segments_save(
            segments_path, segments, sample_rate, opts->stretch, opts->pitch);
        if (save_err != SPECTRAL_OK) {
            spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_CORE, save_err,
                                     "Segments export failed (path=%s, segments=%u, sample_rate=%d, stretch=%.6g, pitch=%.6g)",
                                     segments_path,
                                     segments->count,
                                     sample_rate,
                                     (double)opts->stretch,
                                     (double)opts->pitch);
            return PIPELINE_ERR_OUTPUT;
        }
    }
    {
        size_t segment_bytes = 0;
        (void)spectral_size_mul((size_t)segments->count, sizeof(Segment), &segment_bytes);
        SPECTRAL_LOG_INFO("Saved %u segments (%.1f MB): %s",
                          segments->count, BYTES_TO_MB(segment_bytes), segments_path);
    }
    return PIPELINE_OK;
}

static PipelineError pipeline_try_run_export_backend(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    int sample_rate,
    SpectralTimingResults* timing,
    int* out_handled)
{
    PipelineError result = PIPELINE_OK;

    if (out_handled) *out_handled = 0;
    if (!opts || !resources || !timing || !out_handled) return PIPELINE_ERR_INPUT;
    if (opts->backend != BACKEND_EXPORT) return PIPELINE_OK;

    result = pipeline_apply_processing_mask(&resources->segments, sample_rate, opts->hop, opts->n_fft, opts->processing_mask);
    if (result != PIPELINE_OK) return result;
    result = pipeline_export_segments(opts, &resources->segments, sample_rate);
    if (result != PIPELINE_OK) return result;

    SPECTRAL_LOG_INFO("FFT: %.3fms Track: %.3fms", timing->t_fft * 1000, timing->t_track * 1000);
    *out_handled = 1;
    return PIPELINE_OK;
}

typedef struct {
    int enabled;
    int loaded;
    uint64_t key;
    char cache_dir[512];
    double t_lookup;
    SpectralSegCacheLookupResult lr; /* kept alive for mmap lifetime */
} PipelineCacheState;

static PipelineError pipeline_prepare_cache_state(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    PipelineCacheState* cache_state,
    size_t* out_n_samples,
    int* out_sample_rate)
{
    if (!opts || !resources || !cache_state || !out_n_samples || !out_sample_rate) {
        return PIPELINE_ERR_INPUT;
    }

    *cache_state = (PipelineCacheState){0};
    cache_state->enabled = opts->enable_cache;
    if (cache_state->enabled) {
        if (ensure_output_dir_exists() != PIPELINE_OK ||
            ensure_dir_exists(g_paths.output_cache_dir) != PIPELINE_OK) {
            SPECTRAL_LOG_WARN("Warning: cache disabled (cannot create cache directory)");
            cache_state->enabled = 0;
        } else {
            size_t dir_len = strlen(g_paths.output_cache_dir);
            if (dir_len >= sizeof(cache_state->cache_dir)) {
                cache_state->enabled = 0;
            } else {
                memcpy(cache_state->cache_dir, g_paths.output_cache_dir, dir_len + 1);
                cache_state->key = build_cache_key(opts);
                if (cache_state->key == 0) {
                    SPECTRAL_LOG_WARN("Warning: cache disabled (invalid cache key)");
                    cache_state->enabled = 0;
                }
            }
        }
    }

    if (cache_state->enabled) {
        double lookup_start = omp_get_wtime();
        SpectralError cache_err = spectral_seg_cache_lookup(
            cache_state->cache_dir, cache_state->key, &cache_state->lr);
        cache_state->t_lookup = omp_get_wtime() - lookup_start;
        if (cache_err == SPECTRAL_OK && cache_state->lr.hit) {
            resources->segments = spectral_seg_cache_result_take_segments(&cache_state->lr);
            *out_sample_rate = cache_state->lr.sample_rate;
            *out_n_samples = cache_state->lr.output_length;
            cache_state->loaded = 1;
            SPECTRAL_LOG_INFO("Cache hit: key=%016llx (%u segments)",
                              (unsigned long long)cache_state->key,
                              resources->segments.count);
        } else if (cache_err != SPECTRAL_OK) {
            SPECTRAL_LOG_INFO("Cache index unreadable — will rebuild");
        }
    }
    return PIPELINE_OK;
}

static PipelineError pipeline_prepare_cached_segments_for_synthesis(
    const PipelineCacheState* cache_state,
    PipelineRunResources* resources,
    SegmentArray* out_cached_segments)
{
    if (!cache_state || !resources || !out_cached_segments) {
        return PIPELINE_ERR_INPUT;
    }

    /* Both cache-hit (mmap'd, capacity=0) and cache-miss (heap, capacity>0)
     * segments are transferred here.  pipeline_reset_segment_array respects
     * the capacity flag — mmap'd segments are not freed by free(). */
    *out_cached_segments = pipeline_take_segment_array(&resources->segments);
    return PIPELINE_OK;
}

static PipelineError pipeline_ensure_segments_mutable(SegmentArray* segments)
{
    Segment* copy = NULL;
    size_t bytes = 0;

    if (!segments) return PIPELINE_ERR_INPUT;
    if (segments->count == 0 || segments->capacity > 0) return PIPELINE_OK;
    if (!segments->segs) return PIPELINE_ERR_INPUT;

    if (!spectral_array_bytes((size_t)segments->count, sizeof(Segment), &bytes)) {
        return PIPELINE_ERR_MEMORY;
    }

    copy = (Segment*)spectral_malloc_array((size_t)segments->count, sizeof(Segment));
    if (!copy) return PIPELINE_ERR_MEMORY;

    memcpy(copy, segments->segs, bytes);
    segments->segs = copy;
    segments->capacity = segments->count;
    return PIPELINE_OK;
}

static void pipeline_print_cache_mode_summary(
    const SpectralTimingResults* timing,
    int cache_was_loaded,
    int cache_built_this_run,
    double t_cache_io,
    double cached_render_total,
    double cache_mode_total)
{
    if (!timing) return;

    SPECTRAL_LOG_INFO("\n--- Cache Mode ---");
    SPECTRAL_LOG_INFO("Shader/backend prewarm: complete");
    if (t_cache_io > 0.0) {
        SPECTRAL_LOG_INFO("Cache I/O: %.1fms",
                          t_cache_io * SPECTRAL_MILLIS_PER_SECOND_D);
    }
    if (cache_built_this_run) {
        SPECTRAL_LOG_INFO("Cache build (analysis only): FFT %.1fms Track %.1fms",
                          timing->t_fft * SPECTRAL_MILLIS_PER_SECOND_D,
                          timing->t_track * SPECTRAL_MILLIS_PER_SECOND_D);
    } else if (cache_was_loaded) {
        SPECTRAL_LOG_INFO("Cache build (analysis only): skipped (cache hit)");
    } else {
        SPECTRAL_LOG_INFO("Cache build (analysis only): FFT %.1fms Track %.1fms (in-memory)",
                          timing->t_fft * SPECTRAL_MILLIS_PER_SECOND_D,
                          timing->t_track * SPECTRAL_MILLIS_PER_SECOND_D);
    }
    SPECTRAL_LOG_INFO("Render: Synth %.1fms Norm %.1fms Write %.1fms (%.1fms)",
                      timing->t_synth * SPECTRAL_MILLIS_PER_SECOND_D,
                      timing->t_norm * SPECTRAL_MILLIS_PER_SECOND_D,
                      timing->t_write * SPECTRAL_MILLIS_PER_SECOND_D,
                      cached_render_total * SPECTRAL_MILLIS_PER_SECOND_D);
    SPECTRAL_LOG_INFO("Cache-mode end-to-end total: %.1fms",
                      cache_mode_total * SPECTRAL_MILLIS_PER_SECOND_D);
}

static PipelineError pipeline_execute_cache_mode(
    const SpectralCliOptions* opts,
    PipelineRunResources* resources,
    SpectralTimingResults* timing,
    size_t* out_n_samples,
    int* out_sample_rate,
    const PipelineCacheState* cache_state,
    double cache_wall_start,
    int* out_mode_handled)
{
    PipelineError result = PIPELINE_OK;
    const int cache_was_loaded = cache_state ? cache_state->loaded : 0;

    if (out_mode_handled) *out_mode_handled = 0;
    if (!opts || !resources || !timing || !out_n_samples || !out_sample_rate || !cache_state) {
        return PIPELINE_ERR_INPUT;
    }

    if (!cache_state->enabled) return PIPELINE_OK;

    {
        int cache_built_this_run = 0;
        GpuTileData td_store = {0};
        int owns_td_store = 0;
        const int can_reuse_gpu_aux = (opts->processing_mask == SPECTRAL_PROC_NONE);

        if (!cache_was_loaded) {
            const PipelineAnalyzeRequest analyze_request =
                pipeline_make_cache_analyze_request();
            result = pipeline_analyze_from_input(
                opts, resources, timing, out_n_samples, out_sample_rate,
                &analyze_request);
            if (result != PIPELINE_OK) return result;

            /* Pre-compute GPU tile data for cache storage + this-run synth. */
#if !SPECTRAL_EMBEDDED
            {
                size_t out_len = (size_t)((double)*out_n_samples * (double)opts->stretch);
                if (out_len > 0) {
                    SpectralError te = gpu_tile_preprocess(
                        resources->segments, opts->stretch,
                        SPECTRAL_GPU_TILE_SIZE, out_len, &td_store);
                    if (te == SPECTRAL_OK) owns_td_store = 1;
                }
            }
#endif

            {
                SpectralError store_err = spectral_seg_cache_store(
                    cache_state->cache_dir, cache_state->key,
                    &resources->segments, *out_sample_rate,
                    opts->stretch, opts->pitch, *out_n_samples,
                    owns_td_store ? td_store.ranges : NULL,
                    owns_td_store ? td_store.segment_ids : NULL,
                    owns_td_store ? td_store.num_tiles : 0,
                    owns_td_store ? td_store.total_refs : 0);
                if (store_err == SPECTRAL_OK) {
                    SPECTRAL_LOG_INFO("Cache stored: key=%016llx (%u segments, %u tiles)",
                                      (unsigned long long)cache_state->key,
                                      resources->segments.count,
                                      owns_td_store ? td_store.num_tiles : 0u);
                    cache_built_this_run = 1;
                } else {
                    spectral_log_warn_codef(SPECTRAL_ERROR_DOMAIN_CORE, store_err,
                                            "Cache store failed (key=%016llx)",
                                            (unsigned long long)cache_state->key);
                }
            }
        }

        /* Reuse cached GPU aux buffers only when no processing stages run.
         * Processing stages may mutate segments before synth, which would
         * invalidate pre-packed/tiled cache payloads. */
        if (can_reuse_gpu_aux) {
            size_t out_len = (size_t)((double)*out_n_samples * (double)opts->stretch);

            /* Set the global GPU tile cache so GPU backends (CUDA / Metal)
             * skip tile preprocessing.  Prefer mmap'd tile data (cache hit)
             * over freshly-computed data (cache miss). */
            if (cache_was_loaded &&
                cache_state->lr.tile_count > 0 &&
                cache_state->lr.tile_size == SPECTRAL_GPU_TILE_SIZE &&
                fabsf(cache_state->lr.stretch - opts->stretch) <= 1e-6f) {
                gpu_tile_cache_set(cache_state->lr.tile_ranges,
                                   cache_state->lr.tile_segment_ids,
                                   cache_state->lr.tile_count,
                                   cache_state->lr.tile_total_refs,
                                   opts->stretch, out_len);
            } else if (cache_was_loaded && cache_state->lr.tile_count > 0) {
                SPECTRAL_LOG_INFO("Cache tile metadata mismatch — skipping tile reuse");
            } else if (owns_td_store && td_store.num_tiles > 0) {
                gpu_tile_cache_set(td_store.ranges, td_store.segment_ids,
                                   td_store.num_tiles, td_store.total_refs,
                                   opts->stretch, out_len);
            }

            /* Set the global GPU segment cache so GPU backends (CUDA / Metal)
             * use pre-packed SegmentGpu data directly (zero pack loop). */
            if (cache_was_loaded &&
                cache_state->lr.gpu_segs &&
                fabsf(cache_state->lr.stretch - opts->stretch) <= 1e-6f) {
                gpu_seg_cache_set(cache_state->lr.gpu_segs,
                                  cache_state->lr.seg_count);
            }
        } else if (cache_was_loaded &&
                   (cache_state->lr.tile_count > 0 || cache_state->lr.gpu_segs)) {
            SPECTRAL_LOG_INFO("Cache GPU aux reuse disabled due to processing mask");
        }

        {
            SegmentArray sa_cached = {0};
            result = pipeline_prepare_cached_segments_for_synthesis(
                cache_state, resources, &sa_cached);
            if (result != PIPELINE_OK) {
                gpu_tile_cache_clear();
                gpu_seg_cache_clear();
                if (owns_td_store) gpu_tile_data_free(&td_store);
                return result;
            }

            {
                PipelineRunResources synth_res = {0};
                synth_res.segments = sa_cached;
                sa_cached = (SegmentArray){0};

                if (opts->processing_mask != SPECTRAL_PROC_NONE) {
                    result = pipeline_ensure_segments_mutable(&synth_res.segments);
                    if (result != PIPELINE_OK) {
                        gpu_tile_cache_clear();
                        gpu_seg_cache_clear();
                        if (owns_td_store) gpu_tile_data_free(&td_store);
                        pipeline_release_resources(&synth_res);
                        return result;
                    }
                }

                double render_start = omp_get_wtime();
                result = pipeline_run_synthesis_stage(
                    opts, &synth_res, *out_sample_rate, *out_n_samples, timing);

                double cached_render_total = omp_get_wtime() - render_start;
                double cache_mode_total = omp_get_wtime() - cache_wall_start;

                gpu_tile_cache_clear();
                gpu_seg_cache_clear();

                pipeline_print_cache_mode_summary(
                    timing, cache_was_loaded, cache_built_this_run,
                    cache_state->t_lookup,
                    cached_render_total, cache_mode_total);
                timing->t_total = cache_mode_total;

                pipeline_release_resources(&synth_res);

                if (owns_td_store) gpu_tile_data_free(&td_store);

                if (result != PIPELINE_OK) return result;
            }
        }

        if (out_mode_handled) *out_mode_handled = 1;
    }
    return PIPELINE_OK;
}
#endif

/* Load wavetable from file */
static PipelineError load_wavetable(const char* path, SpectralWavetableBank* bank, 
                                     SpectralTimbre timbre) {
    int ext_spwt = 0;
    int ext_raw = 0;
    int ext_bin = 0;
    int ext_hex = 0;
    int loaded = 0;
    spectral_wavetable_init(bank);

    ext_spwt = pipeline_path_has_suffix_ci(path, ".spwt");
    ext_raw = pipeline_path_has_suffix_ci(path, ".raw");
    ext_bin = pipeline_path_has_suffix_ci(path, ".bin");
    ext_hex = pipeline_path_has_suffix_ci(path, ".hex");

    {
        WavetableError wt_result = WAVETABLE_ERR_FORMAT;

        if (ext_spwt) {
            SPECTRAL_LOG_INFO("Loading wavetable from %s (.spwt format)...", path);
            wt_result = spectral_wavetable_load(bank, path, (uint8_t)timbre);
            loaded = 1;
        } else if (ext_raw || ext_bin) {
            SPECTRAL_LOG_INFO("Loading wavetable from %s (raw format)...", path);
            wt_result = spectral_wavetable_load_raw(bank, path, (uint8_t)timbre);
            loaded = 1;
        } else if (ext_hex) {
            SPECTRAL_LOG_INFO("Loading wavetable from %s (.hex format)...", path);
            wt_result = spectral_wavetable_load_hex(bank, path, (uint8_t)timbre);
            loaded = 1;
        } else {
#if SPECTRAL_EMBEDDED
            int first = 0;
            int has_first = pipeline_peek_first_nonspace_char(path, &first);
            int prefer_hex = has_first && first == ':';
            SPECTRAL_LOG_INFO("Loading wavetable from %s (auto-detect raw/.hex)...", path);
            if (prefer_hex) {
                wt_result = spectral_wavetable_load_hex(bank, path, (uint8_t)timbre);
            } else {
                wt_result = spectral_wavetable_load_raw(bank, path, (uint8_t)timbre);
                if (wt_result != WAVETABLE_OK) {
                    wt_result = spectral_wavetable_load_hex(bank, path, (uint8_t)timbre);
                }
            }
#else
            SPECTRAL_LOG_INFO("Loading wavetable from %s (auto-detect .spwt/raw/.hex)...", path);
            wt_result = spectral_wavetable_load(bank, path, (uint8_t)timbre);
            if (wt_result != WAVETABLE_OK) {
                wt_result = spectral_wavetable_load_raw(bank, path, (uint8_t)timbre);
            }
            if (wt_result != WAVETABLE_OK) {
                wt_result = spectral_wavetable_load_hex(bank, path, (uint8_t)timbre);
            }
#endif
            loaded = 1;
        }

        if (loaded && wt_result != WAVETABLE_OK) {
            spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_WAVETABLE, wt_result,
                                     "Wavetable load failed (path=%s, timbre=%s/%d)",
                                     path, timbre_name(timbre), (int)timbre);
            return PIPELINE_ERR_WAVETABLE;
        }
    }

    if (!loaded) {
        SPECTRAL_LOG_ERROR("Wavetable load could not start (path=%s)", path);
        return PIPELINE_ERR_WAVETABLE;
    }

    {
        const char* format_label = "auto";
        if (ext_spwt) format_label = ".spwt";
        else if (ext_raw || ext_bin) format_label = "raw";
        else if (ext_hex) format_label = ".hex";
        SPECTRAL_LOG_INFO("Wavetable loaded (%d samples, format=%s)",
                          SPECTRAL_WAVETABLE_SIZE, format_label);
    }

    return PIPELINE_OK;
}

/* Run synthesis with appropriate backend via engine dispatch */
static SpectralError run_synthesis(const SpectralCliOptions* opts, SegmentArray sa,
                                   float* out_buf, size_t out_len,
                                   SpectralWavetableBank* wt_bank, double* t_synth) {
    SynthBackend effective_backend = BACKEND_CPU;
    SpectralTimbre effective_timbre = opts->timbre;

    /* Anti-alias oscillator quality is a CPU-float-path feature; the GPU (Metal/
     * CUDA) and Q15-native backends ignore osc_get_quality().  To let the user
     * actually audition a non-naive mode, force the CPU backend for it.  NAIVE
     * leaves the requested backend untouched (and renders bit-identically). */
    osc_set_quality(opts->osc_quality);

    /* CPU oscillator execution strategy: SIMD by default (~1.8x on saw/square,
     * <=1 ULP vs scalar), scalar reference on --scalar.  Only affects the CPU
     * float path; GPU and Q15-native backends have their own kernels. */
    osc_set_dispatch(opts->osc_force_scalar ? OSC_DISPATCH_ALL_SCALAR
                                            : OSC_DISPATCH_ALL_SIMD);

    SynthBackend req_backend = opts->backend;
    if (opts->osc_quality != SPECTRAL_OSC_QUALITY_NAIVE) {
        req_backend = BACKEND_CPU;
        SPECTRAL_LOG_INFO("Oscillator quality: %s (CPU float path; forcing CPU backend)",
                          osc_quality_name(opts->osc_quality));
    }

    /* Opt-in Q15 fixed-point compute domain (--q15). The Q15 kernels (packed
     * 8xQ15 SIMD, or scalar Q15 under --scalar) live on the CPU float-synth
     * dispatch — GPU backends have their own kernels and ignore the mask — so,
     * like the quality modes, enabling it forces the CPU backend. A render uses a
     * single timbre, so only that timbre's bit matters; timbres without a Q15
     * path (asin/quantized/pwm) stay on float. */
    if (opts->enable_q15) {
        if (osc_q15_available(opts->timbre)) {
            osc_set_q15_enable(OSC_Q15_BIT(opts->timbre));
            req_backend = BACKEND_CPU;
            int packed = 0;
#if defined(OSC_SIMD_GENERIC)
            packed = !opts->osc_force_scalar && osc_simd_q15_available(opts->timbre);
#endif
            SPECTRAL_LOG_INFO("Q15 compute domain: ENABLED for %s (%s; forcing CPU backend)",
                              timbre_name(opts->timbre),
                              packed ? "packed 8-wide SIMD Q15" : "scalar Q15");
        } else {
            SPECTRAL_LOG_INFO("Q15 compute domain requested but unavailable for %s; rendering float",
                              timbre_name(opts->timbre));
        }
    }

    SPECTRAL_LOG_INFO("Rendering request=%s (%u segs, timbre=%s, threads=%d)%s...",
                      spectral_backend_name(req_backend), sa.count, timbre_name(opts->timbre),
                      opts->n_threads, wt_bank ? " (wavetable)" : "");

    SpectralError err = spectral_synth_dispatch_ex(sa, out_buf, out_len,
                                                   opts->stretch, opts->pitch, opts->timbre,
                                                   req_backend, wt_bank,
                                                   opts->n_threads, t_synth,
                                                   &effective_backend, &effective_timbre);
    if (err == SPECTRAL_OK) {
#if SPECTRAL_IS_EMBEDDED_SIM && !SPECTRAL_NO_PERF
        {
            EmbeddedTargetConfig sim_cfg;
            EmbeddedPerfEstimate sim_est;
            EmbeddedMemoryUsage sim_mem;
            if (embedded_sim_last_report(&sim_cfg, &sim_est, &sim_mem)) {
                embedded_perf_print(&sim_cfg, &sim_est);
                embedded_memory_print(&sim_mem);
            }
        }
#endif
        SPECTRAL_LOG_INFO("Backend used: %s", spectral_backend_name(effective_backend));
        SPECTRAL_LOG_INFO("Timbre used: %s", timbre_name(effective_timbre));
#if HAS_CUDA
        if (effective_backend == BACKEND_CUDA) {
            size_t vram = cuda_vram_usage_bytes();
            SPECTRAL_LOG_INFO("VRAM used: %.1f MB", (double)vram / (1024.0 * 1024.0));
        }
#endif
    }
    return err;
}

PipelineError spectral_pipeline_run(const SpectralCliOptions* opts, 
                                     SpectralTimingResults* timing) {
    SpectralCliOptions run_opts_storage;
    const SpectralCliOptions* run_opts = NULL;

    if (!opts) return PIPELINE_ERR_INPUT;
    run_opts_storage = *opts;
    if (!spectral_cli_validate(&run_opts_storage)) {
        if (run_opts_storage.error_message) {
            SPECTRAL_LOG_ERROR("%s", run_opts_storage.error_message);
        }
        return PIPELINE_ERR_INPUT;
    }
    run_opts = &run_opts_storage;

    PipelineError result = PIPELINE_OK;
    SpectralTimingResults t = {0};
    PipelineRunResources resources = {0};
    int sample_rate = 0;
    size_t n_samples = 0;

#if !SPECTRAL_NO_PERF
    double wall_start = omp_get_wtime();
#endif

    omp_set_num_threads(run_opts->n_threads);
    prewarm_backend_if_requested(run_opts);

#if !SPECTRAL_NO_PERF
    perf_reset_tracking();
    PerfMetrics perf_start = perf_snapshot(wall_start);
#endif

#if SPECTRAL_RESTRICTED_MODE
    SPECTRAL_LOG_INFO("Loading segments from %s...", run_opts->input_path);
    {
        SpectralError load_err = pipeline_load_segments_with_metadata(
            run_opts->input_path, &resources.segments, &sample_rate);
        if (load_err != SPECTRAL_OK) {
            spectral_log_error_codef(SPECTRAL_ERROR_DOMAIN_CORE, load_err,
                                     "Segment input load failed (path=%s)",
                                     run_opts->input_path);
            result = PIPELINE_ERR_INPUT;
            goto cleanup;
        }
    }
    SPECTRAL_LOG_INFO("Loaded %u segments (sr=%d)", resources.segments.count, sample_rate);
    n_samples = segment_array_output_length(&resources.segments);
#else
    PipelineCacheState cache_state = {0};
    int cache_mode_handled = 0;
    double cache_wall_start = omp_get_wtime();

    result = pipeline_prepare_cache_state(
        run_opts, &resources, &cache_state, &n_samples, &sample_rate);
    if (result != PIPELINE_OK) goto cleanup;

    result = pipeline_execute_cache_mode(
        run_opts, &resources, &t, &n_samples, &sample_rate, &cache_state,
        cache_wall_start, &cache_mode_handled);
    if (result != PIPELINE_OK) goto cleanup;
    if (cache_mode_handled) {
        if (timing) *timing = t;
        result = PIPELINE_OK;
#if !SPECTRAL_NO_PERF
        {
            PerfMetrics perf_end = perf_snapshot(wall_start);
            perf_print(&perf_start, &perf_end, run_opts->n_threads);
        }
#endif
        goto cleanup;
    }

    if (!cache_state.loaded) {
        int export_handled = 0;
        const PipelineAnalyzeRequest analyze_request =
            pipeline_make_default_analyze_request();
        result = pipeline_analyze_from_input(
            run_opts, &resources, &t, &n_samples, &sample_rate,
            &analyze_request);
        if (result != PIPELINE_OK) {
            goto cleanup;
        }

        result = pipeline_try_run_export_backend(
            run_opts, &resources, sample_rate, &t, &export_handled);
        if (result != PIPELINE_OK) goto cleanup;
        if (export_handled) {
            if (timing) *timing = t;
            result = PIPELINE_OK;
            goto cleanup;
        }
    }
#endif

    result = pipeline_run_synthesis_stage(run_opts, &resources, sample_rate, n_samples, &t);
    if (result != PIPELINE_OK) goto cleanup;

    t.t_total = t.t_fft + t.t_track + t.t_synth + t.t_norm + t.t_write;
    t.audio_dur = (double)n_samples / sample_rate;
    t.realtime_x = (t.t_total > 0) ? (t.audio_dur / t.t_total) : 0.0;
    spectral_pipeline_print_timing(&t, resources.segments.count);

#if !SPECTRAL_NO_PERF
    {
        PerfMetrics perf_end = perf_snapshot(wall_start);
        perf_print(&perf_start, &perf_end, run_opts->n_threads);
    }
#endif

    if (timing) *timing = t;
    result = PIPELINE_OK;

cleanup:
#if !SPECTRAL_RESTRICTED_MODE
    spectral_seg_cache_result_free(&cache_state.lr);
#endif
    pipeline_release_resources(&resources);
    return result;
}

void spectral_pipeline_print_timing(const SpectralTimingResults* t, uint32_t segment_count) {
    if (!t) return;
    
#if SPECTRAL_RESTRICTED_MODE
    SPECTRAL_LOG_INFO("\n--- Timing ---");
    SPECTRAL_LOG_INFO("Synth: %.1fms Norm: %.1fms Write: %.1fms Total: %.1fms",
                      t->t_synth*1000, t->t_norm*1000, t->t_write*1000, t->t_total*1000);
#else
    SPECTRAL_LOG_INFO("\n--- Timing ---");
    SPECTRAL_LOG_INFO("FFT: %.1fms Track: %.1fms Synth: %.1fms Norm: %.1fms Write: %.1fms Total: %.1fms",
                      t->t_fft*1000, t->t_track*1000, t->t_synth*1000, t->t_norm*1000, t->t_write*1000, t->t_total*1000);
#endif
    SPECTRAL_LOG_INFO("Audio: %.2fs Realtime: %.1fx Segs/sec: %.0fK",
                      t->audio_dur, t->realtime_x,
                      (t->t_total > 0) ? (segment_count / t->t_total / 1000) : 0.0);
}
