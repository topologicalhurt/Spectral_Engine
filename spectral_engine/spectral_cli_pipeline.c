/* spectral_cli_pipeline.c - Pipeline orchestration */
#include "spectral_cli_pipeline.h"
#include "spectral_synth.h"
#include "spectral_segment_parser.h"
#include "spectral_io.h"
#include "spectral_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

#include "spectral_omp.h"

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

/* Perf tracking (includes spectral_get_time_sec) */
#if !SPECTRAL_NO_PERF
#include "spectral_perf.h"
#define HAS_PERF 1
#define PERF_FREE_INPUT_BYTES(bytes) do { if ((bytes) > 0) perf_track_free((bytes)); } while (0)
#else
#define HAS_PERF 0
#define PERF_FREE_INPUT_BYTES(bytes) ((void)(bytes))
#endif

#define OUTPUT_DIR_PRIMARY "../output"
#define OUTPUT_DIR_FALLBACK "output"
#define OUTPUT_WAV_NAME "out_c.wav"
#define OUTPUT_CACHE_SUBDIR "cache"

typedef struct {
    int resolved;
    char output_dir[PATH_MAX];
    char output_wav[PATH_MAX];
    char output_cache_dir[PATH_MAX];
} PipelineOutputPaths;

static PipelineOutputPaths g_paths = {0};

static PipelineError ensure_dir_exists(const char* path) {
    if (!path || !path[0]) return PIPELINE_ERR_INPUT;
    if (mkdir(path, 0775) == 0) return PIPELINE_OK;
    if (errno == EEXIST) {
        struct stat st;
        if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) return PIPELINE_OK;
    }
    return PIPELINE_ERR_OUTPUT;
}

static int path_join(char* out, size_t out_size, const char* a, const char* b) {
    if (!out || out_size == 0 || !a || !b) return 0;
    int w = snprintf(out, out_size, "%s/%s", a, b);
    return (w > 0 && (size_t)w < out_size);
}

static PipelineError resolve_output_paths(void) {
    if (g_paths.resolved) return PIPELINE_OK;

    int in_engine_dir = (access("core", F_OK) == 0 && access("spectral_cli_pipeline.c", F_OK) == 0);
    const char* candidates[] = {
        in_engine_dir ? OUTPUT_DIR_PRIMARY : OUTPUT_DIR_FALLBACK,
        in_engine_dir ? OUTPUT_DIR_FALLBACK : OUTPUT_DIR_PRIMARY
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        const char* base = candidates[i];
        if (ensure_dir_exists(base) != PIPELINE_OK) continue;

        if (!path_join(g_paths.output_wav, sizeof(g_paths.output_wav), base, OUTPUT_WAV_NAME)) {
            return PIPELINE_ERR_OUTPUT;
        }
        if (!path_join(g_paths.output_cache_dir, sizeof(g_paths.output_cache_dir), base, OUTPUT_CACHE_SUBDIR)) {
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

static const char* basename_no_ext(const char* path, char* out, size_t out_size) {
    if (!path || !out || out_size == 0) return NULL;
    const char* base = strrchr(path, '/');
    base = base ? (base + 1) : path;
    const char* dot = strrchr(base, '.');
    size_t len = dot && dot > base ? (size_t)(dot - base) : strlen(base);
    if (len >= out_size) len = out_size - 1;
    memcpy(out, base, len);
    out[len] = '\0';
    return out;
}

static PipelineError build_cache_path(const SpectralCliOptions* opts, char* out, size_t out_size) {
    if (!opts || !out || out_size == 0) return PIPELINE_ERR_INPUT;
    if (resolve_output_paths() != PIPELINE_OK) return PIPELINE_ERR_OUTPUT;

    char stem[192] = {0};
    if (!basename_no_ext(opts->input_path, stem, sizeof(stem)) || stem[0] == '\0') {
        return PIPELINE_ERR_INPUT;
    }

    int db10 = (int)(opts->db_thresh * 10.0f);
    int start_ms = (int)(opts->start_sec * 1000.0f);
    int end_ms = (int)(opts->end_sec * 1000.0f);
    int w = snprintf(out, out_size,
                     "%s/%s_n%d_h%d_db%d_s%d_e%d.segbin",
                     g_paths.output_cache_dir, stem,
                     opts->n_fft, opts->hop, db10, start_ms, end_ms);
    return (w > 0 && (size_t)w < out_size) ? PIPELINE_OK : PIPELINE_ERR_INPUT;
}

static size_t segment_array_output_length(const SegmentArray* sa) {
    size_t max_end = 0;
    if (!sa || !sa->segs) return 0;
    for (size_t i = 0; i < sa->count; i++) {
        double seg_end_f = (double)sa->segs[i].start + (double)sa->segs[i].length;
        if (!isfinite(seg_end_f) || seg_end_f <= 0.0 || seg_end_f > (double)SIZE_MAX) continue;
        size_t seg_end = (size_t)seg_end_f;
        if (seg_end > max_end) max_end = seg_end;
    }
    return max_end;
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
            printf("Cache prewarm: initialized %s backend\n", spectral_backend_name(selected));
        }
    }
}

static PipelineError load_wavetable(const char* path, SpectralWavetableBank* bank,
                                    SpectralTimbre timbre);
static SpectralError run_synthesis(const SpectralCliOptions* opts, SegmentArray sa,
                                   float* out_buf, size_t out_len,
                                   SpectralWavetableBank* wt_bank, double* t_synth);

static PipelineError run_synthesis_from_segments(
    const SpectralCliOptions* opts,
    SegmentArray* sa,
    int sample_rate,
    SpectralTimingResults* t,
    float* mono,
    size_t input_alloc_bytes)
{
    if (!opts || !sa || !sa->segs || !t) return PIPELINE_ERR_INPUT;

    SpectralWavetableBank wt_bank;
    SpectralWavetableBank* wt_bank_ptr = NULL;
    if (opts->use_wavetable && opts->wavetable_path) {
        PipelineError wt_result = load_wavetable(opts->wavetable_path, &wt_bank, opts->timbre);
        if (wt_result != PIPELINE_OK) {
            free(sa->segs);
            if (mono) free(mono);
            return wt_result;
        }
        wt_bank_ptr = &wt_bank;
    }

    SpectralProcessReport proc_report = {0};
    SpectralError proc_err = spectral_process_chain_apply(
        sa, sample_rate, opts->processing_mask, &proc_report);
    if (proc_err != SPECTRAL_OK) {
        printf("Error: Processing chain failed (%s)\n", spectral_strerror(proc_err));
        free(sa->segs);
        if (mono) free(mono);
        return PIPELINE_ERR_ANALYSIS;
    }

    {
        char requested_buf[192] = {0};
        char pending_buf[192] = {0};
        spectral_process_mask_to_string(proc_report.requested, requested_buf, sizeof(requested_buf));
        spectral_process_mask_to_string(proc_report.pending, pending_buf, sizeof(pending_buf));
        if (proc_report.pending) {
            printf("Processing mask: %s (pending: %s)\n", requested_buf, pending_buf);
        } else {
            printf("Processing mask: %s\n", requested_buf);
        }
    }

    size_t n_samples = segment_array_output_length(sa);
    double out_len_f = (double)n_samples * (double)opts->stretch;
    if (out_len_f <= 0.0 || out_len_f > (double)SIZE_MAX) {
        free(sa->segs);
        if (mono) free(mono);
        return PIPELINE_ERR_INPUT;
    }
    size_t out_len = (size_t)out_len_f;
    if (out_len == 0 || out_len > SIZE_MAX / sizeof(float)) {
        free(sa->segs);
        if (mono) free(mono);
        return PIPELINE_ERR_INPUT;
    }

    float* out_buf = calloc(out_len, sizeof(float));
    if (!out_buf) {
        free(sa->segs);
        if (mono) free(mono);
        return PIPELINE_ERR_MEMORY;
    }

#if HAS_PERF
    perf_track_alloc(out_len * sizeof(float));
    perf_track_alloc(sa->count * sizeof(Segment));
#endif

    SpectralError synth_err = run_synthesis(opts, *sa, out_buf, out_len, wt_bank_ptr, &t->t_synth);
    if (synth_err != SPECTRAL_OK) {
        printf("Error: Synthesis failed (%s)\n", spectral_strerror(synth_err));
#if HAS_PERF
        perf_track_free(sa->count * sizeof(Segment));
        perf_track_free(out_len * sizeof(float));
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        free(sa->segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_SYNTHESIS;
    }

    double norm_start = omp_get_wtime();
    spectral_normalize_float(out_buf, out_len, SPECTRAL_NORMALIZE_HEADROOM);
    t->t_norm = omp_get_wtime() - norm_start;

    if (ensure_output_dir_exists() != PIPELINE_OK) {
        printf("Error: Failed to create output directory (%s)\n",
               g_paths.resolved ? g_paths.output_dir : OUTPUT_DIR_PRIMARY);
#if HAS_PERF
        perf_track_free(sa->count * sizeof(Segment));
        perf_track_free(out_len * sizeof(float));
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        free(sa->segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_OUTPUT;
    }

    SpectralError write_err = spectral_audio_write(g_paths.output_wav, out_buf, out_len, sample_rate, 1);
    if (write_err != SPECTRAL_OK) {
        printf("Error: Failed to write output file (%s)\n", spectral_strerror(write_err));
#if HAS_PERF
        perf_track_free(sa->count * sizeof(Segment));
        perf_track_free(out_len * sizeof(float));
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        free(sa->segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_OUTPUT;
    }
    printf("Wrote output: %s\n", g_paths.output_wav);

#if HAS_PERF
    perf_track_free(sa->count * sizeof(Segment));
    perf_track_free(out_len * sizeof(float));
    PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif

    free(sa->segs);
    free(out_buf);
    if (mono) free(mono);
    return PIPELINE_OK;
}

/* Analysis is only available when NOT in restricted mode */
#if !defined(SPECTRAL_RESTRICTED_MODE) || !SPECTRAL_RESTRICTED_MODE
#include "spectral_analysis.h"
#define HAS_ANALYSIS 1
#else
#define HAS_ANALYSIS 0
#endif

/* Load wavetable from file */
static PipelineError load_wavetable(const char* path, SpectralWavetableBank* bank, 
                                     SpectralTimbre timbre) {
    spectral_wavetable_init(bank);
    
    size_t len = strlen(path);
    WavetableError wt_result;
    
    if (len > 5 && strcmp(path + len - 5, ".spwt") == 0) {
        printf("Loading wavetable from %s (.spwt format)...\n", path);
        wt_result = spectral_wavetable_load(bank, path, (uint8_t)timbre);
    } else if (len > 4 && strcmp(path + len - 4, ".hex") == 0) {
        printf("Loading wavetable from %s (Intel HEX)...\n", path);
        wt_result = spectral_wavetable_load_hex(bank, path, (uint8_t)timbre);
    } else {
        printf("Loading wavetable from %s (raw binary)...\n", path);
        wt_result = spectral_wavetable_load_raw(bank, path, (uint8_t)timbre);
    }
    
    if (wt_result != WAVETABLE_OK) {
        printf("Error: Failed to load wavetable: %s\n", wavetable_strerror(wt_result));
        return PIPELINE_ERR_WAVETABLE;
    }
    
    printf("Wavetable loaded (%d samples)\n", SPECTRAL_WAVETABLE_SIZE);
    return PIPELINE_OK;
}

/* Run synthesis with appropriate backend via engine dispatch */
static SpectralError run_synthesis(const SpectralCliOptions* opts, SegmentArray sa,
                                   float* out_buf, size_t out_len,
                                   SpectralWavetableBank* wt_bank, double* t_synth) {
    const char* backend_name = spectral_backend_name(opts->backend);
    printf("Rendering with %s (%u segs, timbre=%s, threads=%d)%s...\n",
           backend_name, sa.count, timbre_name(opts->timbre),
           opts->n_threads, wt_bank ? " (wavetable)" : "");

    return spectral_synth_dispatch(sa, out_buf, out_len,
                                   opts->stretch, opts->pitch, opts->timbre,
                                   opts->backend, wt_bank,
                                   opts->n_threads, t_synth);
}

PipelineError spectral_pipeline_run(const SpectralCliOptions* opts, 
                                     SpectralTimingResults* timing) {
    if (!opts || !opts->valid) return PIPELINE_ERR_INPUT;
    
    SpectralTimingResults t = {0};
    double wall_start = omp_get_wtime();
    
    omp_set_num_threads(opts->n_threads);
    prewarm_backend_if_requested(opts);
    
#if HAS_PERF
    perf_reset_tracking();
    PerfMetrics perf_start = perf_snapshot(wall_start);
#endif
    
    SegmentArray sa = {0};
    int sample_rate = 0;
    size_t n_samples = 0;
    float* mono = NULL;
    SpectralAudioInfo audio_info = {0};
    size_t input_alloc_bytes = 0;
    
#if SPECTRAL_RESTRICTED_MODE
    /* Restricted mode: load pre-analyzed segments */
    printf("Loading segments from %s...\n", opts->input_path);
    int sr_loaded;
    float stretch_loaded, pitch_loaded;
    SpectralError load_err = segments_load(opts->input_path, &sa, &sr_loaded, &stretch_loaded, &pitch_loaded);
    if (load_err != SPECTRAL_OK) {
        printf("Error: Cannot load segments from %s (%s)\n", opts->input_path, spectral_strerror(load_err));
        return PIPELINE_ERR_INPUT;
    }
    (void)stretch_loaded;
    (void)pitch_loaded;
    sample_rate = sr_loaded;
    printf("Loaded %u segments (sr=%d)\n", sa.count, sample_rate);
    
    /* Calculate output length */
    n_samples = segment_array_output_length(&sa);
    
#else
    /* Desktop/emulator mode: read and analyze audio */
    char cache_path[512] = {0};
    int cache_enabled = opts->enable_cache;
    int cache_loaded = 0;
    if (cache_enabled) {
        if (ensure_output_dir_exists() != PIPELINE_OK || ensure_dir_exists(g_paths.output_cache_dir) != PIPELINE_OK) {
            printf("Warning: cache disabled (cannot create cache directory)\n");
            cache_enabled = 0;
        } else if (build_cache_path(opts, cache_path, sizeof(cache_path)) != PIPELINE_OK) {
            printf("Warning: cache disabled (invalid cache key path)\n");
            cache_enabled = 0;
        }
    }

    if (cache_enabled && access(cache_path, F_OK) == 0) {
        int sr_loaded = 0;
        float stretch_loaded = 0.0f;
        float pitch_loaded = 0.0f;
        SpectralError cache_err = segments_load(cache_path, &sa, &sr_loaded, &stretch_loaded, &pitch_loaded);
        if (cache_err == SPECTRAL_OK) {
            (void)stretch_loaded;
            (void)pitch_loaded;
            sample_rate = sr_loaded;
            n_samples = segment_array_output_length(&sa);
            cache_loaded = 1;
            printf("Cache hit: %s (%u segments)\n", cache_path, sa.count);
        } else {
            printf("Cache miss (invalid cache file): %s\n", cache_path);
        }
    }

    if (cache_enabled) {
        double cache_mode_start = omp_get_wtime();
        int cache_built_this_run = 0;
        int cache_saved_this_run = 0;

        if (!cache_loaded) {
            SpectralError read_err = spectral_audio_read(opts->input_path, &audio_info, &mono);
            if (read_err != SPECTRAL_OK) {
                printf("Error: Cannot open %s (%s)\n", opts->input_path, spectral_strerror(read_err));
                return PIPELINE_ERR_INPUT;
            }
            sample_rate = audio_info.sample_rate;
#if HAS_PERF
            input_alloc_bytes = audio_info.frames * sizeof(float);
            perf_track_alloc(input_alloc_bytes);
#endif

            float* windowed_audio;
            SpectralError window_err = spectral_audio_window(mono, audio_info.frames, opts->start_sec, opts->end_sec,
                                                             sample_rate, &windowed_audio, &n_samples);
            if (window_err != SPECTRAL_OK) {
                printf("Error: invalid time window\n");
#if HAS_PERF
                PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
                free(mono);
                return PIPELINE_ERR_INPUT;
            }

            printf("Cache build: analyzing %zu frames (fft=%d hop=%d thresh=%.1f threads=%d)...\n",
                   n_samples, opts->n_fft, opts->hop, opts->db_thresh, opts->n_threads);
            sa = analyze_audio(windowed_audio, n_samples, sample_rate,
                               opts->n_fft, opts->hop, opts->db_thresh, &t.t_fft, &t.t_track);
            printf("Cache build: found %u segments\n", sa.count);

            SpectralError cache_save_err = segments_save(cache_path, &sa, sample_rate, opts->stretch, opts->pitch);
            if (cache_save_err == SPECTRAL_OK) {
                printf("Cache saved: %s\n", cache_path);
                cache_built_this_run = 1;
                cache_saved_this_run = 1;
            } else {
                printf("Warning: cache save failed (%s)\n", spectral_strerror(cache_save_err));
            }

#if HAS_PERF
            PERF_FREE_INPUT_BYTES(input_alloc_bytes);
            input_alloc_bytes = 0;
#endif
            free(mono);
            mono = NULL;
        }

        SegmentArray sa_cached = {0};
        if (cache_loaded) {
            /* Reuse already-loaded cache segments, avoid redundant disk I/O. */
            sa_cached = sa;
            sa = (SegmentArray){0};
        } else if (cache_saved_this_run) {
            int sr_loaded = 0;
            float stretch_loaded = 0.0f;
            float pitch_loaded = 0.0f;
            SpectralError cache_load_err = segments_load(cache_path, &sa_cached, &sr_loaded, &stretch_loaded, &pitch_loaded);
            if (cache_load_err != SPECTRAL_OK) {
                printf("Error: Cannot load cache segments %s (%s)\n", cache_path, spectral_strerror(cache_load_err));
                free(sa.segs);
                sa = (SegmentArray){0};
                return PIPELINE_ERR_INPUT;
            }
            (void)stretch_loaded;
            (void)pitch_loaded;
            sample_rate = sr_loaded;
            free(sa.segs);
            sa = (SegmentArray){0};
        } else {
            /* Cache write can fail (permissions/disk full); still render from analysis. */
            sa_cached = sa;
            sa = (SegmentArray){0};
            printf("Cache mode: using in-memory analysis result (cache artifact unavailable)\n");
        }

        double cached_synth_start = omp_get_wtime();
        PipelineError cached_run = run_synthesis_from_segments(opts, &sa_cached, sample_rate, &t, NULL, 0);
        if (cached_run != PIPELINE_OK) return cached_run;
        double cached_synth_total = omp_get_wtime() - cached_synth_start;
        double cache_mode_total = omp_get_wtime() - cache_mode_start;

        printf("\n--- Cache Mode ---\n");
        printf("Shader/backend prewarm: complete\n");
        if (cache_built_this_run) {
            printf("Cache build (analysis only): FFT %.1fms Track %.1fms\n",
                   t.t_fft * 1000.0, t.t_track * 1000.0);
        } else if (cache_loaded) {
            printf("Cache build (analysis only): skipped (cache hit)\n");
        } else {
            printf("Cache build (analysis only): FFT %.1fms Track %.1fms (cache artifact unavailable)\n",
                   t.t_fft * 1000.0, t.t_track * 1000.0);
        }
        printf("Segment-binary synth run: Synth %.1fms Norm %.1fms Total %.1fms\n",
               t.t_synth * 1000.0, t.t_norm * 1000.0, cached_synth_total * 1000.0);
        printf("Cache-mode end-to-end total: %.1fms\n", cache_mode_total * 1000.0);

        t.t_total = cache_mode_total;
        if (timing) *timing = t;
        return PIPELINE_OK;
    }

    if (!cache_loaded) {
    SpectralError read_err = spectral_audio_read(opts->input_path, &audio_info, &mono);
    if (read_err != SPECTRAL_OK) {
        printf("Error: Cannot open %s (%s)\n", opts->input_path, spectral_strerror(read_err));
        return PIPELINE_ERR_INPUT;
    }
    sample_rate = audio_info.sample_rate;
    
#if HAS_PERF
    input_alloc_bytes = audio_info.frames * sizeof(float);
    perf_track_alloc(input_alloc_bytes);
#endif
    
    float* windowed_audio;
    SpectralError window_err = spectral_audio_window(mono, audio_info.frames, opts->start_sec, opts->end_sec,
                                                     sample_rate, &windowed_audio, &n_samples);
    if (window_err != SPECTRAL_OK) {
        printf("Error: invalid time window\n");
#if HAS_PERF
    PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        free(mono);
        return PIPELINE_ERR_INPUT;
    }
    
    if (opts->start_sec > 0 || opts->end_sec > 0) {
        printf("Window: %.3f-%.3fs (%zu frames)\n", opts->start_sec,
               (opts->end_sec < 0) ? (float)audio_info.frames/sample_rate : opts->end_sec,
               n_samples);
    }

    printf("Analyzing %zu frames (fft=%d hop=%d thresh=%.1f threads=%d)...\n",
           n_samples, opts->n_fft, opts->hop, opts->db_thresh, opts->n_threads);
    
    sa = analyze_audio(windowed_audio, n_samples, sample_rate,
                       opts->n_fft, opts->hop, opts->db_thresh, &t.t_fft, &t.t_track);
    printf("Found %u segments\n", sa.count);

    if (cache_enabled && cache_path[0]) {
        SpectralError cache_save_err = segments_save(cache_path, &sa, sample_rate, opts->stretch, opts->pitch);
        if (cache_save_err == SPECTRAL_OK) {
            printf("Cache saved: %s\n", cache_path);
        } else {
            printf("Warning: cache save failed (%s)\n", spectral_strerror(cache_save_err));
        }
    }

    if (opts->backend == BACKEND_EXPORT) {
        SpectralProcessReport proc_report = {0};
        SpectralError proc_err = spectral_process_chain_apply(
            &sa, sample_rate, opts->processing_mask, &proc_report);
        if (proc_err != SPECTRAL_OK) {
            printf("Error: Processing chain failed (%s)\n", spectral_strerror(proc_err));
#if HAS_PERF
            PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
            free(sa.segs);
            free(mono);
            return PIPELINE_ERR_ANALYSIS;
        }

        {
            char requested_buf[192] = {0};
            char pending_buf[192] = {0};
            spectral_process_mask_to_string(proc_report.requested, requested_buf, sizeof(requested_buf));
            spectral_process_mask_to_string(proc_report.pending, pending_buf, sizeof(pending_buf));
            if (proc_report.pending) {
                printf("Processing mask: %s (pending: %s)\n", requested_buf, pending_buf);
            } else {
                printf("Processing mask: %s\n", requested_buf);
            }
        }
    }
    
    /* Handle export mode */
    if (opts->backend == BACKEND_EXPORT) {
        printf("Exporting to segments.bin...\n");
        SpectralError save_err = segments_save("segments.bin", &sa, sample_rate, opts->stretch, opts->pitch);
        if (save_err == SPECTRAL_OK) {
            printf("Saved %u segments (%.1f MB)\n", sa.count,
                   BYTES_TO_MB(sa.count * sizeof(Segment)));
        } else {
            printf("Error: Failed to save segments (%s)\n", spectral_strerror(save_err));
#if HAS_PERF
            PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
            free(sa.segs);
            free(mono);
            return PIPELINE_ERR_OUTPUT;
        }
#if HAS_PERF
    PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        free(sa.segs);
        free(mono);
        printf("FFT: %.3fms Track: %.3fms\n", t.t_fft*1000, t.t_track*1000);
        if (timing) *timing = t;
        return PIPELINE_OK;
    }
    }
#endif
    
    /* Load wavetable if requested */
    SpectralWavetableBank wt_bank;
    SpectralWavetableBank* wt_bank_ptr = NULL;
    if (opts->use_wavetable && opts->wavetable_path) {
        PipelineError wt_result = load_wavetable(opts->wavetable_path, &wt_bank, opts->timbre);
        if (wt_result != PIPELINE_OK) {
            free(sa.segs);
#if HAS_PERF && !SPECTRAL_RESTRICTED_MODE
            PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
            if (mono) free(mono);
            return wt_result;
        }
        wt_bank_ptr = &wt_bank;
    }

    /* Optional processing chain mask (analysis/load output -> synthesis input) */
    SpectralProcessReport proc_report = {0};
    SpectralError proc_err = spectral_process_chain_apply(
        &sa, sample_rate, opts->processing_mask, &proc_report);
    if (proc_err != SPECTRAL_OK) {
        printf("Error: Processing chain failed (%s)\n", spectral_strerror(proc_err));
        free(sa.segs);
#if HAS_PERF && !SPECTRAL_RESTRICTED_MODE
    PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        if (mono) free(mono);
        return PIPELINE_ERR_ANALYSIS;
    }

    {
        char requested_buf[192] = {0};
        char pending_buf[192] = {0};
        spectral_process_mask_to_string(proc_report.requested, requested_buf, sizeof(requested_buf));
        spectral_process_mask_to_string(proc_report.pending, pending_buf, sizeof(pending_buf));
        if (proc_report.pending) {
            printf("Processing mask: %s (pending: %s)\n", requested_buf, pending_buf);
        } else {
            printf("Processing mask: %s\n", requested_buf);
        }
    }
    
    /* Allocate output buffer */
    double out_len_f = (double)n_samples * (double)opts->stretch;
    if (out_len_f <= 0.0 || out_len_f > (double)SIZE_MAX) {
        free(sa.segs);
#if HAS_PERF && !SPECTRAL_RESTRICTED_MODE
    PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        if (mono) free(mono);
        return PIPELINE_ERR_INPUT;
    }
    size_t out_len = (size_t)out_len_f;
    if (out_len == 0 || out_len > SIZE_MAX / sizeof(float)) {
        free(sa.segs);
#if HAS_PERF && !SPECTRAL_RESTRICTED_MODE
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        if (mono) free(mono);
        return PIPELINE_ERR_INPUT;
    }
    float* out_buf = calloc(out_len, sizeof(float));
    if (!out_buf) {
        free(sa.segs);
#if HAS_PERF && !SPECTRAL_RESTRICTED_MODE
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
        if (mono) free(mono);
        return PIPELINE_ERR_MEMORY;
    }
    
#if HAS_PERF
    perf_track_alloc(out_len * sizeof(float));
    perf_track_alloc(sa.count * sizeof(Segment));
#endif
    
    /* Run synthesis */
    SpectralError synth_err = run_synthesis(opts, sa, out_buf, out_len, wt_bank_ptr, &t.t_synth);
    if (synth_err != SPECTRAL_OK) {
        printf("Error: Synthesis failed (%s)\n", spectral_strerror(synth_err));
    #if HAS_PERF
        perf_track_free(sa.count * sizeof(Segment));
        perf_track_free(out_len * sizeof(float));
    #if !SPECTRAL_RESTRICTED_MODE
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
    #endif
    #endif
        free(sa.segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_SYNTHESIS;
    }
    
    /* Normalize */
    double norm_start = omp_get_wtime();
    spectral_normalize_float(out_buf, out_len, SPECTRAL_NORMALIZE_HEADROOM);
    t.t_norm = omp_get_wtime() - norm_start;
    
        /* Write output */
        if (ensure_output_dir_exists() != PIPELINE_OK) {
        printf("Error: Failed to create output directory (%s)\n",
               g_paths.resolved ? g_paths.output_dir : OUTPUT_DIR_PRIMARY);
    #if HAS_PERF
        perf_track_free(sa.count * sizeof(Segment));
        perf_track_free(out_len * sizeof(float));
    #if !SPECTRAL_RESTRICTED_MODE
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
    #endif
    #endif
        free(sa.segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_OUTPUT;
        }

        SpectralError write_err = spectral_audio_write(g_paths.output_wav, out_buf, out_len, sample_rate, 1);
    if (write_err != SPECTRAL_OK) {
        printf("Error: Failed to write output file (%s)\n", spectral_strerror(write_err));
    #if HAS_PERF
        perf_track_free(sa.count * sizeof(Segment));
        perf_track_free(out_len * sizeof(float));
    #if !SPECTRAL_RESTRICTED_MODE
        PERF_FREE_INPUT_BYTES(input_alloc_bytes);
    #endif
    #endif
        free(sa.segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_OUTPUT;
    }
    printf("Wrote output: %s\n", g_paths.output_wav);
    
    /* Calculate timing results */
    t.t_total = t.t_fft + t.t_track + t.t_synth + t.t_norm;
    t.audio_dur = (double)n_samples / sample_rate;
    t.realtime_x = (t.t_total > 0) ? (t.audio_dur / t.t_total) : 0.0;
    
    /* Print timing */
    spectral_pipeline_print_timing(&t, sa.count);
    
#if HAS_PERF
    PerfMetrics perf_end = perf_snapshot(wall_start);
    perf_print(&perf_start, &perf_end, opts->n_threads);
    perf_track_free(sa.count * sizeof(Segment));
    perf_track_free(out_len * sizeof(float));
#if !SPECTRAL_RESTRICTED_MODE
    PERF_FREE_INPUT_BYTES(input_alloc_bytes);
#endif
#endif
    
    /* Cleanup */
    if (mono) free(mono);
    free(sa.segs);
    free(out_buf);
    
    if (timing) *timing = t;
    return PIPELINE_OK;
}

void spectral_pipeline_print_timing(const SpectralTimingResults* t, uint32_t segment_count) {
    if (!t) return;
    
#if SPECTRAL_RESTRICTED_MODE
    printf("\n--- Timing ---\n");
    printf("Synth: %.1fms Norm: %.1fms Total: %.1fms\n",
           t->t_synth*1000, t->t_norm*1000, t->t_total*1000);
#else
    printf("\n--- Timing ---\n");
    printf("FFT: %.1fms Track: %.1fms Synth: %.1fms Norm: %.1fms Total: %.1fms\n",
           t->t_fft*1000, t->t_track*1000, t->t_synth*1000, t->t_norm*1000, t->t_total*1000);
#endif
    printf("Audio: %.2fs Realtime: %.1fx Segs/sec: %.0fK\n",
            t->audio_dur, t->realtime_x,
            (t->t_total > 0) ? (segment_count / t->t_total / 1000) : 0.0);
}
