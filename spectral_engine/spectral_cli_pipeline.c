/* spectral_pipeline.c - Main Processing Pipeline Implementation
 * 
 * Orchestrates analysis, synthesis, and output operations.
 */
#include "spectral_cli_pipeline.h"
#include "spectral_synth.h"
#include "spectral_segment_parser.h"
#include "spectral_io.h"
#include "spectral_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
static void omp_set_num_threads_wrapper(int n) { omp_set_num_threads(n); }
#else
static void omp_set_num_threads_wrapper(int n) { (void)n; }
#endif

/* Analysis is only available when NOT in restricted mode */
#if !defined(SPECTRAL_RESTRICTED_MODE) || !SPECTRAL_RESTRICTED_MODE
#include "spectral_analysis.h"
#define HAS_ANALYSIS 1
#else
#define HAS_ANALYSIS 0
#endif

/* Perf tracking (includes spectral_get_time_sec) */
#if !SPECTRAL_NO_PERF
#include "spectral_perf.h"
#define HAS_PERF 1
#else
#define HAS_PERF 0
#endif

#define NORMALIZE_HEADROOM 0.95f

/* Load wavetable from file */
static PipelineResult load_wavetable(const char* path, SpectralWavetableBank* bank, 
                                     SpectralTimbre timbre) {
    spectral_wavetable_init(bank);
    
    size_t len = strlen(path);
    WavetableResult wt_result;
    
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
        printf("Error: Failed to load wavetable (code %d)\n", wt_result);
        return PIPELINE_ERR_WAVETABLE;
    }
    
    printf("Wavetable loaded (%d samples)\n", SPECTRAL_WAVETABLE_SIZE);
    return PIPELINE_OK;
}

/* Run synthesis with appropriate backend */
static void run_synthesis(const SpectralCliOptions* opts, SegmentArray sa,
                          float* out_buf, size_t out_len, 
                          SpectralWavetableBank* wt_bank, double* t_synth) {
    
#if SPECTRAL_RESTRICTED_MODE
    printf("Rendering with %d CPU threads%s (restricted mode)...\n", 
           opts->n_threads, wt_bank ? " (wavetable)" : "");
    if (wt_bank) {
        synth_cpu_wavetable(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                           wt_bank, opts->timbre, opts->n_threads, t_synth);
    } else {
        synth_cpu(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                 opts->timbre, opts->n_threads, t_synth);
    }
    
#elif HAS_METAL
    if (opts->backend != BACKEND_CPU && opts->backend != BACKEND_CUDA) metal_init();
    int metal_supports_timbre = (opts->timbre <= BACKEND_METAL_TIMBRE_MAX);
    int use_metal = (opts->backend == BACKEND_METAL) || 
                    (opts->backend == BACKEND_AUTO && metal_available() && metal_supports_timbre);
    
    if (use_metal && metal_available()) {
        printf("Rendering with Metal GPU (%u segs, timbre=%s)...\n", sa.count, timbre_name(opts->timbre));
        synth_metal(sa, out_buf, out_len, opts->stretch, opts->pitch, opts->timbre, t_synth);
        if (*t_synth < 0) {
            printf("Metal fallback to CPU...\n");
            if (wt_bank) {
                synth_cpu_wavetable(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                                   wt_bank, opts->timbre, opts->n_threads, t_synth);
            } else {
                synth_cpu(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                         opts->timbre, opts->n_threads, t_synth);
            }
        }
    } else {
        if (opts->backend == BACKEND_METAL && !metal_available()) 
            printf("Warning: Metal unavailable\n");
        else if (opts->backend == BACKEND_METAL && !metal_supports_timbre)
            printf("Warning: Metal doesn't support timbre %s, using CPU\n", timbre_name(opts->timbre));
        printf("Rendering with %d CPU threads%s...\n", opts->n_threads, wt_bank ? " (wavetable)" : "");
        if (wt_bank) {
            synth_cpu_wavetable(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                               wt_bank, opts->timbre, opts->n_threads, t_synth);
        } else {
            synth_cpu(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                     opts->timbre, opts->n_threads, t_synth);
        }
    }
    
#elif HAS_CUDA
    if (opts->backend == BACKEND_CUDA || opts->backend == BACKEND_AUTO) {
        cuda_init();
        if (cuda_available() && opts->timbre == TIMBRE_SINE && !wt_bank) {
            printf("Rendering with CUDA GPU (%u segs)...\n", sa.count);
            synth_cuda(sa, out_buf, out_len, opts->stretch, opts->pitch, t_synth);
        } else {
            printf("Rendering with %d CPU threads%s...\n", opts->n_threads, wt_bank ? " (wavetable)" : "");
            if (wt_bank) {
                synth_cpu_wavetable(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                                   wt_bank, opts->timbre, opts->n_threads, t_synth);
            } else {
                synth_cpu(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                         opts->timbre, opts->n_threads, t_synth);
            }
        }
    } else {
        printf("Rendering with %d CPU threads%s...\n", opts->n_threads, wt_bank ? " (wavetable)" : "");
        if (wt_bank) {
            synth_cpu_wavetable(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                               wt_bank, opts->timbre, opts->n_threads, t_synth);
        } else {
            synth_cpu(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                     opts->timbre, opts->n_threads, t_synth);
        }
    }
    
#else
    printf("Rendering with %d threads%s...\n", opts->n_threads, wt_bank ? " (wavetable)" : "");
    if (wt_bank) {
        synth_cpu_wavetable(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                           wt_bank, opts->timbre, opts->n_threads, t_synth);
    } else {
        synth_cpu(sa, out_buf, out_len, opts->stretch, opts->pitch, 
                 opts->timbre, opts->n_threads, t_synth);
    }
#endif
}

PipelineResult spectral_pipeline_run(const SpectralCliOptions* opts, 
                                     SpectralTimingResults* timing) {
    if (!opts || !opts->valid) return PIPELINE_ERR_INPUT;
    
    SpectralTimingResults t = {0};
    double wall_start = spectral_get_time_sec();
    
    omp_set_num_threads_wrapper(opts->n_threads);
    
#if HAS_PERF
    perf_reset_tracking();
    PerfMetrics perf_start = perf_snapshot(wall_start);
#endif
    
    SegmentArray sa = {0};
    int sample_rate = 0;
    size_t n_samples = 0;
    float* mono = NULL;
    
#if SPECTRAL_RESTRICTED_MODE
    /* Restricted mode: load pre-analyzed segments */
    printf("Loading segments from %s...\n", opts->input_path);
    int sr_loaded;
    float stretch_loaded, pitch_loaded;
    if (segments_load(opts->input_path, &sa, &sr_loaded, &stretch_loaded, &pitch_loaded) != 0) {
        printf("Error: Cannot load segments from %s\n", opts->input_path);
        return PIPELINE_ERR_INPUT;
    }
    sample_rate = sr_loaded;
    printf("Loaded %u segments (sr=%d)\n", sa.count, sample_rate);
    
    /* Calculate output length */
    size_t max_end = 0;
    for (size_t i = 0; i < sa.count; i++) {
        size_t seg_end = (size_t)(sa.segs[i].start + sa.segs[i].length);
        if (seg_end > max_end) max_end = seg_end;
    }
    n_samples = max_end;
    
#else
    /* Desktop/emulator mode: read and analyze audio */
    SpectralAudioInfo audio_info;
    if (spectral_audio_read(opts->input_path, &audio_info, &mono) != SPECTRAL_OK) {
        printf("Error: Cannot open %s\n", opts->input_path);
        return PIPELINE_ERR_INPUT;
    }
    sample_rate = audio_info.sample_rate;
    
#if HAS_PERF
    perf_track_alloc(audio_info.frames * sizeof(float));
#endif
    
    float* windowed_audio;
    if (spectral_audio_window(mono, audio_info.frames, opts->start_sec, opts->end_sec,
                              sample_rate, &windowed_audio, &n_samples) != SPECTRAL_OK) {
        printf("Error: invalid time window\n");
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
    
    /* Handle export mode */
    if (opts->backend == BACKEND_EXPORT) {
        printf("Exporting to segments.bin...\n");
        if (segments_save("segments.bin", &sa, sample_rate, opts->stretch, opts->pitch) == 0) {
            printf("Saved %u segments (%.1f MB)\n", sa.count, 
                   sa.count * sizeof(Segment) / (1024.0 * 1024.0));
        }
        free(sa.segs);
        free(mono);
        printf("FFT: %.3fms Track: %.3fms\n", t.t_fft*1000, t.t_track*1000);
        if (timing) *timing = t;
        return PIPELINE_OK;
    }
#endif
    
    /* Load wavetable if requested */
    SpectralWavetableBank wt_bank;
    SpectralWavetableBank* wt_bank_ptr = NULL;
    if (opts->use_wavetable && opts->wavetable_path) {
        PipelineResult wt_result = load_wavetable(opts->wavetable_path, &wt_bank, opts->timbre);
        if (wt_result != PIPELINE_OK) {
            free(sa.segs);
            if (mono) free(mono);
            return wt_result;
        }
        wt_bank_ptr = &wt_bank;
    }
    
    /* Allocate output buffer */
    size_t out_len = (size_t)(n_samples * opts->stretch);
    float* out_buf = calloc(out_len, sizeof(float));
    if (!out_buf) {
        free(sa.segs);
        if (mono) free(mono);
        return PIPELINE_ERR_MEMORY;
    }
    
#if HAS_PERF
    perf_track_alloc(out_len * sizeof(float));
    perf_track_alloc(sa.count * sizeof(Segment));
#endif
    
    /* Run synthesis */
    run_synthesis(opts, sa, out_buf, out_len, wt_bank_ptr, &t.t_synth);
    
    /* Normalize */
    double norm_start = spectral_get_time_sec();
    spectral_normalize_float(out_buf, out_len, NORMALIZE_HEADROOM);
    t.t_norm = spectral_get_time_sec() - norm_start;
    
    /* Write output */
    if (spectral_audio_write("out_c.wav", out_buf, out_len, sample_rate, 1) != SPECTRAL_OK) {
        printf("Error: Failed to write output file\n");
        free(sa.segs);
        free(out_buf);
        if (mono) free(mono);
        return PIPELINE_ERR_OUTPUT;
    }
    
    /* Calculate timing results */
    t.t_total = t.t_fft + t.t_track + t.t_synth + t.t_norm;
    t.audio_dur = (double)n_samples / sample_rate;
    t.realtime_x = t.audio_dur / t.t_total;
    
    /* Print timing */
    spectral_pipeline_print_timing(&t, sa.count);
    
#if HAS_PERF
    PerfMetrics perf_end = perf_snapshot(wall_start);
    perf_print(&perf_start, &perf_end, opts->n_threads);
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
           t->audio_dur, t->realtime_x, segment_count / t->t_total / 1000);
}
