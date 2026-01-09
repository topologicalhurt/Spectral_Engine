/* main.c - Spectral audio processor entry point */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sndfile.h>

#include "spectral_common.h"
#include "spectral_analysis.h"
#include "spectral_synth.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_VDSP 1
#else
#define USE_VDSP 0
#endif

static void print_usage(void) {
    printf("Usage: ./spectral input.wav [timbre 0-7] [stretch] [pitch] [n_fft] [hop] [db_thresh] [n_threads] [backend] [start_sec] [end_sec]\n");
    printf("Defaults: timbre=0, stretch=1.0, pitch=0.0, n_fft=%d, hop=%d, db_thresh=%.1f, n_threads=auto, backend=auto\n",
           DEFAULT_N_FFT, DEFAULT_HOP, DEFAULT_DB_THRESH);
    printf("Backend: 0=auto, 1=cpu, 2=metal, 3=cuda, 4=export (save segments.bin for CUDA)\n");
    printf("Time window: start_sec and end_sec in seconds (default: full file, use -1 for end)\n");
    printf("\nTimbres: 0=sine, 1=saw, 2=square, 3=triangle, 4=asin, 5=parabola, 6=quantized, 7=pwm\n");
}

int main(int argc, char** argv) {
    if (argc < 2) { 
        print_usage();
        return 1;
    }
    
    const char* path = argv[1];
    int timbre = (argc > 2) ? atoi(argv[2]) : 0;
    float stretch = (argc > 3) ? atof(argv[3]) : 1.0f;
    float pitch = (argc > 4) ? atof(argv[4]) : 0.0f;
    int n_fft = (argc > 5) ? atoi(argv[5]) : DEFAULT_N_FFT;
    int hop = (argc > 6) ? atoi(argv[6]) : DEFAULT_HOP;
    float db_thresh = (argc > 7) ? atof(argv[7]) : DEFAULT_DB_THRESH;
    int n_threads = (argc > 8) ? atoi(argv[8]) : omp_get_max_threads();
    int backend = (argc > 9) ? atoi(argv[9]) : BACKEND_AUTO;
    float start_sec = (argc > 10) ? atof(argv[10]) : 0.0f;
    float end_sec = (argc > 11) ? atof(argv[11]) : -1.0f;
    
    if ((n_fft & (n_fft - 1)) != 0 || n_fft < 64) {
        printf("Error: n_fft must be a power of 2 >= 64 (got %d)\n", n_fft);
        return 1;
    }
    
    if (n_threads < 1) n_threads = 1;
    omp_set_num_threads(n_threads);
    
    double wall_start = omp_get_wtime();
    perf_reset_tracking();
    PerfMetrics perf_start = perf_snapshot(wall_start);

    SF_INFO sfinfo;
    SNDFILE* infile = sf_open(path, SFM_READ, &sfinfo);
    if (!infile) { 
        printf("Error opening file: %s\n", path); 
        return 1; 
    }
    
    size_t audio_size = sfinfo.frames * sfinfo.channels * sizeof(float);
    float* audio = malloc(audio_size);
    perf_track_alloc(audio_size);
    sf_readf_float(infile, audio, sfinfo.frames);
    sf_close(infile);
    
    size_t mono_size = sfinfo.frames * sizeof(float);
    float* mono = malloc(mono_size);
    perf_track_alloc(mono_size);
    for(int i = 0; i < sfinfo.frames; i++) {
        mono[i] = audio[i * sfinfo.channels];
    }
    free(audio);
    perf_track_free(audio_size);
    
    size_t start_frame = (size_t)(start_sec * sfinfo.samplerate);
    size_t end_frame = (end_sec < 0) ? (size_t)sfinfo.frames : (size_t)(end_sec * sfinfo.samplerate);
    
    if (start_frame >= (size_t)sfinfo.frames) start_frame = 0;
    if (end_frame > (size_t)sfinfo.frames) end_frame = sfinfo.frames;
    if (end_frame <= start_frame) {
        printf("Error: end_sec must be greater than start_sec\n");
        free(mono);
        return 1;
    }
    
    size_t n_samples = end_frame - start_frame;
    float* windowed_audio = mono + start_frame;
    
    if (start_sec > 0 || end_sec > 0) {
        printf("Time window: %.3f - %.3f sec (%zu frames)\n", 
               start_sec, (end_sec < 0) ? (float)sfinfo.frames/sfinfo.samplerate : end_sec, n_samples);
    }

    printf("Analyzing %zu frames (n_fft=%d, hop=%d, db_thresh=%.1f, threads=%d)...\n", 
           n_samples, n_fft, hop, db_thresh, n_threads);
    
    double t_fft, t_track, t_synth, t_norm;
    
    SegmentArray sa = analyze_audio(windowed_audio, n_samples, sfinfo.samplerate, 
                                    n_fft, hop, db_thresh, &t_fft, &t_track);
    printf("Found %zu segments.\n", sa.count);

    if (backend == BACKEND_EXPORT) {
        const char* seg_path = "segments.bin";
        printf("Exporting segments to %s...\n", seg_path);
        if (segments_export(seg_path, &sa, sfinfo.samplerate, stretch, pitch) == 0) {
            printf("Saved %zu segments (%.1f MB)\n", sa.count, 
                   (sa.count * sizeof(Segment)) / (1024.0 * 1024.0));
            printf("To synthesize with CUDA: ./spectral_cuda %s output.wav\n", seg_path);
        } else {
            printf("Error saving segments\n");
        }
        free(sa.segs);
        free(mono);
        printf("\n--- Timing ---\n");
        printf("FFT:       %.3f ms\n", t_fft * 1000.0);
        printf("Tracking:  %.3f ms\n", t_track * 1000.0);
        printf("Total:     %.3f ms\n", (t_fft + t_track) * 1000.0);
        return 0;
    }

    size_t out_len = (size_t)(n_samples * stretch);
    size_t out_buf_size = out_len * sizeof(float);
    float* out_buf = calloc(out_len, sizeof(float));
    perf_track_alloc(out_buf_size);
    perf_track_alloc(sa.count * sizeof(Segment));  // Track segment array
    
#if HAS_METAL
    if (backend != BACKEND_CPU && backend != BACKEND_CUDA) {
        metal_init();
    }
    
    float density = (float)sa.count / (float)out_len;
    int use_metal = (backend == BACKEND_METAL) || 
                    (backend == BACKEND_AUTO && metal_available() && timbre == 0);
    
    if (use_metal && metal_available()) {
        printf("Rendering with Metal GPU (tile-parallel, %zu segs, density=%.1f)...\n", sa.count, density);
        synth_metal(sa, out_buf, out_len, stretch, pitch, &t_synth);
        if (t_synth < 0) {
            printf("Metal fallback to CPU...\n");
            synth_cpu(sa, out_buf, out_len, stretch, pitch, timbre, n_threads, &t_synth);
        }
    } else if (backend == BACKEND_CUDA) {
        printf("CUDA not available on macOS\n");
        printf("Rendering with %d CPU threads...\n", n_threads);
        synth_cpu(sa, out_buf, out_len, stretch, pitch, timbre, n_threads, &t_synth);
    } else {
        if (backend == BACKEND_METAL && !metal_available()) {
            printf("Warning: Metal requested but not available\n");
        }
        printf("Rendering with %d CPU threads...\n", n_threads);
        synth_cpu(sa, out_buf, out_len, stretch, pitch, timbre, n_threads, &t_synth);
    }
#elif HAS_CUDA
    if (backend == BACKEND_CUDA || backend == BACKEND_AUTO) {
        cuda_init();
        if (cuda_available() && timbre == 0) {
            printf("Rendering with CUDA GPU (%zu segs)...\n", sa.count);
            synth_cuda(sa, out_buf, out_len, stretch, pitch, &t_synth);
        } else {
            if (backend == BACKEND_CUDA) {
                printf("Warning: CUDA requested but not available\n");
            }
            printf("Rendering with %d CPU threads...\n", n_threads);
            synth_cpu(sa, out_buf, out_len, stretch, pitch, timbre, n_threads, &t_synth);
        }
    } else {
        printf("Rendering with %d CPU threads...\n", n_threads);
        synth_cpu(sa, out_buf, out_len, stretch, pitch, timbre, n_threads, &t_synth);
    }
#else
    if (backend == BACKEND_METAL) {
        printf("Warning: Metal requested but not compiled in\n");
    }
    if (backend == BACKEND_CUDA) {
        printf("Warning: CUDA requested but not compiled in\n");
    }
    printf("Rendering with %d threads...\n", n_threads);
    synth_cpu(sa, out_buf, out_len, stretch, pitch, timbre, n_threads, &t_synth);
#endif

    double norm_start = omp_get_wtime();
#if USE_VDSP
    float max_amp = 0.0f;
    vDSP_maxmgv(out_buf, 1, &max_amp, out_len);
    if (max_amp > 0) {
        float scaler = 0.95f / max_amp;
        vDSP_vsmul(out_buf, 1, &scaler, out_buf, 1, out_len);
    }
#else
    float max_amp = 0.0f;
    #pragma omp parallel for reduction(max:max_amp)
    for(size_t i = 0; i < out_len; i++) {
        float a = fabsf(out_buf[i]);
        if (a > max_amp) max_amp = a;
    }
    if (max_amp > 0) {
        float scaler = 0.95f / max_amp;
        #pragma omp parallel for
        for(size_t i = 0; i < out_len; i++) out_buf[i] *= scaler;
    }
#endif
    t_norm = omp_get_wtime() - norm_start;

    SF_INFO out_info = {0};
    out_info.samplerate = sfinfo.samplerate;
    out_info.frames = out_len;
    out_info.channels = 1;
    out_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* outfile = sf_open("out_c.wav", SFM_WRITE, &out_info);
    sf_writef_float(outfile, out_buf, out_len);
    sf_close(outfile);

    PerfMetrics perf_end = perf_snapshot(wall_start);
    
    printf("\n--- Timing ---\n");
    printf("FFT:       %.3f ms\n", t_fft * 1000.0);
    printf("Tracking:  %.3f ms\n", t_track * 1000.0);
    printf("Synthesis: %.3f ms\n", t_synth * 1000.0);
    printf("Normalize: %.3f ms\n", t_norm * 1000.0);
    printf("Total:     %.3f ms\n", (t_fft + t_track + t_synth + t_norm) * 1000.0);
    
    double total_time = t_fft + t_track + t_synth + t_norm;
    double audio_duration = (double)n_samples / sfinfo.samplerate;
    double realtime_factor = audio_duration / total_time;
    printf("\n--- Throughput ---\n");
    printf("Audio:     %.2f sec\n", audio_duration);
    printf("Realtime:  %.1fx\n", realtime_factor);
    printf("Segs/sec:  %.0f K\n", (sa.count / total_time) / 1000.0);
    
    perf_print(&perf_start, &perf_end, n_threads);

    free(mono);
    free(sa.segs);
    free(out_buf);
    printf("\nDone.\n");
    return 0;
}
