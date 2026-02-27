/* spectral_analysis_fused.c
 *
 * Fully fused parallel STFT + peak-tracking pipeline for large inputs.
 * Each thread computes FFTs and tracks peaks in the same loop, operating
 * on a per-thread L1-sized sliding window (3 rows) to bypass DRAM.
 */
#include "spectral_analysis_internal.h"
#include "spectral_peak_track.h"

SegmentArray spectral_analysis_run_fused(const float* audio, size_t n_samples,
                                         int sr, int n_fft, int hop, float db_thresh,
                                         size_t n_frames, size_t n_freqs,
                                         double* t_fft, double* t_track)
{
    size_t n_fft_f32_bytes = 0;
    float* window_func = NULL;
    SpectralFftResources res = {0};
    int n_threads = omp_get_max_threads();
    double fft_time_total = 0.0;
    float global_max_magsq = 0.0f;
    SpectralTracker* tracker = NULL;

    (void)n_samples;

    if (!spectral_array_bytes((size_t)n_fft, sizeof(float), &n_fft_f32_bytes)) goto fail;

    window_func = spectral_aligned_alloc(n_fft_f32_bytes);
    if (!window_func) goto fail;
    spectral_window_hann(window_func, n_fft);

    /* Use 100% of all available threads for simultaneous FFT+Tracking */
    int actual_threads = n_threads;
    if (actual_threads < 1) actual_threads = 1;

    if (!spectral_fft_resources_alloc(&res, actual_threads, (size_t)n_fft, n_freqs)) goto fail;

#if !SPECTRAL_CUSTOM_FAST_MATH_MODE
    {
        /* Pass 1: Global Maximum Discovery (Strict Reproducibility) */
        float pass1_max = 0.0f;
        double p1_start = omp_get_wtime();
        
        omp_set_num_threads(actual_threads);
        #pragma omp parallel reduction(max:pass1_max)
        {
            int tid = omp_get_thread_num();
            float* scratch_magsq = spectral_aligned_alloc(n_freqs * sizeof(float));
            if (scratch_magsq) {
                #pragma omp for schedule(static)
                for (size_t t = 0; t < n_frames; t++) {
                    float frame_max = 0.0f;
                    spectral_fft_single_frame(&res, tid, audio, hop, window_func, t,
                                              scratch_magsq, NULL, &frame_max);
                    if (frame_max > pass1_max) pass1_max = frame_max;
                }
                free(scratch_magsq);
            }
        }
        global_max_magsq = pass1_max;
        fft_time_total += (omp_get_wtime() - p1_start);
        
        SPECTRAL_LOG_INFO("Analysis Pass 1 Max Magsq: %.3f (time=%.3fms)",
                          global_max_magsq,
                          (omp_get_wtime() - p1_start) * SPECTRAL_MILLIS_PER_SECOND_D);
    }
#endif

    tracker = spectral_tracker_create(actual_threads, n_freqs, sr, n_fft, hop, db_thresh, global_max_magsq);
    if (!tracker) goto fail;

    double parallel_start = omp_get_wtime();
    omp_set_num_threads(actual_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        /* Each thread gets its own L1-sized sliding window (3 rows) to completely bypass DRAM */
        float* row_prev = spectral_aligned_alloc(n_freqs * sizeof(float));
        float* row_curr = spectral_aligned_alloc(n_freqs * sizeof(float));
        float* row_next = spectral_aligned_alloc(n_freqs * sizeof(float));
        float* phase_curr = spectral_aligned_alloc(n_freqs * sizeof(float));
        
        if (!row_prev || !row_curr || !row_next || !phase_curr) {
            spectral_tracker_set_failed(tracker);
        }

        uint32_t candidate_batch[SPECTRAL_TRACK_CANDIDATE_BATCH];
        size_t candidate_batch_count = 0;
        uint64_t local_pairs = 0;
        uint64_t local_candidates = 0;
        uint64_t local_segments = 0;
        float threshsq = spectral_tracker_get_threshsq(tracker);
        double local_fft_time = 0.0;
        double local_track_time = 0.0;
#if SPECTRAL_TRACK_DEBUG_TIMING
        double local_validate_time = 0.0;
        double local_emit_time = 0.0;
        double local_emit_alloc_time = 0.0;
        double local_emit_interp_time = 0.0;
        double local_emit_amp_time = 0.0;
#endif

        #pragma omp barrier

        if (!spectral_tracker_has_failed(tracker)) {
            size_t total_chunks = (n_frames + 255) / 256;
            #pragma omp for schedule(dynamic, 1)
            for (size_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
                if (spectral_tracker_has_failed(tracker)) continue;
                
                size_t start_frame = chunk_idx * 256;
                size_t end_frame = start_frame + 256;
                if (end_frame > n_frames) end_frame = n_frames;
                
                /* Process overlap row if not the absolute last chunk */
                size_t fft_end = (end_frame == n_frames) ? end_frame : end_frame + 1;
                
                /* STFT Prime: Frame 0 */
                double fft_start = omp_get_wtime();
                float max0 = 0.0f;
                spectral_fft_single_frame(&res, tid, audio, hop, window_func, start_frame,
                                          row_prev, phase_curr, &max0);
                local_fft_time += (omp_get_wtime() - fft_start);
                
                if (fft_end > start_frame + 1) {
                    /* STFT Prime: Frame 1 */
                    fft_start = omp_get_wtime();
                    float max1 = 0.0f;
                    spectral_fft_single_frame(&res, tid, audio, hop, window_func, start_frame + 1,
                                              row_curr, phase_curr, &max1);
                    local_fft_time += (omp_get_wtime() - fft_start);
                    
                    /* Main sliding pipeline */
                    for (size_t t = start_frame + 1; t < fft_end && !spectral_tracker_has_failed(tracker); t++) {
                        float frame_max = 0.0f;
                        float* target_row = row_next;
                        float t_hop = (t - 1) * (float)hop;
                        int can_start_new = (t - 1 < end_frame);
                        
                        if (t + 1 < n_frames) {
                            fft_start = omp_get_wtime();
                            spectral_fft_single_frame(&res, tid, audio, hop, window_func, t + 1,
                                                      target_row, phase_curr, &frame_max);
                            local_fft_time += (omp_get_wtime() - fft_start);
                        } else {
                            /* Pad zeros for absolute last frame */
                            memset(target_row, 0, n_freqs * sizeof(float));
                        }

                        double track_start = omp_get_wtime();
                        
                        local_pairs++;
                        SpectralFrameContext fctx = {
                            .row = row_curr,
                            .next_row = target_row,
                            .phase_row = phase_curr,
                            .t_hop = t_hop,
                            .threshsq = threshsq,
                            .can_start_new = can_start_new
                        };
                        if (spectral_tracker_run_fused_frame(
                            tracker, tid, candidate_batch, &candidate_batch_count,
                            &fctx,
                            &local_candidates, &local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                            , &local_validate_time, &local_emit_time, &local_emit_alloc_time,
                            &local_emit_interp_time, &local_emit_amp_time
#endif
                        )) {
                            spectral_tracker_set_failed(tracker);
                        }

                        local_track_time += (omp_get_wtime() - track_start);
                        max1 = frame_max;

                        /* Pointer rotation (zero-copy swap) */
                        float* temp = row_prev;
                        row_prev = row_curr;
                        row_curr = row_next;
                        row_next = temp;
                    }
                    
                }
            }
        }

        #pragma omp atomic update
        fft_time_total += local_fft_time;

#if SPECTRAL_TRACK_DEBUG_TIMING
        double local_scan_time = (local_track_time - local_validate_time - local_emit_time > 0.0)
            ? (local_track_time - local_validate_time - local_emit_time) : 0.0;
#endif
        spectral_tracker_accumulate_stats(tracker,
            local_pairs, local_candidates, local_segments, local_track_time
#if SPECTRAL_TRACK_DEBUG_TIMING
            , local_scan_time, local_validate_time, local_emit_time,
            local_emit_alloc_time, local_emit_interp_time, local_emit_amp_time
#endif
        );

        free(row_prev);
        free(row_curr);
        free(row_next);
        free(phase_curr);
    }
    
    double parallel_wall_time = omp_get_wtime() - parallel_start;
    double process_time = spectral_tracker_get_process_time(tracker);
    double measured_total = fft_time_total + process_time;
    if (measured_total > 0.0) {
        *t_fft = parallel_wall_time * (fft_time_total / measured_total);
        parallel_wall_time = parallel_wall_time * (process_time / measured_total);
    } else {
        *t_fft = parallel_wall_time / 2.0;
        parallel_wall_time = parallel_wall_time / 2.0;
    }
    
    spectral_fft_resources_free(&res);
    free(window_func);

    SegmentArray final_res = spectral_tracker_finalize(tracker, t_track, parallel_wall_time);
    spectral_tracker_destroy(tracker);

    return final_res;

fail:
    if (tracker) {
        (void)spectral_tracker_finalize(tracker, NULL, 0.0);
        spectral_tracker_destroy(tracker);
    }
    spectral_fft_resources_free(&res);
    free(window_func);
    return spectral_analysis_return_empty(t_fft, t_track);
}
