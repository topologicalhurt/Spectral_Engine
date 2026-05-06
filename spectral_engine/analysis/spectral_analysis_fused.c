/* spectral_analysis_fused.c
 *
 * Correctness-first fused parallel STFT + peak-tracking path for large inputs.
 *
 * The fused path processes explicit adjacent frame pairs. Each work item owns a
 * contiguous range of pair indices [pair_start, pair_end), computes the current
 * frame once, then computes the next frame for each pair and rotates the row and
 * phase buffers together. This preserves the same frame-pair contract as the
 * full-matrix path:
 *
 *   row       = magsq[pair]
 *   next_row  = magsq[pair + 1]
 *   phase_row = phase[pair]
 *   t_hop     = pair * hop
 *
 * Keeping this contract explicit is more important than shaving one duplicate
 * FFT at chunk boundaries. The latter can be reintroduced only after parity
 * tests prove identical segment output against spectral_analysis_run_full().
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
    SpectralWindowMetrics window_metrics = {0};
    SpectralFftResources res = {0};
    int n_threads = omp_get_max_threads();
    int actual_threads = n_threads;
    double fft_time_total = 0.0;
    float global_max_magsq = 0.0f;
    SpectralTracker* tracker = NULL;
    const size_t pair_chunk_size = 256u;

    (void)n_samples;

    if (actual_threads < 1) actual_threads = 1;
    if (n_frames < 2u) goto fail;
    if (!spectral_array_bytes((size_t)n_fft, sizeof(float), &n_fft_f32_bytes)) goto fail;

    window_func = spectral_aligned_alloc(n_fft_f32_bytes);
    if (!window_func) goto fail;
    spectral_window_generate(window_func, (size_t)n_fft, SPECTRAL_WINDOW_HANN);
    window_metrics = spectral_window_metrics(window_func, (size_t)n_fft);

    if (!spectral_fft_resources_alloc(&res, actual_threads, (size_t)n_fft, n_freqs)) goto fail;
    spectral_fft_resources_set_magsq_scales(&res,
        window_metrics.endpoint_bin_magsq_scale,
        window_metrics.positive_bin_magsq_scale);

    {
        /* Pass 1: Global maximum discovery is required for stable dB thresholding.
         * It must never be compiled out by a speed profile. */
        float pass1_max = 0.0f;
        int pass1_failed = 0;
        double p1_start = omp_get_wtime();

        omp_set_num_threads(actual_threads);
        #pragma omp parallel reduction(max:pass1_max)
        {
            int tid = omp_get_thread_num();
            int local_failed = 0;
            float* scratch_magsq = spectral_aligned_alloc(n_freqs * sizeof(float));
            if (!scratch_magsq) {
                local_failed = 1;
                #pragma omp atomic write
                pass1_failed = 1;
            }

            #pragma omp for schedule(static)
            for (size_t t = 0; t < n_frames; t++) {
                if (local_failed) continue;
                float frame_max = 0.0f;
                spectral_fft_single_frame(&res, tid, audio, hop, window_func, t,
                                          scratch_magsq, NULL, &frame_max);
                if (frame_max > pass1_max) pass1_max = frame_max;
            }
            free(scratch_magsq);
        }

        if (pass1_failed) goto fail;
        global_max_magsq = pass1_max;
        fft_time_total += (omp_get_wtime() - p1_start);

        SPECTRAL_LOG_INFO("Analysis fused max pass: max_magsq=%.9g time=%.3fms",
                          global_max_magsq,
                          (omp_get_wtime() - p1_start) * SPECTRAL_MILLIS_PER_SECOND_D);
    }

    tracker = spectral_tracker_create(actual_threads, n_freqs, sr, n_fft, hop, db_thresh, global_max_magsq);
    if (!tracker) goto fail;
    spectral_tracker_set_window_descriptor(tracker, spectral_window_descriptor(SPECTRAL_WINDOW_HANN));

    {
        const size_t pair_count = n_frames - 1u;
        const size_t total_chunks = (pair_count + pair_chunk_size - 1u) / pair_chunk_size;
        double parallel_start = omp_get_wtime();

        omp_set_num_threads(actual_threads);
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float* row_curr = spectral_aligned_alloc(n_freqs * sizeof(float));
            float* row_next = spectral_aligned_alloc(n_freqs * sizeof(float));
            float* phase_curr = spectral_aligned_alloc(n_freqs * sizeof(float));
            float* phase_next = spectral_aligned_alloc(n_freqs * sizeof(float));
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

            if (!row_curr || !row_next || !phase_curr || !phase_next) {
                spectral_tracker_set_failed(tracker);
            }

            #pragma omp barrier

            if (!spectral_tracker_has_failed(tracker)) {
                #pragma omp for schedule(dynamic, 1)
                for (size_t chunk_idx = 0; chunk_idx < total_chunks; chunk_idx++) {
                    size_t pair_start = chunk_idx * pair_chunk_size;
                    size_t pair_end = pair_start + pair_chunk_size;
                    double fft_start = 0.0;
                    float frame_max = 0.0f;

                    if (pair_end > pair_count) pair_end = pair_count;
                    if (pair_start >= pair_end || spectral_tracker_has_failed(tracker)) continue;

                    /* Prime the current row for this pair range. */
                    fft_start = omp_get_wtime();
                    spectral_fft_single_frame(&res, tid, audio, hop, window_func, pair_start,
                                              row_curr, phase_curr, &frame_max);
                    local_fft_time += (omp_get_wtime() - fft_start);

                    for (size_t pair = pair_start;
                         pair < pair_end && !spectral_tracker_has_failed(tracker);
                         pair++) {
                        float t_hop = (float)pair * (float)hop;
                        SpectralFrameContext fctx;
                        double track_start = 0.0;

                        fft_start = omp_get_wtime();
                        spectral_fft_single_frame(&res, tid, audio, hop, window_func, pair + 1u,
                                                  row_next, phase_next, &frame_max);
                        local_fft_time += (omp_get_wtime() - fft_start);

                        fctx.row = row_curr;
                        fctx.next_row = row_next;
                        fctx.phase_row = phase_curr;
                        fctx.next_phase_row = phase_next;
                        fctx.t_hop = t_hop;
                        fctx.threshsq = threshsq;
                        fctx.can_start_new = 1;

                        track_start = omp_get_wtime();
                        local_pairs++;
                        /* spectral_tracker_run_fused_frame follows tracker helper polarity */
                        if (!spectral_tracker_run_fused_frame(
                            tracker, tid, candidate_batch, &candidate_batch_count,
                            &fctx,
                            &local_candidates, &local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                            , &local_validate_time, &local_emit_time, &local_emit_alloc_time,
                            &local_emit_interp_time, &local_emit_amp_time
#endif
                        )) {
                            spectral_tracker_set_failed(tracker);
                            break;
                        }
                        local_track_time += (omp_get_wtime() - track_start);

                        {
                            float* tmp = row_curr;
                            row_curr = row_next;
                            row_next = tmp;
                        }
                        {
                            float* tmp = phase_curr;
                            phase_curr = phase_next;
                            phase_next = tmp;
                        }
                    }
                }
            }

            #pragma omp atomic update
            fft_time_total += local_fft_time;

#if SPECTRAL_TRACK_DEBUG_TIMING
            {
                double local_scan_time = (local_track_time - local_validate_time - local_emit_time > 0.0)
                    ? (local_track_time - local_validate_time - local_emit_time) : 0.0;
                spectral_tracker_accumulate_stats(tracker,
                    local_pairs, local_candidates, local_segments, local_track_time,
                    local_scan_time, local_validate_time, local_emit_time,
                    local_emit_alloc_time, local_emit_interp_time, local_emit_amp_time);
            }
#else
            spectral_tracker_accumulate_stats(tracker,
                local_pairs, local_candidates, local_segments, local_track_time);
#endif

            free(row_curr);
            free(row_next);
            free(phase_curr);
            free(phase_next);
        }

        {
            double parallel_wall_time = omp_get_wtime() - parallel_start;
            double process_time = spectral_tracker_get_process_time(tracker);
            double measured_total = fft_time_total + process_time;
            double track_wall_time = 0.0;

            if (measured_total > 0.0) {
                *t_fft = parallel_wall_time * (fft_time_total / measured_total);
                track_wall_time = parallel_wall_time * (process_time / measured_total);
            } else {
                *t_fft = parallel_wall_time / 2.0;
                track_wall_time = parallel_wall_time / 2.0;
            }

            spectral_fft_resources_free(&res);
            free(window_func);
            res = (SpectralFftResources){0};
            window_func = NULL;

            {
                SegmentArray final_res = spectral_tracker_finalize(tracker, t_track, track_wall_time);
                spectral_tracker_destroy(tracker);
                return final_res;
            }
        }
    }

fail:
    if (tracker) {
        (void)spectral_tracker_finalize(tracker, NULL, 0.0);
        spectral_tracker_destroy(tracker);
    }
    spectral_fft_resources_free(&res);
    free(window_func);
    return spectral_analysis_return_empty(t_fft, t_track);
}
