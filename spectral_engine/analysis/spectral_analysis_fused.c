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
 *   frame_start_sample     = pair * hop
 *
 * Keeping this contract explicit is more important than shaving one duplicate
 * FFT at chunk boundaries. The latter can be reintroduced only after parity
 * tests prove identical segment output against spectral_analysis_run_full().
 */
#include "spectral_analysis_internal.h"
#include "spectral_peak_track.h"
#include "spectral_peak_track_internal.h"
#include <float.h>

typedef struct SpectralFusedScratchRows {
    float* row_curr;
    float* row_next;
    /* Complex spectrum rows: phase is taken via atan2f(im,re) at peaks in the
     * tracker, not densely here (phase-at-peaks). Two extra scratch rows/thread. */
    float* re_curr;
    float* im_curr;
    float* re_next;
    float* im_next;
} SpectralFusedScratchRows;

static int spectral_fused_scratch_rows_alloc(SpectralFusedScratchRows* rows, size_t row_bytes)
{
    if (!rows || row_bytes == 0u) return 0;
    *rows = (SpectralFusedScratchRows){0};

    rows->row_curr = (float*)spectral_aligned_alloc(row_bytes);
    rows->row_next = (float*)spectral_aligned_alloc(row_bytes);
    rows->re_curr = (float*)spectral_aligned_alloc(row_bytes);
    rows->im_curr = (float*)spectral_aligned_alloc(row_bytes);
    rows->re_next = (float*)spectral_aligned_alloc(row_bytes);
    rows->im_next = (float*)spectral_aligned_alloc(row_bytes);

    if (!rows->row_curr || !rows->row_next || !rows->re_curr || !rows->im_curr ||
        !rows->re_next || !rows->im_next) {
        free(rows->row_curr);
        free(rows->row_next);
        free(rows->re_curr);
        free(rows->im_curr);
        free(rows->re_next);
        free(rows->im_next);
        *rows = (SpectralFusedScratchRows){0};
        return 0;
    }

    return 1;
}

static void spectral_fused_scratch_rows_free(SpectralFusedScratchRows* rows)
{
    if (!rows) return;
    free(rows->row_curr);
    free(rows->row_next);
    free(rows->re_curr);
    free(rows->im_curr);
    free(rows->re_next);
    free(rows->im_next);
    *rows = (SpectralFusedScratchRows){0};
}

static void spectral_fused_scratch_rows_rotate(SpectralFusedScratchRows* rows)
{
    float* tmp = NULL;
    if (!rows) return;

    tmp = rows->row_curr;
    rows->row_curr = rows->row_next;
    rows->row_next = tmp;

    tmp = rows->re_curr;
    rows->re_curr = rows->re_next;
    rows->re_next = tmp;

    tmp = rows->im_curr;
    rows->im_curr = rows->im_next;
    rows->im_next = tmp;
}

SegmentArray spectral_analysis_run_fused(const float* audio, size_t n_samples,
                                         int sr, int n_fft, int hop, float db_thresh,
                                         size_t n_frames, size_t n_freqs,
                                         double* t_fft, double* t_track)
{
    size_t n_freqs_f32_bytes = 0;
    SpectralAnalysisWindowContext window_ctx = {0};
    SpectralFftResources res = {0};
    int n_threads = spectral_omp_effective_thread_count();
    int actual_threads = n_threads;
    double fft_time_total = 0.0;
    float global_max_magsq = 0.0f;
    SpectralTracker* tracker = NULL;
    const size_t pair_chunk_size = 256u;

    (void)n_samples;

    if (actual_threads < 1) actual_threads = 1;
    if (n_frames < 2u) goto fail;
    if (!spectral_array_bytes(n_freqs, sizeof(float), &n_freqs_f32_bytes)) goto fail;

    if (spectral_analysis_window_context_init(&window_ctx, (size_t)n_fft, SPECTRAL_WINDOW_HANN) != SPECTRAL_OK) goto fail;

    if (!spectral_fft_resources_alloc(&res, actual_threads, (size_t)n_fft, n_freqs)) goto fail;
    spectral_analysis_window_context_apply_magsq_scales(&window_ctx, &res);

    {
        /* Pass 1: Global maximum discovery is required for stable dB thresholding.
         * It must never be compiled out by a speed profile. */
        float pass1_max = 0.0f;
        int pass1_failed = 0;
        double p1_start = omp_get_wtime();

        #pragma omp parallel reduction(max:pass1_max) num_threads(actual_threads)
        {
            int tid = omp_get_thread_num();
            int local_failed = 0;
            float* scratch_magsq = spectral_aligned_alloc(n_freqs_f32_bytes);
            if (!scratch_magsq) {
                local_failed = 1;
                #pragma omp atomic write
                pass1_failed = 1;
            }

            #pragma omp for schedule(static)
            for (size_t t = 0; t < n_frames; t++) {
                if (local_failed) continue;
                float frame_max = 0.0f;
                spectral_fft_single_frame(&res, tid, audio, hop, window_ctx.samples, t,
                                          scratch_magsq, NULL, NULL, &frame_max);
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
    spectral_tracker_set_window_descriptor(tracker, window_ctx.descriptor);

    {
        const size_t pair_count = n_frames - 1u;
        const size_t total_chunks = (pair_count + pair_chunk_size - 1u) / pair_chunk_size;
        double parallel_start = omp_get_wtime();

        #pragma omp parallel num_threads(actual_threads)
        {
            int tid = omp_get_thread_num();
            SpectralFusedScratchRows rows = {0};
            SpectralTrackerCandidateBatch candidate_batch = {0};
            uint64_t local_pairs = 0;
            uint64_t local_candidates = 0;
            uint64_t local_segments = 0;
            SpectralTrackerWorkerStats worker_stats = {0};
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

            if (!spectral_fused_scratch_rows_alloc(&rows, n_freqs_f32_bytes)) {
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
                    spectral_fft_single_frame(&res, tid, audio, hop, window_ctx.samples, pair_start,
                                              rows.row_curr, rows.re_curr, rows.im_curr, &frame_max);
                    local_fft_time += (omp_get_wtime() - fft_start);

                    for (size_t pair = pair_start;
                         pair < pair_end && !spectral_tracker_has_failed(tracker);
                         pair++) {
                        SpectralFrameContext fctx;
                        double track_start = 0.0;

                        fft_start = omp_get_wtime();
                        spectral_fft_single_frame(&res, tid, audio, hop, window_ctx.samples, pair + 1u,
                                                  rows.row_next, rows.re_next, rows.im_next, &frame_max);
                        local_fft_time += (omp_get_wtime() - fft_start);

                        if (spectral_tracker_frame_context_init(&fctx,
                                                               rows.row_curr,
                                                               rows.row_next,
                                                               rows.re_curr,
                                                               rows.im_curr,
                                                               rows.re_next,
                                                               rows.im_next,
                                                               pair,
                                                               (float)hop,
                                                               threshsq,
                                                               1) != SPECTRAL_OK) {
                            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
                            break;
                        }

                        track_start = omp_get_wtime();
                        local_pairs++;
                        /* spectral_tracker_run_fused_frame follows tracker helper polarity */
                        if (!spectral_tracker_run_fused_frame(
                            tracker, tid, candidate_batch.ids, &candidate_batch.count,
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

                        spectral_fused_scratch_rows_rotate(&rows);
                    }
                }
            }

            #pragma omp atomic update
            fft_time_total += local_fft_time;

            worker_stats.pairs = local_pairs;
            worker_stats.candidates = local_candidates;
            worker_stats.segments = local_segments;
            worker_stats.track_time = local_track_time;
#if SPECTRAL_TRACK_DEBUG_TIMING
            worker_stats.scan_time = (local_track_time - local_validate_time - local_emit_time > 0.0)
                ? (local_track_time - local_validate_time - local_emit_time) : 0.0;
            worker_stats.validate_time = local_validate_time;
            worker_stats.emit_time = local_emit_time;
            worker_stats.emit_alloc_time = local_emit_alloc_time;
            worker_stats.emit_interp_time = local_emit_interp_time;
            worker_stats.emit_amp_time = local_emit_amp_time;
#endif
            spectral_tracker_worker_stats_commit(tracker, &worker_stats);

            spectral_fused_scratch_rows_free(&rows);
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
            spectral_analysis_window_context_free(&window_ctx);
            res = (SpectralFftResources){0};

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
    spectral_analysis_window_context_free(&window_ctx);
    return spectral_analysis_return_empty(t_fft, t_track);
}
