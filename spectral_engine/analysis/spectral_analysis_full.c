/* spectral_analysis_full.c
 *
 * Full-matrix STFT analysis path for smaller/medium inputs.
 */
#include "spectral_analysis_internal.h"

SegmentArray spectral_analysis_run_full(const float* audio, size_t n_samples,
                                        int sr, int n_fft, int hop, float db_thresh,
                                        size_t n_frames, size_t n_freqs,
                                        double* t_fft, double* t_track)
{
    size_t fft_bytes = 0;
    SpectralAnalysisWindowContext window_ctx = {0};
    SpectralAnalysisStftMatrix stft = {0};
    float max_magsq = 0.0f;
    SpectralFftResources res = {0};
    int n_threads = spectral_omp_effective_thread_count();

    (void)n_samples;

    if (spectral_analysis_stft_matrix_alloc(&stft, n_frames, n_freqs) != SPECTRAL_OK ||
        spectral_analysis_window_context_init(&window_ctx, (size_t)n_fft, SPECTRAL_WINDOW_HANN) != SPECTRAL_OK) {
        spectral_analysis_stft_matrix_free(&stft);
        spectral_analysis_window_context_free(&window_ctx);
        return spectral_analysis_return_empty(t_fft, t_track);
    }

#if defined(POSIX_MADV_SEQUENTIAL)
    posix_madvise(stft.magsq, stft.total_bytes, POSIX_MADV_SEQUENTIAL);
    posix_madvise(stft.phases, stft.total_bytes, POSIX_MADV_SEQUENTIAL);
    posix_madvise(stft.re, stft.total_bytes, POSIX_MADV_SEQUENTIAL);
    posix_madvise(stft.im, stft.total_bytes, POSIX_MADV_SEQUENTIAL);
#endif

    if (!spectral_fft_resources_alloc(&res, n_threads, (size_t)n_fft, n_freqs)) {
        spectral_fft_resources_free(&res);
        spectral_analysis_stft_matrix_free(&stft);
        spectral_analysis_window_context_free(&window_ctx);
        return spectral_analysis_return_empty(t_fft, t_track);
    }
    spectral_analysis_window_context_apply_magsq_scales(&window_ctx, &res);

    {
        double fft_start = omp_get_wtime();
        max_magsq = spectral_fft_frames(&res, audio, hop, window_ctx.samples,
                                        0, n_frames, stft.magsq, stft.phases,
                                        stft.re, stft.im, 0);
        *t_fft = omp_get_wtime() - fft_start;

        {
            /* The byte estimate is for the bandwidth log line only; it must never
             * gate the analysis. On size_t overflow it reports 0 (= unknown). */
            double fft_bw_gibps = 0.0;
            if (spectral_analysis_estimate_fft_bytes(n_frames, (size_t)n_fft, n_freqs, &fft_bytes))
                fft_bw_gibps = spectral_bandwidth_gibps(fft_bytes, *t_fft);
            else
                fft_bytes = 0;
            SPECTRAL_LOG_INFO("Analysis full FFT: frames=%zu bytes=%.1fMiB time=%.3fms bw=%.2fGiB/s",
                              n_frames,
                              BYTES_TO_MB(fft_bytes),
                              (*t_fft) * SPECTRAL_MILLIS_PER_SECOND_D,
                              fft_bw_gibps);
        }
    }

    spectral_fft_resources_free(&res);

    {
        /* window_ctx is freed AFTER this call: spectral_analysis_window_context_free
         * zeroes the struct (descriptor -> NULL), so reading window_ctx.descriptor
         * here would otherwise always take the fallback and silently substitute the
         * HANN descriptor for whatever window was actually used. */
        SegmentArray result = spectral_track_peaks_with_window_descriptor(
            stft.magsq, stft.phases, max_magsq,
            n_frames, n_freqs,
            sr, n_fft, hop,
            db_thresh,
            window_ctx.descriptor ? window_ctx.descriptor : spectral_window_descriptor(SPECTRAL_WINDOW_HANN),
            SPECTRAL_PEAK_ESTIMATOR_AUTO,
            t_track);
        spectral_analysis_window_context_free(&window_ctx);
        spectral_analysis_stft_matrix_free(&stft);
        return result;
    }
}

