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
    size_t total_bins = 0;
    size_t total_bytes = 0;
    size_t n_fft_f32_bytes = 0;
    size_t fft_bytes = 0;
    float* window_func = NULL;
    float* magsq = NULL;
    float* phases = NULL;
    float max_magsq = 0.0f;
    SpectralWindowMetrics window_metrics = {0};
    SpectralFftResources res = {0};
    int n_threads = omp_get_max_threads();

    (void)n_samples;

    if (!spectral_size_mul(n_frames, n_freqs, &total_bins) ||
        !spectral_size_mul(total_bins, sizeof(float), &total_bytes) ||
        !spectral_array_bytes((size_t)n_fft, sizeof(float), &n_fft_f32_bytes) ||
        !spectral_analysis_estimate_fft_bytes(n_frames, (size_t)n_fft, n_freqs, &fft_bytes)) {
        return spectral_analysis_return_empty(t_fft, t_track);
    }

    window_func = spectral_aligned_alloc(n_fft_f32_bytes);
    magsq = spectral_aligned_alloc(total_bytes);
    phases = spectral_aligned_alloc(total_bytes);
    if (!window_func || !magsq || !phases) {
        free(window_func);
        free(magsq);
        free(phases);
        return spectral_analysis_return_empty(t_fft, t_track);
    }
    spectral_window_generate(window_func, (size_t)n_fft, SPECTRAL_WINDOW_HANN);
    window_metrics = spectral_window_metrics(window_func, (size_t)n_fft);

#if defined(POSIX_MADV_SEQUENTIAL)
    posix_madvise(magsq, total_bytes, POSIX_MADV_SEQUENTIAL);
    posix_madvise(phases, total_bytes, POSIX_MADV_SEQUENTIAL);
#endif

    if (!spectral_fft_resources_alloc(&res, n_threads, (size_t)n_fft, n_freqs)) {
        spectral_fft_resources_free(&res);
        free(magsq);
        free(phases);
        free(window_func);
        return spectral_analysis_return_empty(t_fft, t_track);
    }
    spectral_fft_resources_set_magsq_scales(&res,
        window_metrics.endpoint_bin_magsq_scale,
        window_metrics.positive_bin_magsq_scale);

    {
        double fft_start = omp_get_wtime();
        max_magsq = spectral_fft_frames(&res, audio, hop, window_func,
                                        0, n_frames, magsq, phases, 0);
        *t_fft = omp_get_wtime() - fft_start;

        {
            double fft_bw_gibps = spectral_bandwidth_gibps(fft_bytes, *t_fft);
            SPECTRAL_LOG_INFO("Analysis full FFT: frames=%zu bytes=%.1fMiB time=%.3fms bw=%.2fGiB/s",
                              n_frames,
                              BYTES_TO_MB(fft_bytes),
                              (*t_fft) * SPECTRAL_MILLIS_PER_SECOND_D,
                              fft_bw_gibps);
        }
    }

    spectral_fft_resources_free(&res);
    free(window_func);

    {
        SegmentArray result = spectral_track_peaks(magsq, phases, max_magsq,
                                                   n_frames, n_freqs,
                                                   sr, n_fft, hop,
                                                   db_thresh, t_track);
        free(phases);
        free(magsq);
        return result;
    }
}
