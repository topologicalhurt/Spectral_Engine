/* spectral_analysis.c - FFT-based STFT analysis
 *
 * Performs parallel STFT to compute magnitude-squared and phase matrices,
 * then delegates peak tracking to spectral_peak_track.c.
 *
 * Two code paths:
 *   1. Standard: allocates full STFT matrices, single-shot tracking
 *   2. Chunked:  for large datasets (>256MB STFT), processes in L3-resident
 *                chunks to avoid page fault storms and cache thrashing
 */

#include "spectral_analysis.h"
#include "spectral_peak_track.h"
#include "spectral_windows.h"
#include "spectral_vector_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
static inline double omp_get_wtime(void) { return 0.0; }
#endif

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#else
#include <fftw3.h>
#endif

/* Forward declaration */
static SegmentArray analyze_audio_chunked(const float* audio, size_t n_samples,
                                           int sr, int n_fft, int hop,
                                           float db_thresh,
                                           size_t n_frames, size_t n_freqs,
                                           double* t_fft, double* t_track);

SegmentArray analyze_audio(const float* audio, size_t n_samples, int sr,
                           int n_fft, int hop, float db_thresh,
                           double* t_fft, double* t_track) {
    SegmentArray empty_result = {NULL, 0, 0};
    if (!audio || !t_fft || !t_track || n_fft <= 0 || hop <= 0 || n_samples < (size_t)n_fft) {
        if (t_fft) *t_fft = 0;
        if (t_track) *t_track = 0;
        return empty_result;
    }
    size_t n_frames = (n_samples - n_fft) / hop + 1;
    size_t n_freqs = n_fft / 2 + 1;
    size_t total_bins = n_frames * n_freqs;

    /* Dispatch to chunked path for large datasets */
    if (total_bins > STFT_CHUNK_THRESHOLD) {
        return analyze_audio_chunked(audio, n_samples, sr, n_fft, hop,
                                      db_thresh, n_frames, n_freqs,
                                      t_fft, t_track);
    }

    /* --- Standard path: full STFT allocation --- */

    float* window_func = spectral_aligned_alloc(n_fft * sizeof(float));
    float* magsq = spectral_aligned_alloc(total_bins * sizeof(float));
    float* phases = spectral_aligned_alloc(total_bins * sizeof(float));
    if (!window_func || !magsq || !phases) {
        free(window_func);
        free(magsq);
        free(phases);
        *t_fft = 0;
        *t_track = 0;
        return empty_result;
    }
    spectral_window_hann(window_func, n_fft);

    /* Hint to the OS that STFT matrices will be accessed sequentially */
    posix_madvise(magsq, total_bins * sizeof(float), POSIX_MADV_SEQUENTIAL);
    posix_madvise(phases, total_bins * sizeof(float), POSIX_MADV_SEQUENTIAL);

    float max_magsq = 0.0f;

#if SPECTRAL_USE_VDSP
    vDSP_Length log2n = (vDSP_Length)log2(n_fft);
    int n_threads = omp_get_max_threads();
    FFTSetup* fft_setups = malloc(n_threads * sizeof(FFTSetup));
    float** thread_real = malloc(n_threads * sizeof(float*));
    float** thread_imag = malloc(n_threads * sizeof(float*));
    float** thread_windowed = malloc(n_threads * sizeof(float*));
    float** thread_imag_sq = malloc(n_threads * sizeof(float*));

    if (!fft_setups || !thread_real || !thread_imag || !thread_windowed || !thread_imag_sq) {
        free(fft_setups); free(thread_real); free(thread_imag);
        free(thread_windowed); free(thread_imag_sq);
        free(magsq); free(phases); free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    for (int t = 0; t < n_threads; t++) {
        fft_setups[t] = NULL;
        thread_real[t] = NULL;
        thread_imag[t] = NULL;
        thread_windowed[t] = NULL;
        thread_imag_sq[t] = NULL;
    }

    int alloc_success = 1;
    for (int t = 0; t < n_threads && alloc_success; t++) {
        fft_setups[t] = vDSP_create_fftsetup(log2n, FFT_RADIX2);
        thread_real[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_imag[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_windowed[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_imag_sq[t] = spectral_aligned_alloc(n_freqs * sizeof(float));
        if (!fft_setups[t] || !thread_real[t] || !thread_imag[t] ||
            !thread_windowed[t] || !thread_imag_sq[t]) {
            alloc_success = 0;
        }
    }

    if (!alloc_success) {
        for (int t = 0; t < n_threads; t++) {
            if (fft_setups[t]) vDSP_destroy_fftsetup(fft_setups[t]);
            free(thread_real[t]);
            free(thread_imag[t]);
            free(thread_windowed[t]);
            free(thread_imag_sq[t]);
        }
        free(fft_setups); free(thread_real); free(thread_imag);
        free(thread_windowed); free(thread_imag_sq);
        free(magsq); free(phases); free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    double fft_start = omp_get_wtime();

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        FFTSetup setup = fft_setups[tid];
        DSPSplitComplex split = { thread_real[tid], thread_imag[tid] };
        float* windowed = thread_windowed[tid];
        float* imag_sq_tmp = thread_imag_sq[tid];

        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_frames; t++) {
            const float* src = audio + t * hop;
            vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
            vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft/2);
            vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD);

            float* magsq_row = magsq + t * n_freqs;
            float* phase_row = phases + t * n_freqs;

            float dc = split.realp[0];
            magsq_row[0] = dc * dc;
            phase_row[0] = dc >= 0.0f ? 0.0f : SPECTRAL_PI_F;

            float ny = split.imagp[0];
            magsq_row[n_freqs - 1] = ny * ny;
            phase_row[n_freqs - 1] = ny >= 0.0f ? 0.0f : SPECTRAL_PI_F;

            size_t mid = n_freqs - 2;
            vDSP_vsq(split.realp + 1, 1, magsq_row + 1, 1, mid);
            vDSP_vsq(split.imagp + 1, 1, imag_sq_tmp, 1, mid);
            vDSP_vadd(magsq_row + 1, 1, imag_sq_tmp, 1, magsq_row + 1, 1, mid);
            int mid_int = (int)mid;
            vvatan2f(phase_row + 1, split.imagp + 1, split.realp + 1, &mid_int);

            float frame_max;
            vDSP_maxv(magsq_row, 1, &frame_max, n_freqs);
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    *t_fft = omp_get_wtime() - fft_start;

    for (int t = 0; t < n_threads; t++) {
        vDSP_destroy_fftsetup(fft_setups[t]);
        free(thread_real[t]);
        free(thread_imag[t]);
        free(thread_windowed[t]);
        free(thread_imag_sq[t]);
    }
    free(fft_setups);
    free(thread_real);
    free(thread_imag);
    free(thread_windowed);
    free(thread_imag_sq);
#else
    int n_threads = omp_get_max_threads();
    fftwf_plan* fft_plans = malloc(n_threads * sizeof(fftwf_plan));
    float** thread_in = malloc(n_threads * sizeof(float*));
    fftwf_complex** thread_out = malloc(n_threads * sizeof(fftwf_complex*));

    if (!fft_plans || !thread_in || !thread_out) {
        free(fft_plans); free(thread_in); free(thread_out);
        free(magsq); free(phases); free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    for (int t = 0; t < n_threads; t++) {
        fft_plans[t] = NULL;
        thread_in[t] = NULL;
        thread_out[t] = NULL;
    }

    int alloc_success = 1;
    for (int t = 0; t < n_threads && alloc_success; t++) {
        thread_in[t] = fftwf_alloc_real(n_fft);
        thread_out[t] = fftwf_alloc_complex(n_freqs);
        if (!thread_in[t] || !thread_out[t]) {
            alloc_success = 0;
        } else {
            fft_plans[t] = fftwf_plan_dft_r2c_1d(n_fft, thread_in[t], thread_out[t], FFTW_ESTIMATE);
            if (!fft_plans[t]) alloc_success = 0;
        }
    }

    if (!alloc_success) {
        for (int t = 0; t < n_threads; t++) {
            if (fft_plans[t]) fftwf_destroy_plan(fft_plans[t]);
            fftwf_free(thread_in[t]);
            fftwf_free(thread_out[t]);
        }
        free(fft_plans); free(thread_in); free(thread_out);
        free(magsq); free(phases); free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    double fft_start = omp_get_wtime();

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        float* in_buf = thread_in[tid];
        fftwf_complex* out_buf = thread_out[tid];
        fftwf_plan plan = fft_plans[tid];

        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_frames; t++) {
            const float* src = audio + t * hop;
            spectral_vmul(src, window_func, in_buf, (size_t)n_fft);
            fftwf_execute(plan);

            float frame_max;
            spectral_magsq_phase((float*)out_buf,
                                 magsq + t * n_freqs,
                                 phases + t * n_freqs,
                                 &frame_max, n_freqs);
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    *t_fft = omp_get_wtime() - fft_start;

    for (int t = 0; t < n_threads; t++) {
        fftwf_destroy_plan(fft_plans[t]);
        fftwf_free(thread_in[t]);
        fftwf_free(thread_out[t]);
    }
    free(fft_plans);
    free(thread_in);
    free(thread_out);
#endif
    free(window_func);

    SegmentArray result = spectral_track_peaks(magsq, phases, max_magsq,
                                               n_frames, n_freqs,
                                               sr, n_fft, hop,
                                               db_thresh, t_track);
    free(phases);
    free(magsq);
    return result;
}

/* Chunked analysis path — for large datasets (>256MB STFT).
 * Pass 1: Max scan (no permanent STFT storage).
 * Pass 2: Chunked FFT+Track (STFT_CHUNK_FRAMES at a time). */

/* Helper: compute FFT for a range of frames into magsq/phases buffers.
 * The buffers must hold (frame_end - frame_start) * n_freqs floats.
 * Returns max magsq across all computed frames (only meaningful if
 * out_phases != NULL; otherwise max is still tracked for magsq-only mode). */
#if SPECTRAL_USE_VDSP

static float fft_frames_vdsp(const float* audio, int hop,
                               const float* window_func, size_t n_fft,
                               size_t n_freqs, vDSP_Length log2n,
                               FFTSetup* fft_setups,
                               float** thread_real, float** thread_imag,
                               float** thread_windowed, float** thread_imag_sq,
                               size_t frame_start, size_t frame_end,
                               float* out_magsq, float* out_phases,
                               int magsq_only) {
    float max_magsq = 0.0f;
    size_t local_n_frames = frame_end - frame_start;

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        FFTSetup setup = fft_setups[tid];
        DSPSplitComplex split = { thread_real[tid], thread_imag[tid] };
        float* windowed = thread_windowed[tid];
        float* imag_sq_tmp = thread_imag_sq[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < local_n_frames; i++) {
            size_t t = frame_start + i;
            const float* src = audio + t * hop;
            vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
            vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft/2);
            vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD);

            float* magsq_row = out_magsq + i * n_freqs;

            float dc = split.realp[0];
            magsq_row[0] = dc * dc;

            float ny = split.imagp[0];
            magsq_row[n_freqs - 1] = ny * ny;

            size_t mid = n_freqs - 2;
            vDSP_vsq(split.realp + 1, 1, magsq_row + 1, 1, mid);
            vDSP_vsq(split.imagp + 1, 1, imag_sq_tmp, 1, mid);
            vDSP_vadd(magsq_row + 1, 1, imag_sq_tmp, 1, magsq_row + 1, 1, mid);

            if (!magsq_only) {
                float* phase_row = out_phases + i * n_freqs;
                phase_row[0] = dc >= 0.0f ? 0.0f : SPECTRAL_PI_F;
                phase_row[n_freqs - 1] = ny >= 0.0f ? 0.0f : SPECTRAL_PI_F;
                int mid_int = (int)mid;
                vvatan2f(phase_row + 1, split.imagp + 1, split.realp + 1, &mid_int);
            }

            float frame_max;
            vDSP_maxv(magsq_row, 1, &frame_max, n_freqs);
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    return max_magsq;
}

#else /* FFTW path */

static float fft_frames_fftw(const float* audio, int hop,
                               const float* window_func, size_t n_fft,
                               size_t n_freqs,
                               fftwf_plan* fft_plans,
                               float** thread_in, fftwf_complex** thread_out,
                               size_t frame_start, size_t frame_end,
                               float* out_magsq, float* out_phases,
                               int magsq_only) {
    float max_magsq = 0.0f;
    size_t local_n_frames = frame_end - frame_start;

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        float* in_buf = thread_in[tid];
        fftwf_complex* out_buf = thread_out[tid];
        fftwf_plan plan = fft_plans[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < local_n_frames; i++) {
            size_t t = frame_start + i;
            const float* src = audio + t * hop;
            spectral_vmul(src, window_func, in_buf, n_fft);
            fftwf_execute(plan);

            float frame_max;
            if (magsq_only) {
                spectral_magsq_only((float*)out_buf,
                                     out_magsq + i * n_freqs,
                                     &frame_max, n_freqs);
            } else {
                spectral_magsq_phase((float*)out_buf,
                                      out_magsq + i * n_freqs,
                                      out_phases + i * n_freqs,
                                      &frame_max, n_freqs);
            }
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    return max_magsq;
}

#endif /* SPECTRAL_USE_VDSP */

static SegmentArray analyze_audio_chunked(const float* audio, size_t n_samples,
                                           int sr, int n_fft, int hop,
                                           float db_thresh,
                                           size_t n_frames, size_t n_freqs,
                                           double* t_fft, double* t_track) {
    SegmentArray empty_result = {NULL, 0, 0};
    (void)n_samples;

    int n_threads = omp_get_max_threads();

    float* window_func = spectral_aligned_alloc(n_fft * sizeof(float));
    if (!window_func) {
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }
    spectral_window_hann(window_func, n_fft);

    /* --- Allocate per-thread FFT resources (shared across both passes) --- */

#if SPECTRAL_USE_VDSP
    vDSP_Length log2n = (vDSP_Length)log2(n_fft);
    FFTSetup* fft_setups = malloc(n_threads * sizeof(FFTSetup));
    float** thread_real = malloc(n_threads * sizeof(float*));
    float** thread_imag = malloc(n_threads * sizeof(float*));
    float** thread_windowed = malloc(n_threads * sizeof(float*));
    float** thread_imag_sq = malloc(n_threads * sizeof(float*));

    if (!fft_setups || !thread_real || !thread_imag || !thread_windowed || !thread_imag_sq) {
        free(fft_setups); free(thread_real); free(thread_imag);
        free(thread_windowed); free(thread_imag_sq);
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    for (int t = 0; t < n_threads; t++) {
        fft_setups[t] = NULL;
        thread_real[t] = NULL;
        thread_imag[t] = NULL;
        thread_windowed[t] = NULL;
        thread_imag_sq[t] = NULL;
    }

    int alloc_success = 1;
    for (int t = 0; t < n_threads && alloc_success; t++) {
        fft_setups[t] = vDSP_create_fftsetup(log2n, FFT_RADIX2);
        thread_real[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_imag[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_windowed[t] = spectral_aligned_alloc(n_fft * sizeof(float));
        thread_imag_sq[t] = spectral_aligned_alloc(n_freqs * sizeof(float));
        if (!fft_setups[t] || !thread_real[t] || !thread_imag[t] ||
            !thread_windowed[t] || !thread_imag_sq[t]) {
            alloc_success = 0;
        }
    }

    if (!alloc_success) {
        for (int t = 0; t < n_threads; t++) {
            if (fft_setups[t]) vDSP_destroy_fftsetup(fft_setups[t]);
            free(thread_real[t]);
            free(thread_imag[t]);
            free(thread_windowed[t]);
            free(thread_imag_sq[t]);
        }
        free(fft_setups); free(thread_real); free(thread_imag);
        free(thread_windowed); free(thread_imag_sq);
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    #define FFT_CLEANUP() do { \
        for (int _t = 0; _t < n_threads; _t++) { \
            if (fft_setups[_t]) vDSP_destroy_fftsetup(fft_setups[_t]); \
            free(thread_real[_t]); \
            free(thread_imag[_t]); \
            free(thread_windowed[_t]); \
            free(thread_imag_sq[_t]); \
        } \
        free(fft_setups); free(thread_real); free(thread_imag); \
        free(thread_windowed); free(thread_imag_sq); \
    } while (0)

    #define FFT_FRAMES(start, end, out_m, out_p, mo) \
        fft_frames_vdsp(audio, hop, window_func, n_fft, n_freqs, log2n, \
                         fft_setups, thread_real, thread_imag, \
                         thread_windowed, thread_imag_sq, \
                         (start), (end), (out_m), (out_p), (mo))

#else /* FFTW */
    fftwf_plan* fft_plans = malloc(n_threads * sizeof(fftwf_plan));
    float** thread_in = malloc(n_threads * sizeof(float*));
    fftwf_complex** thread_out = malloc(n_threads * sizeof(fftwf_complex*));

    if (!fft_plans || !thread_in || !thread_out) {
        free(fft_plans); free(thread_in); free(thread_out);
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    for (int t = 0; t < n_threads; t++) {
        fft_plans[t] = NULL;
        thread_in[t] = NULL;
        thread_out[t] = NULL;
    }

    int alloc_success = 1;
    for (int t = 0; t < n_threads && alloc_success; t++) {
        thread_in[t] = fftwf_alloc_real(n_fft);
        thread_out[t] = fftwf_alloc_complex(n_freqs);
        if (!thread_in[t] || !thread_out[t]) {
            alloc_success = 0;
        } else {
            fft_plans[t] = fftwf_plan_dft_r2c_1d(n_fft, thread_in[t], thread_out[t], FFTW_ESTIMATE);
            if (!fft_plans[t]) alloc_success = 0;
        }
    }

    if (!alloc_success) {
        for (int t = 0; t < n_threads; t++) {
            if (fft_plans[t]) fftwf_destroy_plan(fft_plans[t]);
            fftwf_free(thread_in[t]);
            fftwf_free(thread_out[t]);
        }
        free(fft_plans); free(thread_in); free(thread_out);
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    #define FFT_CLEANUP() do { \
        for (int _t = 0; _t < n_threads; _t++) { \
            if (fft_plans[_t]) fftwf_destroy_plan(fft_plans[_t]); \
            fftwf_free(thread_in[_t]); \
            fftwf_free(thread_out[_t]); \
        } \
        free(fft_plans); free(thread_in); free(thread_out); \
    } while (0)

    #define FFT_FRAMES(start, end, out_m, out_p, mo) \
        fft_frames_fftw(audio, hop, window_func, n_fft, n_freqs, \
                         fft_plans, thread_in, thread_out, \
                         (start), (end), (out_m), (out_p), (mo))

#endif /* SPECTRAL_USE_VDSP */

    /* Pass 1: Max scan — per-thread scratch (no permanent STFT), skip atan2 */

    double fft_time_total = 0.0;
    double pass1_start = omp_get_wtime();

    /* Per-thread scratch for max scan (just one row per thread) */
    float* scratch_magsq = spectral_aligned_alloc(n_threads * n_freqs * sizeof(float));
    if (!scratch_magsq) {
        FFT_CLEANUP();
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    float global_max_magsq = 0.0f;

    /* Process all frames in parallel, each thread writes to its own scratch row */
    #pragma omp parallel reduction(max:global_max_magsq)
    {
        int tid = omp_get_thread_num();
        float* my_magsq = scratch_magsq + tid * n_freqs;

        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_frames; t++) {
            const float* src = audio + t * hop;

#if SPECTRAL_USE_VDSP
            FFTSetup setup = fft_setups[tid];
            DSPSplitComplex split = { thread_real[tid], thread_imag[tid] };
            float* windowed = thread_windowed[tid];
            float* imag_sq_tmp = thread_imag_sq[tid];

            vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
            vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft/2);
            vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD);

            float dc = split.realp[0];
            my_magsq[0] = dc * dc;
            float ny = split.imagp[0];
            my_magsq[n_freqs - 1] = ny * ny;

            size_t mid = n_freqs - 2;
            vDSP_vsq(split.realp + 1, 1, my_magsq + 1, 1, mid);
            vDSP_vsq(split.imagp + 1, 1, imag_sq_tmp, 1, mid);
            vDSP_vadd(my_magsq + 1, 1, imag_sq_tmp, 1, my_magsq + 1, 1, mid);

            float frame_max;
            vDSP_maxv(my_magsq, 1, &frame_max, n_freqs);
#else
            float* in_buf = thread_in[tid];
            fftwf_complex* out_buf = thread_out[tid];
            fftwf_plan plan = fft_plans[tid];

            spectral_vmul(src, window_func, in_buf, (size_t)n_fft);
            fftwf_execute(plan);

            float frame_max;
            spectral_magsq_only((float*)out_buf, my_magsq, &frame_max, n_freqs);
#endif
            if (frame_max > global_max_magsq) global_max_magsq = frame_max;
        }
    }

    free(scratch_magsq);
    fft_time_total += omp_get_wtime() - pass1_start;

    /* Pass 2: Chunked FFT + Track — (CHUNK_FRAMES + 1) rows for overlap */

    size_t chunk_frames = STFT_CHUNK_FRAMES;
    size_t chunk_alloc_frames = chunk_frames + 1;
    float* chunk_magsq = spectral_aligned_alloc(chunk_alloc_frames * n_freqs * sizeof(float));
    float* chunk_phases = spectral_aligned_alloc(chunk_alloc_frames * n_freqs * sizeof(float));

    if (!chunk_magsq || !chunk_phases) {
        free(chunk_magsq);
        free(chunk_phases);
        FFT_CLEANUP();
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    SpectralTracker* tracker = spectral_tracker_create(
        n_threads, n_freqs, sr, n_fft, hop, db_thresh, global_max_magsq);
    if (!tracker) {
        free(chunk_magsq);
        free(chunk_phases);
        FFT_CLEANUP();
        free(window_func);
        *t_fft = 0; *t_track = 0;
        return empty_result;
    }

    for (size_t chunk_start = 0; chunk_start < n_frames; chunk_start += chunk_frames) {
        size_t chunk_end = chunk_start + chunk_frames;
        if (chunk_end > n_frames) chunk_end = n_frames;
        size_t this_chunk_frames = chunk_end - chunk_start;

        /* How many frames to FFT: this_chunk + 1 overlap if not the last chunk */
        int is_last_chunk = (chunk_end >= n_frames);
        size_t fft_end = is_last_chunk ? chunk_end : (chunk_end + 1);
        if (fft_end > n_frames) fft_end = n_frames;
        size_t fft_count = fft_end - chunk_start;

        double chunk_fft_start = omp_get_wtime();

        /* FFT this chunk's frames (plus overlap) into chunk buffers */
        FFT_FRAMES(chunk_start, fft_end, chunk_magsq, chunk_phases, 0);

        fft_time_total += omp_get_wtime() - chunk_fft_start;

        /* Overlap row for tracker: points to the extra frame's magsq
         * (only magsq needed, not phases, for the next-row look-ahead) */
        const float* overlap_magsq = NULL;
        if (!is_last_chunk && fft_count > this_chunk_frames) {
            overlap_magsq = chunk_magsq + this_chunk_frames * n_freqs;
        }

        spectral_tracker_process(tracker, chunk_magsq, chunk_phases,
                                  this_chunk_frames, chunk_start,
                                  overlap_magsq);
    }

    free(chunk_magsq);
    free(chunk_phases);
    FFT_CLEANUP();
    free(window_func);

    *t_fft = fft_time_total;

    SegmentArray result = spectral_tracker_finalize(tracker, t_track);
    return result;
}

#undef FFT_CLEANUP
#undef FFT_FRAMES
