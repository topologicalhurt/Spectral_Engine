/* spectral_analysis.c - FFT-based STFT analysis
 *
 * Performs parallel STFT to compute magnitude-squared and phase matrices,
 * then delegates peak tracking to spectral_peak_track.c.
 *
 * Two code paths:
 *   1. Standard: allocates full STFT matrices, single-shot tracking
 *   2. Chunked:  for large datasets (>256MB STFT), single-pass FFT+track
 *                in L3-resident chunks to avoid page fault storms
 */

#include "spectral_analysis.h"
#include "spectral_peak_track.h"
#include "spectral_utils.h"
#include "spectral_windows.h"
#include "spectral_vector_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "spectral_omp.h"

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#else
#include <fftw3.h>
#endif

/* Per-thread FFT resources — shared struct for both standard and chunked paths */

typedef struct {
    int n_threads;
    size_t n_fft;
    size_t n_freqs;
#if SPECTRAL_USE_VDSP
    vDSP_Length log2n;
    FFTSetup* fft_setups;
    float** thread_real;
    float** thread_imag;
    float** thread_windowed;
    float** thread_imag_sq;
#else
    fftwf_plan* fft_plans;
    float** thread_in;
    fftwf_complex** thread_out;
#endif
} FftResources;

/* Allocate per-thread FFT resources. Returns 1 on success, 0 on failure.
 * On failure, caller must still call fft_resources_free(). */
static int fft_resources_alloc(FftResources* res, int n_threads,
                                size_t n_fft, size_t n_freqs) {
    memset(res, 0, sizeof(*res));
    res->n_threads = n_threads;
    res->n_fft = n_fft;
    res->n_freqs = n_freqs;

#if SPECTRAL_USE_VDSP
    res->log2n = (vDSP_Length)log2(n_fft);
    res->fft_setups = spectral_malloc_array((size_t)n_threads, sizeof(FFTSetup));
    res->thread_real = spectral_malloc_array((size_t)n_threads, sizeof(float*));
    res->thread_imag = spectral_malloc_array((size_t)n_threads, sizeof(float*));
    res->thread_windowed = spectral_malloc_array((size_t)n_threads, sizeof(float*));
    res->thread_imag_sq = spectral_malloc_array((size_t)n_threads, sizeof(float*));

    if (!res->fft_setups || !res->thread_real || !res->thread_imag ||
        !res->thread_windowed || !res->thread_imag_sq) {
        return 0;
    }

    size_t n_fft_f32_bytes = 0;
    size_t n_freqs_f32_bytes = 0;
    if (!spectral_array_bytes(n_fft, sizeof(float), &n_fft_f32_bytes) ||
        !spectral_array_bytes(n_freqs, sizeof(float), &n_freqs_f32_bytes)) {
        return 0;
    }

    for (int t = 0; t < n_threads; t++) {
        res->fft_setups[t] = NULL;
        res->thread_real[t] = NULL;
        res->thread_imag[t] = NULL;
        res->thread_windowed[t] = NULL;
        res->thread_imag_sq[t] = NULL;
    }

    for (int t = 0; t < n_threads; t++) {
        res->fft_setups[t] = vDSP_create_fftsetup(res->log2n, FFT_RADIX2);
        res->thread_real[t] = spectral_aligned_alloc(n_fft_f32_bytes);
        res->thread_imag[t] = spectral_aligned_alloc(n_fft_f32_bytes);
        res->thread_windowed[t] = spectral_aligned_alloc(n_fft_f32_bytes);
        res->thread_imag_sq[t] = spectral_aligned_alloc(n_freqs_f32_bytes);
        if (!res->fft_setups[t] || !res->thread_real[t] || !res->thread_imag[t] ||
            !res->thread_windowed[t] || !res->thread_imag_sq[t]) {
            return 0;
        }
    }
#else
    res->fft_plans = spectral_malloc_array((size_t)n_threads, sizeof(fftwf_plan));
    res->thread_in = spectral_malloc_array((size_t)n_threads, sizeof(float*));
    res->thread_out = spectral_malloc_array((size_t)n_threads, sizeof(fftwf_complex*));

    if (!res->fft_plans || !res->thread_in || !res->thread_out) {
        return 0;
    }

    for (int t = 0; t < n_threads; t++) {
        res->fft_plans[t] = NULL;
        res->thread_in[t] = NULL;
        res->thread_out[t] = NULL;
    }

    for (int t = 0; t < n_threads; t++) {
        res->thread_in[t] = fftwf_alloc_real(n_fft);
        res->thread_out[t] = fftwf_alloc_complex(n_freqs);
        if (!res->thread_in[t] || !res->thread_out[t]) {
            return 0;
        }
        res->fft_plans[t] = fftwf_plan_dft_r2c_1d(n_fft, res->thread_in[t],
                                                     res->thread_out[t], FFTW_ESTIMATE);
        if (!res->fft_plans[t]) return 0;
    }
#endif
    return 1;
}

/* Free all per-thread FFT resources. Safe to call after partial alloc. */
static void fft_resources_free(FftResources* res) {
#if SPECTRAL_USE_VDSP
    if (res->fft_setups) {
        for (int t = 0; t < res->n_threads; t++) {
            if (res->fft_setups[t]) vDSP_destroy_fftsetup(res->fft_setups[t]);
        }
        free(res->fft_setups);
    }
    if (res->thread_real) {
        for (int t = 0; t < res->n_threads; t++) free(res->thread_real[t]);
        free(res->thread_real);
    }
    if (res->thread_imag) {
        for (int t = 0; t < res->n_threads; t++) free(res->thread_imag[t]);
        free(res->thread_imag);
    }
    if (res->thread_windowed) {
        for (int t = 0; t < res->n_threads; t++) free(res->thread_windowed[t]);
        free(res->thread_windowed);
    }
    if (res->thread_imag_sq) {
        for (int t = 0; t < res->n_threads; t++) free(res->thread_imag_sq[t]);
        free(res->thread_imag_sq);
    }
#else
    if (res->fft_plans) {
        for (int t = 0; t < res->n_threads; t++) {
            if (res->fft_plans[t]) fftwf_destroy_plan(res->fft_plans[t]);
        }
        free(res->fft_plans);
    }
    if (res->thread_in) {
        for (int t = 0; t < res->n_threads; t++) fftwf_free(res->thread_in[t]);
        free(res->thread_in);
    }
    if (res->thread_out) {
        for (int t = 0; t < res->n_threads; t++) fftwf_free(res->thread_out[t]);
        free(res->thread_out);
    }
#endif
}

/* Forward declaration */
static SegmentArray analyze_audio_chunked(const float* audio, size_t n_samples,
                                           int sr, int n_fft, int hop,
                                           float db_thresh,
                                           size_t n_frames, size_t n_freqs,
                                           double* t_fft, double* t_track);

static SegmentArray analysis_return_empty(double* t_fft, double* t_track) {
    if (t_fft) *t_fft = 0;
    if (t_track) *t_track = 0;
    return (SegmentArray)SEGMENT_ARRAY_EMPTY;
}

SegmentArray analyze_audio(const float* audio, size_t n_samples, int sr,
                           int n_fft, int hop, float db_thresh,
                           double* t_fft, double* t_track) {
    size_t total_bins = 0;
    size_t total_bytes = 0;
    if (!audio || !t_fft || !t_track || n_fft <= 0 || hop <= 0 || n_samples < (size_t)n_fft) {
        return analysis_return_empty(t_fft, t_track);
    }
    size_t n_frames = (n_samples - n_fft) / hop + 1;
    size_t n_freqs = n_fft / 2 + 1;
    if (n_freqs == 0 || !spectral_size_mul(n_frames, n_freqs, &total_bins)) {
        return analysis_return_empty(t_fft, t_track);
    }
    if (!spectral_size_mul(total_bins, sizeof(float), &total_bytes)) {
        return analysis_return_empty(t_fft, t_track);
    }

    /* Dispatch to chunked path for large datasets */
    if (total_bins > SPECTRAL_STFT_CHUNK_THRESHOLD) {
        return analyze_audio_chunked(audio, n_samples, sr, n_fft, hop,
                                      db_thresh, n_frames, n_freqs,
                                      t_fft, t_track);
    }

    /* --- Standard path: full STFT allocation --- */

    size_t n_fft_f32_bytes = 0;
    if (!spectral_array_bytes((size_t)n_fft, sizeof(float), &n_fft_f32_bytes)) {
        return analysis_return_empty(t_fft, t_track);
    }

    float* window_func = spectral_aligned_alloc(n_fft_f32_bytes);
    float* magsq = spectral_aligned_alloc(total_bytes);
    float* phases = spectral_aligned_alloc(total_bytes);
    if (!window_func || !magsq || !phases) {
        free(window_func);
        free(magsq);
        free(phases);
        return analysis_return_empty(t_fft, t_track);
    }
    spectral_window_hann(window_func, n_fft);

    /* Hint to the OS that STFT matrices will be accessed sequentially */
    posix_madvise(magsq, total_bytes, POSIX_MADV_SEQUENTIAL);
    posix_madvise(phases, total_bytes, POSIX_MADV_SEQUENTIAL);

    float max_magsq = 0.0f;
    FftResources res;
    int n_threads = omp_get_max_threads();

    if (!fft_resources_alloc(&res, n_threads, n_fft, n_freqs)) {
        fft_resources_free(&res);
        free(magsq); free(phases); free(window_func);
        return analysis_return_empty(t_fft, t_track);
    }

    double fft_start = omp_get_wtime();

#if SPECTRAL_USE_VDSP
    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        FFTSetup setup = res.fft_setups[tid];
        DSPSplitComplex split = { res.thread_real[tid], res.thread_imag[tid] };
        float* windowed = res.thread_windowed[tid];
        float* imag_sq_tmp = res.thread_imag_sq[tid];

        #pragma omp for schedule(static)
        for (size_t t = 0; t < n_frames; t++) {
            const float* src = audio + t * hop;
            vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
            vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft/2);
            vDSP_fft_zrip(setup, &split, 1, res.log2n, FFT_FORWARD);

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
#else
    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        float* in_buf = res.thread_in[tid];
        fftwf_complex* out_buf = res.thread_out[tid];
        fftwf_plan plan = res.fft_plans[tid];

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
#endif

    *t_fft = omp_get_wtime() - fft_start;
    fft_resources_free(&res);
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
 * Single-pass: FFT+Track in SPECTRAL_STFT_CHUNK_FRAMES-sized chunks with running max. */

/* Helper: compute FFT for a range of frames into magsq/phases buffers.
 * The buffers must hold (frame_end - frame_start) * n_freqs floats.
 * Returns max magsq across all computed frames. */
#if SPECTRAL_USE_VDSP

static float fft_frames(const FftResources* res,
                         const float* audio, int hop,
                         const float* window_func,
                         size_t frame_start, size_t frame_end,
                         float* out_magsq, float* out_phases,
                         int magsq_only) {
    float max_magsq = 0.0f;
    size_t local_n_frames = frame_end - frame_start;
    size_t n_fft = res->n_fft;
    size_t n_freqs = res->n_freqs;

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        FFTSetup setup = res->fft_setups[tid];
        DSPSplitComplex split = { res->thread_real[tid], res->thread_imag[tid] };
        float* windowed = res->thread_windowed[tid];
        float* imag_sq_tmp = res->thread_imag_sq[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < local_n_frames; i++) {
            size_t t = frame_start + i;
            const float* src = audio + t * hop;
            vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
            vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft/2);
            vDSP_fft_zrip(setup, &split, 1, res->log2n, FFT_FORWARD);

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

static float fft_frames(const FftResources* res,
                         const float* audio, int hop,
                         const float* window_func,
                         size_t frame_start, size_t frame_end,
                         float* out_magsq, float* out_phases,
                         int magsq_only) {
    float max_magsq = 0.0f;
    size_t local_n_frames = frame_end - frame_start;
    size_t n_fft = res->n_fft;
    size_t n_freqs = res->n_freqs;

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();
        float* in_buf = res->thread_in[tid];
        fftwf_complex* out_buf = res->thread_out[tid];
        fftwf_plan plan = res->fft_plans[tid];

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
    size_t chunk_bins = 0;
    size_t chunk_bytes = 0;
    float* window_func = NULL;
    float* chunk_magsq = NULL;
    float* chunk_phases = NULL;
    FftResources res = {0};
    (void)n_samples;

    int n_threads = omp_get_max_threads();

    size_t n_fft_f32_bytes = 0;
    if (!spectral_array_bytes((size_t)n_fft, sizeof(float), &n_fft_f32_bytes)) {
        goto fail;
    }

    window_func = spectral_aligned_alloc(n_fft_f32_bytes);
    if (!window_func) {
        goto fail;
    }
    spectral_window_hann(window_func, n_fft);

    if (!fft_resources_alloc(&res, n_threads, n_fft, n_freqs)) {
        goto fail;
    }

    /* Single-pass chunked FFT + Track.
     * Tracker is created after the first chunk using that chunk's max as
     * the initial threshold, then updated with a running max across chunks.
     * If the global max rises significantly, a post-filter pass removes
     * segments that only passed the initial (too-permissive) threshold. */

    double fft_time_total = 0.0;
    float global_max_magsq = 0.0f;
    float first_chunk_max = 0.0f;
    SpectralTracker* tracker = NULL;

    size_t chunk_frames = SPECTRAL_STFT_CHUNK_FRAMES;
    size_t chunk_alloc_frames = chunk_frames + 1;
    if (n_freqs == 0 || !spectral_size_mul(chunk_alloc_frames, n_freqs, &chunk_bins)) {
        goto fail;
    }
    if (!spectral_size_mul(chunk_bins, sizeof(float), &chunk_bytes)) {
        goto fail;
    }
    chunk_magsq = spectral_aligned_alloc(chunk_bytes);
    chunk_phases = spectral_aligned_alloc(chunk_bytes);

    if (!chunk_magsq || !chunk_phases) {
        goto fail;
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

        float chunk_max = fft_frames(&res, audio, hop, window_func,
                                      chunk_start, fft_end, chunk_magsq, chunk_phases, 0);

        fft_time_total += omp_get_wtime() - chunk_fft_start;

        int max_increased = (chunk_max > global_max_magsq);
        if (max_increased) global_max_magsq = chunk_max;

        /* Create tracker after first chunk using first chunk's max */
        if (!tracker) {
            first_chunk_max = global_max_magsq;
            tracker = spectral_tracker_create(
                n_threads, n_freqs, sr, n_fft, hop, db_thresh, global_max_magsq);
            if (!tracker) {
                goto fail;
            }
        } else if (max_increased) {
            /* Refine threshold with updated global max so subsequent chunks
             * track at the correct relative threshold */
            spectral_tracker_update_threshold(tracker, global_max_magsq);
        }

        /* Overlap row for tracker: points to the extra frame's magsq */
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
    fft_resources_free(&res);
    free(window_func);

    *t_fft = fft_time_total;

    if (!tracker) {
        return analysis_return_empty(t_fft, t_track);
    }

    SegmentArray result = spectral_tracker_finalize(tracker, t_track);

    /* Refinement: if the global max rose significantly from the first chunk's
     * max (4x = 6dB power), the first chunk was tracked with a too-permissive
     * threshold. Post-filter removes segments below the corrected threshold. */
    if (first_chunk_max > 0.0f && global_max_magsq > first_chunk_max * 4.0f
        && result.count > 0) {
        float thresh_linear = powf(10.0f, db_thresh / 20.0f);
        float amp_thresh = thresh_linear * sqrtf(global_max_magsq);
        uint32_t kept = 0;
        for (uint32_t i = 0; i < result.count; i++) {
            if (result.segs[i].amp >= amp_thresh) {
                if (kept != i) result.segs[kept] = result.segs[i];
                kept++;
            }
        }
        if (kept < result.count) {
            result.count = kept;
        }
    }

    return result;

fail:
    free(chunk_magsq);
    free(chunk_phases);
    fft_resources_free(&res);
    free(window_func);
    return analysis_return_empty(t_fft, t_track);
}
