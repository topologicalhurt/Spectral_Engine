/* spectral_analysis_fft.c
 *
 * Shared FFT resource management and frame transform helpers.
 */
#include "spectral_analysis_internal.h"

int spectral_fft_resources_alloc(SpectralFftResources* res, int n_threads,
                                 size_t n_fft, size_t n_freqs)
{
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

    {
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
        res->fft_plans[t] = fftwf_plan_dft_r2c_1d((int)n_fft, res->thread_in[t],
                                                  res->thread_out[t], FFTW_ESTIMATE);
        if (!res->fft_plans[t]) return 0;
    }
#endif
    return 1;
}

void spectral_fft_resources_free(SpectralFftResources* res)
{
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

#if SPECTRAL_USE_VDSP

void spectral_fft_single_frame(const SpectralFftResources* res,
                               int tid,
                               const float* audio, int hop,
                               const float* window_func,
                               size_t t,
                               float* out_magsq, float* out_phases,
                               float* out_frame_max)
{
    size_t n_fft = res->n_fft;
    size_t n_freqs = res->n_freqs;
    FFTSetup setup = res->fft_setups[tid];
    DSPSplitComplex split = { res->thread_real[tid], res->thread_imag[tid] };
    float* windowed = res->thread_windowed[tid];
    float* imag_sq_tmp = res->thread_imag_sq[tid];

    const float* src = audio + t * hop;
    vDSP_vmul(src, 1, window_func, 1, windowed, 1, n_fft);
    vDSP_ctoz((DSPComplex*)windowed, 2, &split, 1, n_fft / 2);
    vDSP_fft_zrip(setup, &split, 1, res->log2n, FFT_FORWARD);

    {
        float dc = split.realp[0];
        out_magsq[0] = dc * dc;
        float ny = split.imagp[0];
        out_magsq[n_freqs - 1] = ny * ny;
    }

    {
        size_t mid = n_freqs - 2;
        vDSP_vsq(split.realp + 1, 1, out_magsq + 1, 1, mid);
        vDSP_vsq(split.imagp + 1, 1, imag_sq_tmp, 1, mid);
        vDSP_vadd(out_magsq + 1, 1, imag_sq_tmp, 1, out_magsq + 1, 1, mid);

        if (out_phases) {
            out_phases[0] = (split.realp[0] >= 0.0f) ? 0.0f : SPECTRAL_PI_F;
            out_phases[n_freqs - 1] = (split.imagp[0] >= 0.0f) ? 0.0f : SPECTRAL_PI_F;
            {
                int mid_int = (int)mid;
                vvatan2f(out_phases + 1, split.imagp + 1, split.realp + 1, &mid_int);
            }
        }
    }

    if (out_frame_max) {
        float frame_max = 0.0f;
        vDSP_maxv(out_magsq, 1, &frame_max, n_freqs);
        *out_frame_max = frame_max;
    }
}

float spectral_fft_frames(const SpectralFftResources* res,
                          const float* audio, int hop,
                          const float* window_func,
                          size_t frame_start, size_t frame_end,
                          float* out_magsq, float* out_phases,
                          int magsq_only)
{
    float max_magsq = 0.0f;
    size_t local_n_frames = frame_end - frame_start;
    size_t n_freqs = res->n_freqs;

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < local_n_frames; i++) {
            size_t t = frame_start + i;
            float frame_max = 0.0f;
            spectral_fft_single_frame(res, tid, audio, hop, window_func, t,
                                      out_magsq + i * n_freqs,
                                      magsq_only ? NULL : (out_phases + i * n_freqs),
                                      &frame_max);
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    return max_magsq;
}


#else

void spectral_fft_single_frame(const SpectralFftResources* res,
                               int tid,
                               const float* audio, int hop,
                               const float* window_func,
                               size_t t,
                               float* out_magsq, float* out_phases,
                               float* out_frame_max)
{
    size_t n_fft = res->n_fft;
    size_t n_freqs = res->n_freqs;
    float* in_buf = res->thread_in[tid];
    fftwf_complex* out_buf = res->thread_out[tid];
    fftwf_plan plan = res->fft_plans[tid];

    const float* src = audio + t * hop;
    spectral_vmul(src, window_func, in_buf, n_fft);
    fftwf_execute(plan);

    float frame_max = 0.0f;
    if (!out_phases) {
        spectral_magsq_only((float*)out_buf, out_magsq, &frame_max, n_freqs);
    } else {
        spectral_magsq_phase((float*)out_buf, out_magsq, out_phases, &frame_max, n_freqs);
    }
    
    if (out_frame_max) {
        *out_frame_max = frame_max;
    }
}

float spectral_fft_frames(const SpectralFftResources* res,
                          const float* audio, int hop,
                          const float* window_func,
                          size_t frame_start, size_t frame_end,
                          float* out_magsq, float* out_phases,
                          int magsq_only)
{
    float max_magsq = 0.0f;
    size_t local_n_frames = frame_end - frame_start;
    size_t n_freqs = res->n_freqs;

    #pragma omp parallel reduction(max:max_magsq)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < local_n_frames; i++) {
            size_t t = frame_start + i;
            float frame_max = 0.0f;
            spectral_fft_single_frame(res, tid, audio, hop, window_func, t,
                                      out_magsq + i * n_freqs,
                                      magsq_only ? NULL : (out_phases + i * n_freqs),
                                      &frame_max);
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    return max_magsq;
}

#endif
