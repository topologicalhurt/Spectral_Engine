/* spectral_analysis_fft.c
 *
 * Shared FFT resource management and frame transform helpers.
 */
#include "spectral_analysis_internal.h"
#include <limits.h>

static int spectral_fft_magsq_scale_valid(float scale)
{
    return isfinite(scale) && scale > 0.0f;
}

void spectral_fft_resources_set_magsq_scales(SpectralFftResources* res,
                                             float endpoint_bin_magsq_scale,
                                             float positive_bin_magsq_scale)
{
    if (!res) return;
    if (spectral_fft_magsq_scale_valid(endpoint_bin_magsq_scale)) {
        res->endpoint_bin_magsq_scale = endpoint_bin_magsq_scale;
    }
    if (spectral_fft_magsq_scale_valid(positive_bin_magsq_scale)) {
        res->positive_bin_magsq_scale = positive_bin_magsq_scale;
    }
}

static float spectral_fft_trackable_magsq_max(const float* magsq, size_t n_freqs)
{
    float max_magsq = 0.0f;
    if (!magsq || n_freqs <= 2u) return 0.0f;

    for (size_t i = 1u; i + 1u < n_freqs; i++) {
        if (magsq[i] > max_magsq) {
            max_magsq = magsq[i];
        }
    }
    return max_magsq;
}

static void spectral_fft_apply_magsq_scales(const SpectralFftResources* res,
                                            float* magsq,
                                            size_t n_freqs,
                                            float* frame_max)
{
    float endpoint_scale = res ? res->endpoint_bin_magsq_scale : 1.0f;
    float positive_scale = res ? res->positive_bin_magsq_scale : 1.0f;

    if (!magsq || n_freqs == 0u) {
        if (frame_max) *frame_max = 0.0f;
        return;
    }
    if (!spectral_fft_magsq_scale_valid(endpoint_scale)) endpoint_scale = 1.0f;
    if (!spectral_fft_magsq_scale_valid(positive_scale)) positive_scale = 1.0f;

    /* Real FFT output is one-sided. Interior positive-frequency bins represent
     * paired positive/negative-frequency sinusoids and use the doubled
     * coherent-gain scale, while DC and Nyquist are unpaired endpoints. SciPy's
     * one-sided periodogram doubles only interior bins, and FFTW documents the
     * real-DFT endpoint packing used by the portable path:
     * https://scipy.github.io/devdocs/reference/generated/scipy.signal.periodogram.html
     * https://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
     */
    magsq[0] *= endpoint_scale;
    if (n_freqs > 1u) {
        magsq[n_freqs - 1u] *= endpoint_scale;
    }
    for (size_t i = 1u; i + 1u < n_freqs; i++) {
        magsq[i] *= positive_scale;
    }
    if (frame_max) {
        *frame_max = spectral_fft_trackable_magsq_max(magsq, n_freqs);
    }
}

int spectral_fft_resources_alloc(SpectralFftResources* res, int n_threads,
                                 size_t n_fft, size_t n_freqs)
{
    if (!res) return 0;
    memset(res, 0, sizeof(*res));

    if (n_threads < 1 ||
        n_fft < (size_t)SPECTRAL_MIN_FFT_SIZE ||
        n_fft > (size_t)INT_MAX ||
        (n_fft & (n_fft - 1u)) != 0u ||
        n_freqs != (n_fft / 2u + 1u)) {
        return 0;
    }

    res->n_threads = n_threads;
    res->n_fft = n_fft;
    res->n_freqs = n_freqs;
    res->endpoint_bin_magsq_scale = 1.0f;
    res->positive_bin_magsq_scale = 1.0f;

#if SPECTRAL_USE_VDSP
    res->log2n = (vDSP_Length)log2((double)n_fft);
    res->fft_setups = spectral_calloc_array((size_t)n_threads, sizeof(FFTSetup));
    res->thread_real = spectral_calloc_array((size_t)n_threads, sizeof(float*));
    res->thread_imag = spectral_calloc_array((size_t)n_threads, sizeof(float*));
    res->thread_windowed = spectral_calloc_array((size_t)n_threads, sizeof(float*));
    res->thread_imag_sq = spectral_calloc_array((size_t)n_threads, sizeof(float*));

    if (!res->fft_setups || !res->thread_real || !res->thread_imag ||
        !res->thread_windowed || !res->thread_imag_sq) {
        goto fail;
    }

    {
        size_t n_fft_f32_bytes = 0;
        size_t n_freqs_f32_bytes = 0;
        if (!spectral_array_bytes(n_fft, sizeof(float), &n_fft_f32_bytes) ||
            !spectral_array_bytes(n_freqs, sizeof(float), &n_freqs_f32_bytes)) {
            goto fail;
        }

        for (int t = 0; t < n_threads; t++) {
            res->fft_setups[t] = vDSP_create_fftsetup(res->log2n, FFT_RADIX2);
            res->thread_real[t] = spectral_aligned_alloc(n_fft_f32_bytes);
            res->thread_imag[t] = spectral_aligned_alloc(n_fft_f32_bytes);
            res->thread_windowed[t] = spectral_aligned_alloc(n_fft_f32_bytes);
            res->thread_imag_sq[t] = spectral_aligned_alloc(n_freqs_f32_bytes);
            if (!res->fft_setups[t] || !res->thread_real[t] || !res->thread_imag[t] ||
                !res->thread_windowed[t] || !res->thread_imag_sq[t]) {
                goto fail;
            }
        }
    }
#else
    res->fft_plans = spectral_calloc_array((size_t)n_threads, sizeof(fftwf_plan));
    res->thread_in = spectral_calloc_array((size_t)n_threads, sizeof(float*));
    res->thread_out = spectral_calloc_array((size_t)n_threads, sizeof(fftwf_complex*));

    if (!res->fft_plans || !res->thread_in || !res->thread_out) {
        goto fail;
    }

    for (int t = 0; t < n_threads; t++) {
        res->thread_in[t] = fftwf_alloc_real(n_fft);
        res->thread_out[t] = fftwf_alloc_complex(n_freqs);
        if (!res->thread_in[t] || !res->thread_out[t]) {
            goto fail;
        }
        res->fft_plans[t] = fftwf_plan_dft_r2c_1d((int)n_fft, res->thread_in[t],
                                                  res->thread_out[t], FFTW_ESTIMATE);
        if (!res->fft_plans[t]) goto fail;
    }
#endif
    return 1;

fail:
    spectral_fft_resources_free(res);
    return 0;
}

void spectral_fft_resources_free(SpectralFftResources* res)
{
    if (!res) return;

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

    memset(res, 0, sizeof(*res));
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

    {
        float frame_max = 0.0f;
        spectral_fft_apply_magsq_scales(res, out_magsq, n_freqs, &frame_max);
        if (out_frame_max) {
            *out_frame_max = frame_max;
        }
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
    
    spectral_fft_apply_magsq_scales(res, out_magsq, n_freqs, &frame_max);

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
