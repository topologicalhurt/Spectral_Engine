/* spectral_analysis_fft.c
 *
 * Shared FFT resource management and frame transform helpers.
 */
#include "spectral_analysis_internal.h"
#include <limits.h>
#include <float.h>

static int spectral_fft_magsq_scale_valid(float scale)
{
    return spectral_is_finite_f32(scale) && scale > 0.0f;
}

static float spectral_fft_scaled_magsq(float value, float scale)
{
    double scaled = 0.0;

    if (!spectral_is_finite_f32(value) || value < 0.0f || !spectral_fft_magsq_scale_valid(scale)) {
        return 0.0f;
    }

    scaled = (double)value * (double)scale;
    if (!spectral_is_finite_f64(scaled) || scaled < 0.0 || scaled > (double)FLT_MAX) {
        return 0.0f;
    }

    return (float)scaled;
}

static int spectral_fft_single_frame_args_valid(const SpectralFftResources* res,
                                                int tid,
                                                const float* audio,
                                                int hop,
                                                const float* window_func,
                                                float* out_magsq)
{
    if (!res || tid < 0 || tid >= res->n_threads ||
        !audio || hop <= 0 || !window_func || !out_magsq ||
        res->n_fft == 0u || res->n_freqs == 0u) {
        return 0;
    }

#if SPECTRAL_USE_VDSP
    if (!res->fft_setups || !res->thread_real || !res->thread_imag ||
        !res->thread_windowed || !res->thread_imag_sq ||
        !res->fft_setups[tid] || !res->thread_real[tid] ||
        !res->thread_imag[tid] || !res->thread_windowed[tid] ||
        !res->thread_imag_sq[tid]) {
        return 0;
    }
#else
    if (!res->fft_plans || !res->thread_in || !res->thread_out ||
        !res->fft_plans[tid] || !res->thread_in[tid] || !res->thread_out[tid]) {
        return 0;
    }
#endif

    return 1;
}

static void spectral_fft_single_frame_clear(float* out_magsq,
                                            float* out_re,
                                            float* out_im,
                                            size_t n_freqs,
                                            float* out_frame_max)
{
    size_t bytes = 0;
    if (out_frame_max) *out_frame_max = 0.0f;
    if (n_freqs == 0u || !spectral_array_bytes(n_freqs, sizeof(float), &bytes)) {
        return;
    }
    if (out_magsq) memset(out_magsq, 0, bytes);
    if (out_re) memset(out_re, 0, bytes);
    if (out_im) memset(out_im, 0, bytes);
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
    magsq[0] = spectral_fft_scaled_magsq(magsq[0], endpoint_scale);
    if (n_freqs > 1u) {
        magsq[n_freqs - 1u] = spectral_fft_scaled_magsq(magsq[n_freqs - 1u], endpoint_scale);
    }
    /* Scale the interior bins and track the trackable frame max in the SAME pass:
     * spectral_fft_scaled_magsq always returns a finite value in [0, FLT_MAX], so the
     * running max needs no isfinite re-check and a separate max walk is redundant. */
    float interior_max = 0.0f;
    for (size_t i = 1u; i + 1u < n_freqs; i++) {
        float scaled = spectral_fft_scaled_magsq(magsq[i], positive_scale);
        magsq[i] = scaled;
        if (scaled > interior_max) interior_max = scaled;
    }
    if (frame_max) *frame_max = interior_max;
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
                               float* out_magsq, float* out_re, float* out_im,
                               float* out_frame_max)
{
    if (!spectral_fft_single_frame_args_valid(res, tid, audio, hop, window_func, out_magsq)) {
        spectral_fft_single_frame_clear(out_magsq, out_re, out_im, res ? res->n_freqs : 0u, out_frame_max);
        return;
    }

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

    /* vDSP_fft_zrip(FFT_FORWARD) emits the textbook DFT scaled by 2 (Apple's
     * packed-real convention, uniform across DC realp[0], Nyquist imagp[0] and
     * the interior bins). Undo it here so magnitudes match the textbook/FFTW
     * branch and the window amp scales (2/Σwindow), which are derived for
     * the unscaled DFT. Uniform on re+im, so phases are unaffected. */
    {
        float vdsp_half = 0.5f;
        vDSP_vsmul(split.realp, 1, &vdsp_half, split.realp, 1, n_fft / 2);
        vDSP_vsmul(split.imagp, 1, &vdsp_half, split.imagp, 1, n_fft / 2);
    }

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

        if (out_re && out_im) {
            /* Store the (0.5-scaled) complex spectrum so the tracker can take
             * atan2f only at peak bins. DC/Nyquist carry zero imaginary part,
             * so atan2f(0, re) reproduces the 0/PI phase sign special-case exactly. */
            out_re[0] = split.realp[0];           out_im[0] = 0.0f;
            out_re[n_freqs - 1] = split.imagp[0]; out_im[n_freqs - 1] = 0.0f;
            memcpy(out_re + 1, split.realp + 1, mid * sizeof(float));
            memcpy(out_im + 1, split.imagp + 1, mid * sizeof(float));
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

#else

void spectral_fft_single_frame(const SpectralFftResources* res,
                               int tid,
                               const float* audio, int hop,
                               const float* window_func,
                               size_t t,
                               float* out_magsq, float* out_re, float* out_im,
                               float* out_frame_max)
{
    if (!spectral_fft_single_frame_args_valid(res, tid, audio, hop, window_func, out_magsq)) {
        spectral_fft_single_frame_clear(out_magsq, out_re, out_im, res ? res->n_freqs : 0u, out_frame_max);
        return;
    }

    size_t n_fft = res->n_fft;
    size_t n_freqs = res->n_freqs;
    float* in_buf = res->thread_in[tid];
    fftwf_complex* out_buf = res->thread_out[tid];
    fftwf_plan plan = res->fft_plans[tid];

    const float* src = audio + t * hop;
    spectral_vmul(src, window_func, in_buf, n_fft);
    fftwf_execute(plan);

    float frame_max = 0.0f;
    /* Phase-at-peaks: the producer never computes dense phase; magsq drives the
     * peak scan and re/im (below) feed the per-peak atan2f in the tracker. */
    spectral_magsq_only((float*)out_buf, out_magsq, &frame_max, n_freqs);

    if (out_re && out_im) {
        /* fftwf r2c output is interleaved [re,im]; deinterleave so the tracker
         * can take atan2f at peak bins only. DC/Nyquist already carry im==0. */
        const float* cpx = (const float*)out_buf;
        for (size_t k = 0; k < n_freqs; k++) {
            out_re[k] = cpx[2u * k];
            out_im[k] = cpx[2u * k + 1u];
        }
    }

    spectral_fft_apply_magsq_scales(res, out_magsq, n_freqs, &frame_max);

    if (out_frame_max) {
        *out_frame_max = frame_max;
    }
}

#endif

float spectral_fft_frames(const SpectralFftResources* res,
                          const float* audio, int hop,
                          const float* window_func,
                          size_t frame_start, size_t frame_end,
                          float* out_magsq, float* out_re, float* out_im,
                          int magsq_only)
{
    float max_magsq = 0.0f;
    size_t local_n_frames = 0;
    size_t n_freqs = 0;
    size_t output_bins = 0;

    if (!res || !audio || !window_func || !out_magsq || hop <= 0) return 0.0f;
    if (res->n_threads < 1 || res->n_freqs == 0u || res->n_fft == 0u) return 0.0f;
    if (frame_end < frame_start) return 0.0f;
    if (!magsq_only && (!out_re || !out_im)) return 0.0f;

    local_n_frames = frame_end - frame_start;
    n_freqs = res->n_freqs;
    if (local_n_frames == 0u) return 0.0f;
    if (!spectral_size_mul(local_n_frames, n_freqs, &output_bins)) return 0.0f;
    (void)output_bins;

    #pragma omp parallel reduction(max:max_magsq) num_threads(res->n_threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < local_n_frames; i++) {
            size_t t = frame_start + i;
            float frame_max = 0.0f;
            spectral_fft_single_frame(res, tid, audio, hop, window_func, t,
                                      out_magsq + i * n_freqs,
                                      (!magsq_only && out_re) ? (out_re + i * n_freqs) : NULL,
                                      (!magsq_only && out_im) ? (out_im + i * n_freqs) : NULL,
                                      &frame_max);
            if (frame_max > max_magsq) max_magsq = frame_max;
        }
    }

    return max_magsq;
}

