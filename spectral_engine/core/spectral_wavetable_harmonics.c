/* spectral_wavetable_harmonics.c - frequency-domain bridge for wavetables
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3a)
 *
 * Host/sim-side analysis only (guarded out of the real firmware build). The DFT is a one-time
 * per-table extraction; runtime rendering reuses the existing additive IFFT deposit.
 */
#include "spectral_wavetable_harmonics.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

#include <math.h>

static const double SPECTRAL_WT_TWO_PI = 6.283185307179586476925286766559;

size_t spectral_wavetable_harmonics(const SpectralWavetable* table,
                                    float* amp_k, float* phase_k, size_t max_k)
{
    const size_t N = (size_t)SPECTRAL_WAVETABLE_SIZE;
    size_t kmax;
    size_t k;

    if (!table || !amp_k || !phase_k) return 0;

    /* Cap below the table Nyquist bin (N/2): the (2/N) amplitude scaling below is the
     * interior-harmonic form, not the unpaired Nyquist endpoint. */
    kmax = max_k;
    if (kmax > N / 2u - 1u) kmax = N / 2u - 1u;

    for (k = 0; k <= kmax; k++) {
        double re = 0.0;
        double im = 0.0;
        double w = SPECTRAL_WT_TWO_PI * (double)k / (double)N;
        size_t n;
        for (n = 0; n < N; n++) {
            double ang = w * (double)n;
            double s = (double)table->samples[n];
            re += s * cos(ang);
            im -= s * sin(ang);   /* X[k] = sum_n s[n] e^{-i ang} */
        }
        if (k == 0u) {
            amp_k[0] = (float)(re / (double)N);   /* DC (im is ~0 for a real table) */
            phase_k[0] = 0.0f;
        } else {
            amp_k[k] = (float)(2.0 * sqrt(re * re + im * im) / (double)N);
            phase_k[k] = (float)atan2(im, re);
        }
    }
    return kmax + 1u;
}

size_t spectral_wavetable_expand(const float* amp_k, const float* phase_k, size_t n_harm,
                                 float f0_bin, float amp, float phase0, size_t n_fft,
                                 SpectralIfftPartial* out, size_t cap)
{
    const size_t nyquist_bin_i = n_fft / 2u;   /* integer bin n_fft/2 (exact for power-of-two n_fft) */
    const float nyquist_bin = (float)nyquist_bin_i;
    size_t count = 0;
    size_t k;

    if (!amp_k || !phase_k || !out || f0_bin <= 0.0f) return 0;

    for (k = 1; k < n_harm && count < cap; k++) {
        float bin = (float)k * f0_bin;
        if (bin >= nyquist_bin) break;   /* Nyquist truncation = band-limiting */
        out[count].bin = bin;
        out[count].amp = amp * amp_k[k];
        out[count].phase0 = (float)k * phase0 + phase_k[k];
        count++;
    }
    return count;
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */
