/* spectral_ifft_ref.c - host iFFT port: portable radix-2 (no vDSP builds).
 *
 * Implements the spectral_synth_ifft.h backend contract on any platform: the
 * packed half-spectrum is expanded to the full Hermitian complex spectrum and
 * run through an in-place iterative radix-2 inverse FFT (double twiddles
 * computed at create() into float tables; bit-reversal precomputed). This is
 * the correctness fallback, not the speed path — an FFTW port can join it
 * behind the same contract when a Linux benchmark asks for it.
 */
#include "spectral_config.h"

#if !SPECTRAL_USE_VDSP

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_consts.h"
#include "spectral_synth_ifft.h"

struct SpectralIfftBackend {
    size_t n_fft;
    unsigned log2n;
    unsigned* bitrev;       /* n_fft */
    float* tw_re;           /* n_fft/2 twiddles e^{+j 2 pi k / N} (inverse) */
    float* tw_im;
    float* cre;             /* full complex scratch, n_fft */
    float* cim;
};

SpectralIfftBackend* spectral_ifft_backend_create(size_t n_fft) {
    SpectralIfftBackend* b = (SpectralIfftBackend*)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->n_fft = n_fft;
    b->log2n = 0;
    while (((size_t)1 << b->log2n) < n_fft) b->log2n++;

    b->bitrev = (unsigned*)malloc(n_fft * sizeof(unsigned));
    b->tw_re = (float*)malloc((n_fft / 2) * sizeof(float));
    b->tw_im = (float*)malloc((n_fft / 2) * sizeof(float));
    b->cre = (float*)malloc(n_fft * sizeof(float));
    b->cim = (float*)malloc(n_fft * sizeof(float));
    if (!b->bitrev || !b->tw_re || !b->tw_im || !b->cre || !b->cim) {
        spectral_ifft_backend_destroy(b);
        return NULL;
    }
    for (size_t i = 0; i < n_fft; i++) {
        unsigned r = 0;
        for (unsigned bit = 0; bit < b->log2n; bit++) {
            r = (r << 1) | (unsigned)((i >> bit) & 1u);
        }
        b->bitrev[i] = r;
    }
    for (size_t k = 0; k < n_fft / 2; k++) {
        double a = 2.0 * SPECTRAL_PI_D * (double)k / (double)n_fft;
        b->tw_re[k] = (float)cos(a);
        b->tw_im[k] = (float)sin(a);    /* +j: inverse transform */
    }
    return b;
}

void spectral_ifft_backend_destroy(SpectralIfftBackend* b) {
    if (!b) return;
    free(b->bitrev);
    free(b->tw_re);
    free(b->tw_im);
    free(b->cre);
    free(b->cim);
    free(b);
}

void spectral_ifft_backend_inverse(SpectralIfftBackend* b,
                                   float* re, float* im, float* out) {
    size_t n = b->n_fft;
    size_t half = n / 2;

    /* Expand packed half-spectrum to the full Hermitian spectrum,
     * bit-reversal applied on the way in. */
    for (size_t i = 0; i < n; i++) {
        b->cre[i] = 0.0f;
        b->cim[i] = 0.0f;
    }
    {
        unsigned r0 = b->bitrev[0];
        b->cre[r0] = re[0];                   /* DC (real) */
        unsigned rn = b->bitrev[half];
        b->cre[rn] = im[0];                   /* Nyquist (real, packed in im[0]) */
    }
    for (size_t k = 1; k < half; k++) {
        unsigned rk = b->bitrev[k];
        unsigned rm = b->bitrev[n - k];
        b->cre[rk] = re[k];
        b->cim[rk] = im[k];
        b->cre[rm] = re[k];                   /* conjugate mirror */
        b->cim[rm] = -im[k];
    }

    /* Iterative radix-2 DIT with +j twiddles (inverse), then 1/N. */
    for (size_t len = 2; len <= n; len <<= 1) {
        size_t half_len = len / 2;
        size_t step = n / len;
        for (size_t base = 0; base < n; base += len) {
            for (size_t j = 0; j < half_len; j++) {
                size_t a = base + j;
                size_t c = a + half_len;
                float wr = b->tw_re[j * step];
                float wi = b->tw_im[j * step];
                float xr = b->cre[c] * wr - b->cim[c] * wi;
                float xi = b->cre[c] * wi + b->cim[c] * wr;
                float ar = b->cre[a], ai = b->cim[a];
                b->cre[a] = ar + xr;
                b->cim[a] = ai + xi;
                b->cre[c] = ar - xr;
                b->cim[c] = ai - xi;
            }
        }
    }
    {
        float inv_n = 1.0f / (float)n;
        for (size_t i = 0; i < n; i++) out[i] = b->cre[i] * inv_n;
    }
}

#endif /* !SPECTRAL_USE_VDSP */
