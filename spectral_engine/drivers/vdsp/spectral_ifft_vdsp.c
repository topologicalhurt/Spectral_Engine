/* spectral_ifft_vdsp.c - host iFFT port: Apple vDSP (SPECTRAL_USE_VDSP).
 *
 * Implements the spectral_synth_ifft.h backend contract: textbook inverse
 * real DFT of the packed half-spectrum. vDSP scaling: zrip(FORWARD) emits
 * 2x the textbook DFT and the forward/inverse round trip is 2N x identity,
 * so zrip(INVERSE) applied to a textbook spectrum returns N x the textbook
 * inverse — undone here with a single vsmul (1/N). The packed layout
 * (DC in realp[0], Nyquist in imagp[0]) matches the contract directly.
 * Cross-checked against the reference port's math by the ifft_synth_parity
 * ctest (whichever port is live must hit the F1-measured floors).
 */
#include "spectral_config.h"

#if SPECTRAL_USE_VDSP

#include <Accelerate/Accelerate.h>
#include <stdlib.h>

#include "spectral_synth_ifft.h"

struct SpectralIfftBackend {
    FFTSetup setup;
    vDSP_Length log2n;
    size_t n_fft;
};

SpectralIfftBackend* spectral_ifft_backend_create(size_t n_fft) {
    SpectralIfftBackend* b = (SpectralIfftBackend*)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->n_fft = n_fft;
    b->log2n = 0;
    while (((size_t)1 << b->log2n) < n_fft) b->log2n++;
    b->setup = vDSP_create_fftsetup(b->log2n, FFT_RADIX2);
    if (!b->setup) {
        free(b);
        return NULL;
    }
    return b;
}

void spectral_ifft_backend_destroy(SpectralIfftBackend* b) {
    if (!b) return;
    if (b->setup) vDSP_destroy_fftsetup(b->setup);
    free(b);
}

void spectral_ifft_backend_inverse(SpectralIfftBackend* b,
                                   float* re, float* im, float* out) {
    DSPSplitComplex split = { re, im };
    vDSP_fft_zrip(b->setup, &split, 1, b->log2n, FFT_INVERSE);
    /* Split-complex back to interleaved-real, then undo the N scale. */
    vDSP_ztoc(&split, 1, (DSPComplex*)out, 2, b->n_fft / 2);
    {
        float inv_n = 1.0f / (float)b->n_fft;
        vDSP_vsmul(out, 1, &inv_n, out, 1, b->n_fft);
    }
}

#endif /* SPECTRAL_USE_VDSP */
