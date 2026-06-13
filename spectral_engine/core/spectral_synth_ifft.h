/* spectral_synth_ifft.h - IFFT (Rodet-Depalle) synthesis contract.
 *
 * F-stream F2 (IFFT_SYNTHESIS_PLAN.md). The frame-builder math is the F1
 * formulation validated in tests/core_math/harnesses/ifft_synth_sweep.c:
 * a real, circularly-centered motif placed at each partial's fractional bin
 * with the (-1)^k centering twiddle; periodic-Hann OLA at 50% (exact COLA).
 * Approximation floors are MEASURED and gated (frame -55 dBFS @ K=8 taps,
 * stream RMS -83 dBFS; see the plan's error budget). This path is OPT-IN:
 * the oscillator bank remains the exact default until golden sign-off (F3).
 *
 * Layering (one implementation TU per port contract, selected by the build):
 *   - This header is the ONLY surface; no third-party types leak through.
 *   - The inverse-FFT primitive is a port contract implemented by exactly
 *     one TU per build: drivers/vdsp/spectral_ifft_vdsp.c (Apple) or
 *     arch/ref/spectral_ifft_ref.c (portable radix-2 fallback).
 *     The ARM Q31 port (port/cmsis/) lands at F4.
 */
#ifndef SPECTRAL_SYNTH_IFFT_H
#define SPECTRAL_SYNTH_IFFT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One stationary partial for frame rendering. */
typedef struct {
    float bin;       /* fractional bin = omega * n_fft / (2*pi), in (1, n_fft/2-1) */
    float amp;       /* linear amplitude of the cosine */
    float phase0;    /* phase (radians) at output sample t = 0 */
} SpectralIfftPartial;

typedef struct SpectralIfftSynth SpectralIfftSynth;

/* n_fft: power of two >= 64 (frame length; hop = n_fft/2). NULL on failure. */
SpectralIfftSynth* spectral_ifft_synth_create(size_t n_fft);
void spectral_ifft_synth_destroy(SpectralIfftSynth* s);

size_t spectral_ifft_synth_n_fft(const SpectralIfftSynth* s);

/* Render `total` samples of the stationary partial sum, overlap-added
 * internally (edges fully covered; output valid from sample 0).
 * Returns 0 on success, nonzero on bad args. */
int spectral_ifft_synth_render(SpectralIfftSynth* s,
                               const SpectralIfftPartial* partials, size_t n,
                               float* out, size_t total);

/* --- port contract (implemented by exactly one host TU per build) -------- */

typedef struct SpectralIfftBackend SpectralIfftBackend;

SpectralIfftBackend* spectral_ifft_backend_create(size_t n_fft);
void spectral_ifft_backend_destroy(SpectralIfftBackend* b);

/* Textbook inverse real DFT of the packed half-spectrum:
 *   re/im[k] = F[k] for k = 1..n_fft/2-1; re[0] = F[0] (DC, real);
 *   im[0] = F[n_fft/2] (Nyquist, real).
 * out[n] = (1/N) * sum_k F[k] e^{+j 2 pi k n / N}  (real, length n_fft).
 * Each port owns its library's scaling so the contract stays textbook.
 * MUTATES re/im (scratch). */
void spectral_ifft_backend_inverse(SpectralIfftBackend* b,
                                   float* re, float* im, float* out);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SYNTH_IFFT_H */
