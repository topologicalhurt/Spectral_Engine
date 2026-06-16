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
typedef struct SpectralIfftBackend SpectralIfftBackend;   /* port contract, defined below */

/* n_fft: power of two >= 64 (frame length; hop = n_fft/2). NULL on failure.
 * create() malloc-allocates (host only); the embedded build uses init() below. */
SpectralIfftSynth* spectral_ifft_synth_create(size_t n_fft);
void spectral_ifft_synth_destroy(SpectralIfftSynth* s);

/* Static-allocation init for the libc-free embedded build (no malloc). Bytes needed
 * for the struct + all scratch (NOT the FFT backend — pass that in, statically made
 * by the per-port backend). pool must be >= pool_bytes and 16-byte aligned; the
 * caller owns the pool + backend for the synth's lifetime and destroy() is a no-op. */
size_t spectral_ifft_synth_pool_bytes(size_t n_fft);
SpectralIfftSynth* spectral_ifft_synth_init(void* pool, size_t pool_bytes,
                                            size_t n_fft, SpectralIfftBackend* backend);

size_t spectral_ifft_synth_n_fft(const SpectralIfftSynth* s);

/* Render `total` samples of the stationary partial sum, overlap-added
 * internally (edges fully covered; output valid from sample 0).
 * Returns 0 on success, nonzero on bad args. */
int spectral_ifft_synth_render(SpectralIfftSynth* s,
                               const SpectralIfftPartial* partials, size_t n,
                               float* out, size_t total);

/* Per-frame partial provider for the dynamic render path: fill `out` (capacity
 * `out_cap`) with the partials live in the frame centered at `frame_center`
 * (output samples), returning the count. Partials outside the bin domain are
 * dropped by the renderer, so the callback may emit optimistically. */
typedef size_t (*SpectralIfftFramePartialsFn)(void* ctx, double frame_center,
                                              SpectralIfftPartial* out, size_t out_cap);

/* Render `total` samples from a TIME-VARYING partial set: `fn` is queried once
 * per frame for the partials live in that frame (the F2b hybrid router maps the
 * active stationary-sine segment interiors to partials at frame center). OLA and
 * edge coverage identical to the static path; a constant `fn` reproduces it
 * exactly. Returns 0 on success, nonzero on bad args. */
int spectral_ifft_synth_render_dynamic(SpectralIfftSynth* s,
                                       SpectralIfftFramePartialsFn fn, void* ctx,
                                       float* out, size_t total);

/* --- port contract (implemented by exactly one host TU per build) -------- */

SpectralIfftBackend* spectral_ifft_backend_create(size_t n_fft);
void spectral_ifft_backend_destroy(SpectralIfftBackend* b);

/* The n_fft the backend was built for (0 if NULL). spectral_ifft_synth_init uses it
 * to reject a backend/synth size mismatch that would otherwise read/write past the
 * synth's re/im/frame scratch. */
size_t spectral_ifft_backend_n_fft(const SpectralIfftBackend* b);

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
