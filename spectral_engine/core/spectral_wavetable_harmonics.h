/* spectral_wavetable_harmonics.h - frequency-domain bridge for wavetables
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3a)
 *
 * The wavetable renderer is spectral-native: a single-cycle table IS a harmonic spectrum. This
 * unit bridges the EXISTING time-domain wavetable (raw samples read by spectral_wavetable_lookup)
 * to the frequency-domain (IFFT) domain WITHOUT a second representation — it extracts the table's
 * harmonic series {amp_k, phase_k} by DFT of its own samples, then expands one wavetable partial
 * into its harmonics as additive SpectralIfftPartials. So wavetable freq-domain rendering reuses
 * the existing additive IFFT deposit (spectral_ifft_synth_render) rather than forking it, and is
 * consistent-by-construction with the time-domain read. test_wavetable_harmonics pins the math.
 *
 * Convention: table(theta) = sum_k amp_k * cos(2*pi*k*theta + phase_k), theta in [0,1).
 */
#ifndef SPECTRAL_WAVETABLE_HARMONICS_H
#define SPECTRAL_WAVETABLE_HARMONICS_H

#include "spectral_wavetable.h"   /* SpectralWavetable, SPECTRAL_WAVETABLE_SIZE */
#include "spectral_synth_ifft.h"  /* SpectralIfftPartial */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Extract the harmonic spectrum of a single-cycle table by DFT of its samples.
 * Writes amp_k[k] and phase_k[k] for k in [0, min(max_k, SPECTRAL_WAVETABLE_SIZE/2 - 1)];
 * k=0 is DC (phase 0). Both arrays must hold at least that many entries. Returns the harmonic
 * count written (kmax+1), or 0 on a bad argument. */
size_t spectral_wavetable_harmonics(const SpectralWavetable* table,
                                    float* amp_k, float* phase_k, size_t max_k);

/* Expand one wavetable partial into its harmonics as additive IFFT partials. The fundamental is
 * at fractional bin f0_bin (= omega*n_fft/(2*pi)); the partial has linear amplitude amp and phase
 * phase0 (radians). Harmonic k (k>=1) is deposited at bin k*f0_bin with amplitude amp*amp_k[k]
 * and phase k*phase0 + phase_k[k], for as long as k*f0_bin < n_fft/2 (Nyquist truncation =
 * automatic band-limiting). DC (k=0) is not deposited (bin 0 is outside the renderer domain).
 * Returns the number of partials written (<= cap). */
size_t spectral_wavetable_expand(const float* amp_k, const float* phase_k, size_t n_harm,
                                 float f0_bin, float amp, float phase0, size_t n_fft,
                                 SpectralIfftPartial* out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_WAVETABLE_HARMONICS_H */
