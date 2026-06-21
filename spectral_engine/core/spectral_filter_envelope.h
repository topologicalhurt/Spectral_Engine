/* spectral_filter_envelope.h - subtractive filter as a per-band spectral envelope
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3b)
 *
 * The subtractive renderer is source + filter. In the frequency domain the filter is FREE: by the
 * convolution theorem, time-domain filtering is a per-bin spectral multiplication. This unit is
 * the filter half — a piecewise-linear magnitude envelope H(f) over frequency that scales each
 * partial's amplitude by H at its frequency. It is source-agnostic: apply it to wavetable-expanded
 * harmonics, additive partials, or any SpectralIfftPartial set ("subtractive on top of any source").
 *
 * The source half is the existing rich timbres (bandlimited saw/square/PWM) or a wavetable. This
 * fills the subtractive renderer's needs_filter capability (see spectral_renderer.h).
 */
#ifndef SPECTRAL_FILTER_ENVELOPE_H
#define SPECTRAL_FILTER_ENVELOPE_H

#include "spectral_synth_ifft.h"  /* SpectralIfftPartial */
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPECTRAL_FILTER_ENVELOPE_MAX_POINTS 32

/* A piecewise-linear magnitude response: gain[i] (linear) at ascending breakpoint freq_hz[i].
 * Outside [freq_hz[0], freq_hz[n_points-1]] the endpoint gains are held (clamped). */
typedef struct {
    float freq_hz[SPECTRAL_FILTER_ENVELOPE_MAX_POINTS];  /* ascending breakpoint frequencies (Hz) */
    float gain[SPECTRAL_FILTER_ENVELOPE_MAX_POINTS];     /* linear magnitude gain at each breakpoint */
    int   n_points;
} SpectralFilterEnvelope;

/* Linear-interpolated magnitude gain at freq_hz. Clamps to the endpoint gains outside the
 * breakpoint range. Returns 1.0 (passthrough) for an empty/NULL envelope. */
float spectral_filter_envelope_gain(const SpectralFilterEnvelope* env, float freq_hz);

/* Apply the envelope to additive partials in place: each partial's amplitude is multiplied by the
 * envelope gain at its frequency (freq = bin * sr / n_fft). This IS subtractive filtering, realized
 * as spectral multiplication. No-op for a NULL envelope/partials or n_fft == 0. */
void spectral_filter_envelope_apply(const SpectralFilterEnvelope* env,
                                    SpectralIfftPartial* partials, size_t n,
                                    float sr, size_t n_fft);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_FILTER_ENVELOPE_H */
