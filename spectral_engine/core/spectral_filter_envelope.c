/* spectral_filter_envelope.c - subtractive filter as a per-band spectral envelope
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3b)
 *
 * Host/sim-side (guarded out of the real firmware build, like the other freq-domain renderer
 * units). Pure math — a per-partial gain lookup; no allocation, no I/O.
 */
#include "spectral_filter_envelope.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

float spectral_filter_envelope_gain(const SpectralFilterEnvelope* env, float freq_hz)
{
    int last;
    int i;

    if (!env || env->n_points <= 0) return 1.0f;            /* passthrough */
    if (env->n_points == 1) return env->gain[0];

    last = env->n_points - 1;
    if (freq_hz <= env->freq_hz[0]) return env->gain[0];    /* clamp below */
    if (freq_hz >= env->freq_hz[last]) return env->gain[last]; /* clamp above */

    /* find the bracketing segment [i, i+1] with freq_hz[i] <= freq_hz < freq_hz[i+1] */
    i = 0;
    while (i < last && env->freq_hz[i + 1] < freq_hz) i++;
    {
        float f0 = env->freq_hz[i];
        float f1 = env->freq_hz[i + 1];
        float g0 = env->gain[i];
        float g1 = env->gain[i + 1];
        float t = (f1 > f0) ? (freq_hz - f0) / (f1 - f0) : 0.0f;
        return g0 + t * (g1 - g0);
    }
}

void spectral_filter_envelope_apply(const SpectralFilterEnvelope* env,
                                    SpectralIfftPartial* partials, size_t n,
                                    float sr, size_t n_fft)
{
    float bin_to_hz;
    size_t i;

    if (!env || !partials || n_fft == 0u) return;

    bin_to_hz = sr / (float)n_fft;
    for (i = 0; i < n; i++) {
        float f = partials[i].bin * bin_to_hz;
        partials[i].amp *= spectral_filter_envelope_gain(env, f);
    }
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */
