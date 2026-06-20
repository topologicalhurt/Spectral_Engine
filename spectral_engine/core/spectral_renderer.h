/* spectral_renderer.h - Renderer abstraction (RENDERER_ABSTRACTION_PLAN.md, Stage 1)
 *
 * A *renderer* is a synthesis STRATEGY — what sound is produced — over the shared scene model
 * (a SegmentArray of tracked partials). It is distinct from two other axes:
 *   - DOMAIN: how it is executed — time-domain oscillator vs frequency-domain inverse-FFT
 *             (the OLA/FBS dual readings of the STFT).
 *   - DEVICE: on what silicon — CPU / Metal / CUDA (the existing SpectralBackendVTable).
 *
 * Supported renderers now: additive, wavetable, subtractive (FM/granular/modal/waveguide/
 * stochastic are catalogued in the plan as future).
 *
 * Stage 1 is the descriptor + registry layer: it names and classifies the renderers and their
 * capabilities, and maps a SpectralTimbre to its renderer. Production dispatch (the host synth
 * path) consumes the classification for a capability-decision log; the full router rewiring is
 * Stage 2. test_renderer_dispatch pins this contract.
 */
#ifndef SPECTRAL_RENDERER_H
#define SPECTRAL_RENDERER_H

#include "spectral_config.h"   /* SpectralTimbre, TIMBRE_* */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPECTRAL_RENDERER_ADDITIVE    = 0,  /* sine partials — one spectral lobe per partial */
    SPECTRAL_RENDERER_WAVETABLE   = 1,  /* a stored harmonic template placed per partial */
    SPECTRAL_RENDERER_SUBTRACTIVE = 2,  /* rich source waveform (+ a filter, plan Stage 3b) */
    SPECTRAL_RENDERER_COUNT       = 3
} SpectralRendererId;

/* Renderer-intrinsic capabilities (NOT domain/device properties — e.g. whether chirp is
 * honored is a property of the DOMAIN's eval, not of the renderer). */
typedef struct {
    unsigned spectral_native         : 1;  /* a per-frame inverse-FFT deposit exists for this renderer */
    unsigned deposits_harmonic_stack : 1;  /* deposits a harmonic series per partial (vs a single lobe) */
    unsigned needs_filter            : 1;  /* consumes a spectral transfer function H(f) (subtractive) */
} SpectralRendererCaps;

typedef struct {
    SpectralRendererId   id;
    const char*          name;
    SpectralRendererCaps caps;
} SpectralRenderer;

/* Registry lookup. Returns NULL for an out-of-range id. */
const SpectralRenderer* spectral_renderer_by_id(SpectralRendererId id);

/* Classify a timbre into its renderer: SINE -> additive; the rich analytic waveforms
 * (SAW/SQUARE/TRIANGLE/ASIN/PARABOLA/QUANTIZED/PWM) -> subtractive (the source half).
 * Wavetable is selected by the presence of a bank, not a timbre id, so it has no timbre
 * mapping. An out-of-range timbre returns additive (the safe default). */
SpectralRendererId spectral_renderer_for_timbre(SpectralTimbre timbre);

/* Convenience: the renderer's display name, or "unknown" for an out-of-range id. */
const char* spectral_renderer_name(SpectralRendererId id);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_RENDERER_H */
