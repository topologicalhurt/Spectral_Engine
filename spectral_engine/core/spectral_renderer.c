/* spectral_renderer.c - Renderer registry (RENDERER_ABSTRACTION_PLAN.md, Stage 1)
 *
 * Host-side classification metadata only — guarded out of the real firmware build (the M7 synth
 * path renders directly via spectral_arm32_process and has no use for renderer classification).
 * The tests_all targets are host/simulation builds (SPECTRAL_IS_EMBEDDED_SIM), so the body
 * compiles there and is exercised by test_renderer_dispatch.
 */
#include "spectral_renderer.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

/* caps: {spectral_native, deposits_harmonic_stack, needs_filter}.
 * All three renderers are spectral-native (a per-frame deposit exists). Additive deposits a
 * single lobe per partial; wavetable and subtractive deposit a harmonic stack. Only subtractive
 * needs a spectral filter H(f) (its source half exists today; the filter is plan Stage 3b). */
static const SpectralRenderer k_renderers[SPECTRAL_RENDERER_COUNT] = {
    [SPECTRAL_RENDERER_ADDITIVE]    = { SPECTRAL_RENDERER_ADDITIVE,    "additive",
                                        { 1u, 0u, 0u } },
    [SPECTRAL_RENDERER_WAVETABLE]   = { SPECTRAL_RENDERER_WAVETABLE,   "wavetable",
                                        { 1u, 1u, 0u } },
    [SPECTRAL_RENDERER_SUBTRACTIVE] = { SPECTRAL_RENDERER_SUBTRACTIVE, "subtractive",
                                        { 1u, 1u, 1u } },
};

const SpectralRenderer* spectral_renderer_by_id(SpectralRendererId id)
{
    if ((unsigned)id >= (unsigned)SPECTRAL_RENDERER_COUNT) return NULL;
    return &k_renderers[id];
}

SpectralRendererId spectral_renderer_for_timbre(SpectralTimbre timbre)
{
    switch (timbre) {
        case TIMBRE_SINE:
            return SPECTRAL_RENDERER_ADDITIVE;
        case TIMBRE_SAW:
        case TIMBRE_SQUARE:
        case TIMBRE_TRIANGLE:
        case TIMBRE_ASIN:
        case TIMBRE_PARABOLA:
        case TIMBRE_QUANTIZED:
        case TIMBRE_PWM:
            return SPECTRAL_RENDERER_SUBTRACTIVE;
        default:
            return SPECTRAL_RENDERER_ADDITIVE;
    }
}

const char* spectral_renderer_name(SpectralRendererId id)
{
    const SpectralRenderer* r = spectral_renderer_by_id(id);
    return r ? r->name : "unknown";
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */
