/* oscillator.c - Oscillator implementation */
#include "oscillator.h"
#include "oscillator_dispatch.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"
#include "spectral_utils.h"
#include <math.h>

static OscDispatchWord g_osc_dispatch = OSC_DISPATCH_ALL_SCALAR;

void osc_set_dispatch(OscDispatchWord dispatch) { g_osc_dispatch = dispatch; }
OscDispatchWord osc_get_dispatch(void) { return g_osc_dispatch; }

typedef float (*TimbreFunc)(float rads, float width);

static const TimbreFunc timbre_table[TIMBRE_COUNT] = {
    spectral_osc_sine,
    spectral_osc_saw,
    spectral_osc_square,
    spectral_osc_triangle,
    spectral_osc_asin,
    spectral_osc_parabola,
    spectral_osc_quantized,
    spectral_osc_pwm
};

float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width) {
    unsigned int idx = (unsigned int)timbre;
    if (idx >= TIMBRE_COUNT) {
        idx = TIMBRE_SINE;
    }

    /* Canonical phase normalization — must match spectral_normalize_phase() */
    float rads = spectral_normalize_phase(p);

    return a * timbre_table[idx](rads, width);
}

static void synth_segment_scalar(
    float* dst, size_t len, float phase0, float alpha, float beta,
    float amp0, float d_amp, float width, const FadeParams* fp, TimbreFunc osc_fn
) {
    size_t fade_in_end = fp->fade_len;
    size_t fade_out_start = fp->fade_out_start;

    /* Fade-in region */
    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float p = spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_in(j, fp->inv_fade);
        dst[j] += amp * wave;
    }

    /* Sustain region (envelope = 1.0, no branching) */
    for (size_t j = fade_in_end; j < fade_out_start && j < len; j++) {
        float p = spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j);
        dst[j] += amp * wave;
    }

    /* Fade-out region */
    for (size_t j = (fade_out_start > fade_in_end ? fade_out_start : fade_in_end); j < len; j++) {
        float p = spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_out(j, len, fp->inv_fade);
        dst[j] += amp * wave;
    }
}

void timbre_synth_segment(float* __restrict__ dst, const struct SegmentLoopParams* lp, SpectralTimbre timbre) {
    if ((unsigned int)timbre >= TIMBRE_COUNT) {
        char resolution[384] = {0};
        SpectralResolutionContext context = {
            .event = SPECTRAL_RESOLUTION_EVENT_FALLBACK,
            .backend = SPECTRAL_RESOLUTION_BACKEND_CPU,
            .scope = SPECTRAL_RESOLUTION_SCOPE_TIMBRE,
            .requested_label = timbre_name(timbre),
            .requested_id = (int)timbre,
            .effective_label = timbre_name(TIMBRE_SINE),
            .effective_id = (int)TIMBRE_SINE,
            .max_supported_id = (int)(TIMBRE_COUNT - 1),
            .mode = spectral_exec_mode_name(),
            .reason = SPECTRAL_RESOLUTION_REASON_INVALID_TIMBRE_ID
        };
        spectral_format_resolution_context(resolution, sizeof(resolution), &context);
        SPECTRAL_WARN_ONCE(TIMBRE_COUNT, "%s", resolution);
        timbre = TIMBRE_SINE;
    }
    
    const size_t len = lp->length;
    if (len == 0) return;
    
    OscDispatchMode dispatch_mode = OSC_GET_MODE(g_osc_dispatch, timbre);
    
    if (dispatch_mode == OSC_MODE_FALLBACK) {
        dispatch_mode = osc_simd_available(timbre) ? OSC_MODE_CPU_SIMD : OSC_MODE_CPU_SCALAR;
    }
    
    if (dispatch_mode == OSC_MODE_CPU_SIMD && osc_simd_available(timbre)) {
        switch (timbre) {
        case TIMBRE_SINE:      osc_simd_segment_sine(dst, lp);      return;
        case TIMBRE_SAW:       osc_simd_segment_saw(dst, lp);       return;
        case TIMBRE_SQUARE:    osc_simd_segment_square(dst, lp);    return;
        case TIMBRE_TRIANGLE:  osc_simd_segment_triangle(dst, lp);  return;
        case TIMBRE_PARABOLA:  osc_simd_segment_parabola(dst, lp);  return;
        case TIMBRE_QUANTIZED: osc_simd_segment_quantized(dst, lp); return;
        case TIMBRE_PWM:       osc_simd_segment_pwm(dst, lp);       return;
        default: break;
        }
    }
    
    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    synth_segment_scalar(
        dst, len, lp->phase, lp->alpha, lp->beta,
        lp->amp, lp->d_amp, lp->width, &fp, timbre_table[timbre]
    );
}

/* Metal shader source */
#if defined(__APPLE__) && !defined(__CUDACC__)

/* All constants injected from spectral_consts.h via METAL_CONST_* macros
 * defined in spectral_osc_formulas.h. Formulas must match the canonical
 * C implementations in spectral_osc_formulas.h exactly. */
_Static_assert(SPECTRAL_OSC_FORMULAS_VERSION == 4,
    "oscillator_metal_source MSL strings are stale — mirror formula changes and bump version");
const char* oscillator_metal_source =
"#define TIMBRE_SINE     0\n"
"#define TIMBRE_SAW      1\n"
"#define TIMBRE_SQUARE   2\n"
"#define TIMBRE_TRIANGLE 3\n"
"#define TIMBRE_ASIN     4\n"
"#define TIMBRE_PARABOLA 5\n"
"\n"
"#define TWO_PI " SPECTRAL_STR(SPECTRAL_TWO_PI) "\n"
"#define INV_TWO_PI " SPECTRAL_STR(SPECTRAL_INV_TWO_PI) "\n"
"#define INV_PI " SPECTRAL_STR(SPECTRAL_INV_PI) "\n"
"#define INV_PI_SQ " SPECTRAL_STR(SPECTRAL_INV_PI_SQ) "\n"
"#define PI " SPECTRAL_STR(SPECTRAL_PI) "\n"
"\n"
"/* Must match spectral_fast_sin_inline() in spectral_osc_formulas.h */\n"
"inline float oscillator_fast_sin(float x) {\n"
"    return sin(x);\n"
"}\n"
"\n"
"/* Must match spectral_normalize_phase() in spectral_osc_formulas.h */\n"
"inline float oscillator_normalize_phase(float p) {\n"
"    return p - TWO_PI * floor(p * INV_TWO_PI + 0.5f);\n"
"}\n"
"\n"
"#define FADE_SAMPLES " SPECTRAL_STR(SPECTRAL_FADE_SAMPLES_DESKTOP) "\n"
"\n"
"/* Must match spectral_fade_envelope_gpu() in spectral_osc_formulas.h */\n"
"inline float fade_envelope(float j, float seg_len) {\n"
"    float fade_len = min(seg_len * 0.25f, (float)FADE_SAMPLES);\n"
"    if (fade_len < 1.0f) fade_len = 1.0f;\n"
"    float inv_fade = 1.0f / fade_len;\n"
"    if (j < fade_len) {\n"
"        return 0.5f * (1.0f + oscillator_fast_sin((j * inv_fade - 0.5f) * PI));\n"
"    }\n"
"    float from_end = seg_len - 1.0f - j;\n"
"    if (from_end < fade_len) {\n"
"        return 0.5f * (1.0f + oscillator_fast_sin((from_end * inv_fade - 0.5f) * PI));\n"
"    }\n"
"    return 1.0f;\n"
"}\n"
"\n"
"/* Must match spectral_osc_* waveforms in spectral_osc_formulas.h */\n"
"inline float oscillator(float phase, uint timbre) {\n"
"    float rads = oscillator_normalize_phase(phase);\n"
"    switch (timbre) {\n"
"        case TIMBRE_SINE:     return oscillator_fast_sin(rads);\n"
"        case TIMBRE_SAW:      return rads * -INV_PI;\n"
"        case TIMBRE_SQUARE:   return (rads > 0.0f) ? 1.0f : -1.0f;\n"
"        case TIMBRE_TRIANGLE: return (1.0f - abs(rads) * INV_PI) * 2.0f - 1.0f;\n"
"        case TIMBRE_ASIN:     return asin(rads * INV_PI);\n"
"        case TIMBRE_PARABOLA: return 1.0f - rads * rads * INV_PI_SQ;\n"
"        default:              return oscillator_fast_sin(rads);\n"
"    }\n"
"}\n";

#endif /* __APPLE__ && !__CUDACC__ */
