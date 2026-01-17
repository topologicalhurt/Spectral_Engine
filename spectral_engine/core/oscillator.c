/* oscillator.c - Oscillator implementation */
#include "oscillator.h"
#include "oscillator_dispatch.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_utils.h"
#include <math.h>

static OscDispatchWord g_osc_dispatch = OSC_DISPATCH_ALL_SCALAR;

void osc_set_dispatch(OscDispatchWord dispatch) { g_osc_dispatch = dispatch; }
OscDispatchWord osc_get_dispatch(void) { return g_osc_dispatch; }

/* Canonical waveform generators - GPU backends must match these formulas */

static inline float osc_sine(float rads, float width) {
    (void)width;
    return fast_sin(rads + PI);
}

static inline float osc_saw(float rads, float width) {
    (void)width;
    return rads * -SPECTRAL_INV_PI;
}

static inline float osc_square(float rads, float width) {
    (void)width;
    return (rads > 0.0f) ? 1.0f : -1.0f;
}

static inline float osc_triangle(float rads, float width) {
    (void)width;
    return (1.0f - fabsf(rads) * SPECTRAL_TWO_INV_PI) * 2.0f - 1.0f;
}

static inline float osc_asin(float rads, float width) {
    (void)width;
    return asinf(rads * SPECTRAL_INV_PI);
}

static inline float osc_parabola(float rads, float width) {
    (void)width;
    return 1.0f - (rads * rads * SPECTRAL_INV_PI_SQ);
}

static inline float osc_quantized(float rads, float width) {
    if (width <= 0.0f) return 0.0f;
    float inv_w = 1.0f / width;  /* Caller should precompute if in hot loop */
    return (float)(int)(rads * width) * inv_w;
}

static inline float osc_pwm(float rads, float width) {
    return (width > 0.0f) ? (((rads + PI) * INV_TWO_PI < width) ? 1.0f : -1.0f) : 1.0f;
}

typedef float (*TimbreFunc)(float rads, float width);

static const TimbreFunc timbre_table[TIMBRE_COUNT] = {
    osc_sine,
    osc_saw,
    osc_square,
    osc_triangle,
    osc_asin,
    osc_parabola,
    osc_quantized,
    osc_pwm
};

float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width) {
    /* Bounds check - clamp invalid values to sine (safest default) */
    unsigned int idx = (unsigned int)timbre;
    if (idx >= TIMBRE_COUNT) {
        idx = TIMBRE_SINE;
    }
    
    /* Normalize phase to [-0.5, 0.5) then scale to [-pi, pi) */
    float norm = p * INV_TWO_PI;
    float rads = TWO_PI * (norm - (int)norm + (norm < 0.0f) - 0.5f);
    
    /* Branch-free dispatch through function pointer table */
    return a * timbre_table[idx](rads, width);
}

static inline float compute_phase(float phase0, float alpha, float beta, size_t j) {
    float jf = (float)j;
    return phase0 + jf * (alpha + beta * jf);
}

static void synth_segment_scalar(
    float* dst, size_t len, float phase0, float alpha, float beta,
    float amp0, float d_amp, float width, const FadeParams* fp, TimbreFunc osc_fn
) {
    for (size_t j = 0; j < len; j++) {
        float p = compute_phase(phase0, alpha, beta, j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, fp, len);
        dst[j] += amp * wave;
    }
}

void timbre_synth_segment(float* __restrict__ dst, const struct SegmentLoopParams* lp, SpectralTimbre timbre) {
    if ((unsigned int)timbre >= TIMBRE_COUNT) {
        SPECTRAL_WARN_ONCE(TIMBRE_COUNT, "Invalid timbre %d, using sine", (int)timbre);
        timbre = TIMBRE_SINE;
    }
    
    const size_t len = lp->length;
    if (len == 0) return;
    
    OscDispatchMode mode = OSC_GET_MODE(g_osc_dispatch, timbre);
    
    if (mode == OSC_MODE_FALLBACK) {
        mode = osc_simd_available(timbre) ? OSC_MODE_CPU_SIMD : OSC_MODE_CPU_SCALAR;
    }
    
    if (mode == OSC_MODE_CPU_SIMD && osc_simd_available(timbre)) {
        switch (timbre) {
        case TIMBRE_SINE:     osc_simd_segment_sine(dst, lp);     return;
        case TIMBRE_SAW:      osc_simd_segment_saw(dst, lp);      return;
        case TIMBRE_SQUARE:   osc_simd_segment_square(dst, lp);   return;
        case TIMBRE_TRIANGLE: osc_simd_segment_triangle(dst, lp); return;
        case TIMBRE_PARABOLA: osc_simd_segment_parabola(dst, lp); return;
        default: break;
        }
    }
    
    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    synth_segment_scalar(
        dst, len, lp->phase, lp->alpha, lp->beta,
        lp->amp, lp->d_amp, lp->width, &fp, timbre_table[timbre]
    );
}

/* Metal shader source */
#if defined(__APPLE__) && !defined(__CUDACC__)

const char* oscillator_metal_source = 
"#define TIMBRE_SINE     0\n"
"#define TIMBRE_SAW      1\n"
"#define TIMBRE_SQUARE   2\n"
"#define TIMBRE_TRIANGLE 3\n"
"#define TIMBRE_ASIN     4\n"
"#define TIMBRE_PARABOLA 5\n"
"\n"
"#define TWO_PI 6.283185307179586f\n"
"#define INV_TWO_PI 0.159154943091895f\n"
"#define INV_PI 0.318309886183791f\n"
"#define TWO_INV_PI 0.636619772367581f\n"
"#define INV_PI_SQ 0.101321183642338f\n"
"\n"
"inline float oscillator_fast_sin(float x) {\n"
"    x = x - TWO_PI * floor(x * INV_TWO_PI + 0.5f);\n"
"    float x2 = x * x;\n"
"    float num = x * (1.0f - x2 * (0.16605f - x2 * 0.00761f));\n"
"    float den = 1.0f + x2 * 0.00766f;\n"
"    return num / den;\n"
"}\n"
"\n"
"inline float oscillator_normalize_phase(float p) {\n"
"    float norm = p * INV_TWO_PI;\n"
"    return TWO_PI * (norm - floor(norm) - 0.5f);\n"
"}\n"
"\n"
"inline float oscillator(float phase, uint timbre) {\n"
"    float rads = oscillator_normalize_phase(phase);\n"
"    switch (timbre) {\n"
"        case TIMBRE_SINE:     return oscillator_fast_sin(phase);\n"
"        case TIMBRE_SAW:      return rads * -INV_PI;\n"
"        case TIMBRE_SQUARE:   return (rads > 0.0f) ? 1.0f : -1.0f;\n"
"        case TIMBRE_TRIANGLE: return (1.0f - abs(rads) * TWO_INV_PI) * 2.0f - 1.0f;\n"
"        case TIMBRE_PARABOLA: return 1.0f - rads * rads * INV_PI_SQ;\n"
"        default:              return oscillator_fast_sin(phase);\n"
"    }\n"
"}\n";

#endif /* __APPLE__ && !__CUDACC__ */
