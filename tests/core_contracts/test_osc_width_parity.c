/* test_osc_width_parity.c - SIMD width-parameterization parity (Q2).
 *
 * Q2 of QTYPE_DOMAIN_PLAN.md width-parameterizes the float L1 sustain kernel
 * (oscillator_simd_kernel.inc) over SIMDE_NATURAL_FLOAT_VECTOR_SIZE: 4-wide
 * __m128 on SSE2/NEON, 8-wide __m256 on an AVX2 x86 target. Production picks ONE
 * width per build, so a single-build test can never compare the two. This test
 * force-instantiates BOTH widths in one translation unit (SIMDe provides a
 * portable __m256 on any host, including Apple Silicon NEON), then asserts:
 *
 *     8-wide  ==  4-wide  ==  scalar      (within the FMA-contraction budget)
 *
 * That pins the plan's Q2 contract: widening the register does not change the
 * synthesized signal beyond the same <=1 ULP class already accepted for SIMD-vs-
 * scalar at PASS200. The 8-wide path is NOT bit-identical to the 4-wide path -
 * the SIMD/scalar-tail boundary moves from a multiple of 4 to a multiple of 8,
 * so a few boundary samples per segment shift between the vector and scalar
 * lanes - but the divergence stays within budget, which is the guarantee.
 *
 * The scalar reference is the REAL production scalar oscillator
 * (timbre_synth_segment under OSC_DISPATCH_ALL_SCALAR); the two widths call the
 * kernel's fused sustain directly. The odd-length segment forces each width's
 * scalar tail.
 *
 * Run: cmake --build build --target osc_width_parity_test \
 *      && ctest --test-dir build -R osc_width_parity
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <limits.h>

/* Force the 8-wide intrinsics into scope on every host (production only pulls
 * avx2.h when __AVX2__ is defined, i.e. a real AVX2 build). On NEON / scalar
 * targets SIMDe emulates __m256 with the identical per-lane float semantics, so
 * the 8-wide logic is exercised faithfully here even though this host is 4-wide. */
#include "simde/x86/avx2.h"

#include "oscillator.h"
#include "oscillator_dispatch.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_osc_formulas.h"
#include "spectral_fast_math.h"
#include "spectral_common.h"
#include "oscillator_simd_scalar_waves.h"

/* Instantiate the kernel at both widths. The .inc suffixes every symbol with
 * OSC_VSUF, so the two instantiations coexist with no clash. */
#define OSC_VW   4
#define OSC_VSUF _w4
#include "oscillator_simd_kernel.inc"
#undef OSC_VW
#undef OSC_VSUF

#define OSC_VW   8
#define OSC_VSUF _w8
#include "oscillator_simd_kernel.inc"
#undef OSC_VW
#undef OSC_VSUF

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

/* Same per-sample budget as osc_parity (FMA-contraction class). */
#define BUDGET_ABS 1.0e-5
#define MAX_LEN    2048

typedef struct {
    size_t length;
    float  alpha;
    float  beta;
    float  phase;
    float  amp;
    float  d_amp;
    float  width;
} WidthSegment;

/* A low and a high partial, a chirped one, and an odd length that forces each
 * width's scalar tail (length not a multiple of 4 or 8). */
static const WidthSegment k_segments[] = {
    { 1000,  0.0576f,   0.0f,    0.30f,   0.80f, -2.0e-4f, 8.0f  },
    {  512,  0.5236f,   0.0f,   -1.10f,   0.50f,  0.0f,    5.0f  },
    {  768,  0.2000f,   1.0e-6f, 0.00f,   0.90f,  1.0e-4f, 6.0f  },
    {  259,  0.1234f,   0.0f,    2.50f,   0.70f,  3.0e-4f, 4.0f  },
};
#define N_SEGMENTS (sizeof(k_segments) / sizeof(k_segments[0]))

typedef struct { SpectralTimbre timbre; const char* name; float width; } WidthTimbre;

static const WidthTimbre k_timbres[] = {
    { TIMBRE_SINE,      "sine",      0.0f },
    { TIMBRE_SAW,       "saw",       0.0f },
    { TIMBRE_SQUARE,    "square",    0.0f },
    { TIMBRE_TRIANGLE,  "triangle",  0.0f },
    { TIMBRE_PARABOLA,  "parabola",  0.0f },
    { TIMBRE_QUANTIZED, "quantized", 8.0f },
    { TIMBRE_PWM,       "pwm",       0.5f },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

static void fill_lp(SegmentLoopParams* lp, const WidthSegment* s, float width) {
    memset(lp, 0, sizeof(*lp));
    lp->start_idx = 0;
    lp->length    = s->length;
    lp->alpha     = s->alpha;
    lp->beta      = s->beta;
    lp->c2        = s->beta;   /* no-linkage default: c2 == beta, c3 == 0 */
    lp->c3        = 0.0f;
    lp->phase     = s->phase;
    lp->amp       = s->amp;
    lp->d_amp     = s->d_amp;
    lp->width     = width;
    lp->valid     = 1;
}

/* Route a timbre through the 4-wide kernel instantiation. */
static void render_w4(float* dst, const SegmentLoopParams* lp, SpectralTimbre t, const float* width) {
    switch (t) {
        case TIMBRE_SINE:      osc_simd_fused_sustain_w4(dst, lp, wave_sine_v_w4,      wave_sine_1,      NULL);  break;
        case TIMBRE_SAW:       osc_simd_fused_sustain_w4(dst, lp, wave_saw_v_w4,       wave_saw_1,       NULL);  break;
        case TIMBRE_SQUARE:    osc_simd_fused_sustain_w4(dst, lp, wave_square_v_w4,    wave_square_1,    NULL);  break;
        case TIMBRE_TRIANGLE:  osc_simd_fused_sustain_w4(dst, lp, wave_triangle_v_w4,  wave_triangle_1,  NULL);  break;
        case TIMBRE_PARABOLA:  osc_simd_fused_sustain_w4(dst, lp, wave_parabola_v_w4,  wave_parabola_1,  NULL);  break;
        case TIMBRE_QUANTIZED: osc_simd_fused_sustain_w4(dst, lp, wave_quantized_v_w4, wave_quantized_1, width); break;
        case TIMBRE_PWM:       osc_simd_fused_sustain_w4(dst, lp, wave_pwm_v_w4,       wave_pwm_1,       width); break;
        default: break;
    }
}

/* Route a timbre through the 8-wide kernel instantiation. */
static void render_w8(float* dst, const SegmentLoopParams* lp, SpectralTimbre t, const float* width) {
    switch (t) {
        case TIMBRE_SINE:      osc_simd_fused_sustain_w8(dst, lp, wave_sine_v_w8,      wave_sine_1,      NULL);  break;
        case TIMBRE_SAW:       osc_simd_fused_sustain_w8(dst, lp, wave_saw_v_w8,       wave_saw_1,       NULL);  break;
        case TIMBRE_SQUARE:    osc_simd_fused_sustain_w8(dst, lp, wave_square_v_w8,    wave_square_1,    NULL);  break;
        case TIMBRE_TRIANGLE:  osc_simd_fused_sustain_w8(dst, lp, wave_triangle_v_w8,  wave_triangle_1,  NULL);  break;
        case TIMBRE_PARABOLA:  osc_simd_fused_sustain_w8(dst, lp, wave_parabola_v_w8,  wave_parabola_1,  NULL);  break;
        case TIMBRE_QUANTIZED: osc_simd_fused_sustain_w8(dst, lp, wave_quantized_v_w8, wave_quantized_1, width); break;
        case TIMBRE_PWM:       osc_simd_fused_sustain_w8(dst, lp, wave_pwm_v_w8,       wave_pwm_1,       width); break;
        default: break;
    }
}

static void test_width_parity(void) {
    double agg_max_8v4 = 0.0, agg_max_8vs = 0.0, agg_max_4vs = 0.0;

    for (size_t t = 0; t < N_TIMBRES; t++) {
        double t_8v4 = 0.0, t_8vs = 0.0, t_4vs = 0.0;
        for (size_t si = 0; si < N_SEGMENTS; si++) {
            SegmentLoopParams lp;
            fill_lp(&lp, &k_segments[si], k_timbres[t].width);
            float width = k_timbres[t].width;

            float buf_scalar[MAX_LEN], buf_w4[MAX_LEN], buf_w8[MAX_LEN];
            memset(buf_scalar, 0, sizeof(buf_scalar));
            memset(buf_w4, 0, sizeof(buf_w4));
            memset(buf_w8, 0, sizeof(buf_w8));

            osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR);
            timbre_synth_segment(buf_scalar, &lp, k_timbres[t].timbre);
            render_w4(buf_w4, &lp, k_timbres[t].timbre, &width);
            render_w8(buf_w8, &lp, k_timbres[t].timbre, &width);

            for (size_t j = 0; j < lp.length; j++) {
                double d84 = fabs((double)buf_w8[j] - (double)buf_w4[j]);
                double d8s = fabs((double)buf_w8[j] - (double)buf_scalar[j]);
                double d4s = fabs((double)buf_w4[j] - (double)buf_scalar[j]);
                if (d84 > t_8v4) t_8v4 = d84;
                if (d8s > t_8vs) t_8vs = d8s;
                if (d4s > t_4vs) t_4vs = d4s;
            }
        }
        if (t_8v4 > agg_max_8v4) agg_max_8v4 = t_8v4;
        if (t_8vs > agg_max_8vs) agg_max_8vs = t_8vs;
        if (t_4vs > agg_max_4vs) agg_max_4vs = t_4vs;
        printf("  %-9s  8-vs-4 = %.3e   8-vs-scalar = %.3e   4-vs-scalar = %.3e\n",
               k_timbres[t].name, t_8v4, t_8vs, t_4vs);
        CHECK(t_8v4 <= BUDGET_ABS, "%s 8-vs-4 drift %.3e > budget %.1e",
              k_timbres[t].name, t_8v4, BUDGET_ABS);
        CHECK(t_8vs <= BUDGET_ABS, "%s 8-vs-scalar drift %.3e > budget %.1e",
              k_timbres[t].name, t_8vs, BUDGET_ABS);
    }

    printf("\n  aggregate  8-vs-4 = %.3e   8-vs-scalar = %.3e   4-vs-scalar = %.3e  (budget %.1e)\n",
           agg_max_8v4, agg_max_8vs, agg_max_4vs, BUDGET_ABS);
    CHECK(agg_max_8v4 <= BUDGET_ABS, "aggregate 8-vs-4 drift %.3e > budget %.1e", agg_max_8v4, BUDGET_ABS);
    CHECK(agg_max_8vs <= BUDGET_ABS, "aggregate 8-vs-scalar drift %.3e > budget %.1e", agg_max_8vs, BUDGET_ABS);
}

int main(void) {
    printf("== SIMD width parity (8-wide == 4-wide == scalar within budget) ==\n");
    test_width_parity();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
