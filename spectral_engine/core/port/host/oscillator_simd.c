/* oscillator_simd.c (host profile) - SIMDe SSE oscillator synthesis.
 *
 * Build-selected for host builds: SIMDe maps SSE2 intrinsics to native NEON
 * (macOS ARM), native SSE (Linux x86), or scalar fallback. The CMSIS (ARM
 * Cortex-M) counterpart lives in core/port/embedded/oscillator_simd.c. Both
 * implement the oscillator_dispatch.h SIMD segment interface. */
#include "oscillator_dispatch.h"
#include "oscillator.h"
#include "spectral_config.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <string.h>

static inline simde__m128 simde_floor_ps_portable(simde__m128 x) {
#if defined(SIMDE_X86_SSE4_1_NATIVE) || defined(SIMDE_X86_SSE4_1_NO_NATIVE)
    return simde_mm_floor_ps(x);
#else
    simde_float32 lanes[4];
    simde_mm_storeu_ps(lanes, x);
    lanes[0] = floorf(lanes[0]);
    lanes[1] = floorf(lanes[1]);
    lanes[2] = floorf(lanes[2]);
    lanes[3] = floorf(lanes[3]);
    return simde_mm_loadu_ps(lanes);
#endif
}

/* Canonical SIMD phase normalization:
 * mirrors spectral_normalize_phase() in spectral_osc_formulas.h exactly. */

static inline simde__m128 simde_normalize_phase_ps(simde__m128 phase) {
    const simde__m128 v_inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
    const simde__m128 v_2pi = simde_mm_set1_ps(SPECTRAL_TWO_PI);
    const simde__m128 v_half = simde_mm_set1_ps(0.5f);
    simde__m128 cycles = simde_mm_add_ps(simde_mm_mul_ps(phase, v_inv_2pi), v_half);
    simde__m128 n = simde_floor_ps_portable(cycles);
    return simde_mm_sub_ps(phase, simde_mm_mul_ps(v_2pi, n));
}

/* Vectorized Padé [5/4] sine approximation — matches fast_sin() but 4-wide */
static inline simde__m128 simde_fast_sin_ps(simde__m128 x) {
#if SPECTRAL_ENABLE_APPROX_TRIG
    const simde__m128 inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
    const simde__m128 two_pi = simde_mm_set1_ps(SPECTRAL_TWO_PI);
    const simde__m128 half = simde_mm_set1_ps(0.5f);
    simde__m128 n = simde_floor_ps_portable(
        simde_mm_add_ps(simde_mm_mul_ps(x, inv_2pi), half));
    x = simde_mm_sub_ps(x, simde_mm_mul_ps(n, two_pi));

    simde__m128 x2 = simde_mm_mul_ps(x, x);
    simde__m128 c15 = simde_mm_set1_ps(-0.0000000000007647163731819816f);
    simde__m128 c13 = simde_mm_set1_ps( 0.00000000016059043836821613f);
    simde__m128 c11 = simde_mm_set1_ps(-0.00000002505210838544172f);
    simde__m128 c9  = simde_mm_set1_ps( 0.0000027557319223985893f);
    simde__m128 c7  = simde_mm_set1_ps(-0.0001984126984126984f);
    simde__m128 c5  = simde_mm_set1_ps( 0.008333333333333333f);
    simde__m128 c3  = simde_mm_set1_ps(-0.16666666666666666f);
    simde__m128 one = simde_mm_set1_ps(1.0f);

    simde__m128 p = c15;
    p = simde_mm_add_ps(c13, simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(c11, simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(c9,  simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(c7,  simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(c5,  simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(c3,  simde_mm_mul_ps(x2, p));
    p = simde_mm_add_ps(one, simde_mm_mul_ps(x2, p));
    return simde_mm_mul_ps(x, p);
#else
    simde_float32 lanes[4];
    simde_mm_storeu_ps(lanes, x);
    lanes[0] = sinf(lanes[0]);
    lanes[1] = sinf(lanes[1]);
    lanes[2] = sinf(lanes[2]);
    lanes[3] = sinf(lanes[3]);
    return simde_mm_loadu_ps(lanes);
#endif
}

/* Scalar waveform functions for fade regions (avoids wasteful SIMD broadcast) */

typedef float (*WaveformFn1)(float rads, const void* ctx);
typedef simde__m128 (*WaveformFn4)(simde__m128 phase, const void* ctx);

static inline float wave_saw_1(float rads, const void* ctx) {
    (void)ctx;
    return spectral_osc_saw(rads, 0.0f);
}
static inline float wave_square_1(float rads, const void* ctx) {
    (void)ctx;
    return spectral_osc_square(rads, 0.0f);
}
static inline float wave_triangle_1(float rads, const void* ctx) {
    (void)ctx;
    return spectral_osc_triangle(rads, 0.0f);
}
static inline float wave_parabola_1(float rads, const void* ctx) {
    (void)ctx;
    return spectral_osc_parabola(rads, 0.0f);
}
static inline float wave_sine_1(float rads, const void* ctx) {
    (void)ctx;
    return spectral_osc_sine(rads, 0.0f);
}
static inline float wave_quantized_1(float rads, const void* ctx) {
    float width = *(const float*)ctx;
    return spectral_osc_quantized(rads, width);
}
static inline float wave_pwm_1(float rads, const void* ctx) {
    float width = *(const float*)ctx;
    return spectral_osc_pwm(rads, width);
}

static inline simde__m128 wave_saw_4(simde__m128 rads, const void* ctx) {
    (void)ctx;
    return simde_mm_mul_ps(rads, simde_mm_set1_ps(-SPECTRAL_INV_PI));
}

static inline simde__m128 wave_square_4(simde__m128 rads, const void* ctx) {
    (void)ctx;
    simde__m128 zero = simde_mm_setzero_ps();
    simde__m128 mask = simde_mm_cmpgt_ps(rads, zero);
    return simde_mm_or_ps(
        simde_mm_and_ps(mask, simde_mm_set1_ps(1.0f)),
        simde_mm_andnot_ps(mask, simde_mm_set1_ps(-1.0f)));
}

static inline simde__m128 wave_triangle_4(simde__m128 rads, const void* ctx) {
    (void)ctx;
    simde__m128 scaled = simde_mm_mul_ps(rads, simde_mm_set1_ps(SPECTRAL_INV_PI));
    simde__m128 abs_scaled = simde_mm_andnot_ps(simde_mm_set1_ps(-0.0f), scaled);
    return simde_mm_add_ps(simde_mm_set1_ps(1.0f),
                           simde_mm_mul_ps(simde_mm_set1_ps(-2.0f), abs_scaled));
}

static inline simde__m128 wave_parabola_4(simde__m128 rads, const void* ctx) {
    (void)ctx;
    simde__m128 sq = simde_mm_mul_ps(rads, rads);
    return simde_mm_add_ps(simde_mm_set1_ps(1.0f),
                           simde_mm_mul_ps(sq, simde_mm_set1_ps(-SPECTRAL_INV_PI_SQ)));
}

static inline simde__m128 wave_sine_4(simde__m128 rads, const void* ctx) {
    (void)ctx;
    return simde_fast_sin_ps(rads);
}

static inline simde__m128 wave_quantized_4(simde__m128 rads, const void* ctx) {
    float width = *(const float*)ctx;
    if (width <= 0.0f) return simde_mm_setzero_ps();
    simde__m128 v_width = simde_mm_set1_ps(width);
    simde__m128 v_inv_w = simde_mm_set1_ps(1.0f / width);
    simde__m128 scaled = simde_mm_mul_ps(rads, v_width);
    /* Mirror the canonical spectral_osc_quantized() domain guard: it returns 0
     * when `scaled` is non-finite or falls outside [INT_MIN, INT_MAX]. Without
     * this, simde_mm_cvttps_epi32() saturates such lanes to INT_MIN (0x80000000)
     * and emits INT_MIN*inv_w (an out-of-[-1,1] value), diverging from both the
     * scalar contract and this segment's own fade-region scalar lane
     * (wave_quantized_1) for finite-but-large widths from deserialized segments.
     * The >=/<= comparisons also reject NaN/Inf (all NaN compares are false). */
    simde__m128 in_range = simde_mm_and_ps(
        simde_mm_cmpge_ps(scaled, simde_mm_set1_ps((float)INT_MIN)),
        simde_mm_cmple_ps(scaled, simde_mm_set1_ps((float)INT_MAX)));
    simde__m128 truncated = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(scaled));
    return simde_mm_and_ps(in_range, simde_mm_mul_ps(truncated, v_inv_w));
}

static inline simde__m128 wave_pwm_4(simde__m128 rads, const void* ctx) {
    float width = *(const float*)ctx;
    /* Mirror the canonical spectral_osc_pwm() domain guard, exactly as
     * wave_quantized_4 mirrors spectral_osc_quantized(). The scalar contract is
     *   !isfinite(rads) || !isfinite(width) -> 0;  width <= 0 -> 1;  else +/-1.
     * Without this the SIMD sustain lane emits +/-1 where both the scalar
     * contract and this segment's own fade-region lane (wave_pwm_1) emit 0 for a
     * non-finite phase (reachable when a deserialized segment's finite-but-huge
     * omega overflows the accumulated phase to +/-Inf -> NaN after
     * normalization), seam-splitting a single PWM segment. */
    simde__m128 wave;
    if (!isfinite(width)) return simde_mm_setzero_ps();
    if (width <= 0.0f) {
        wave = simde_mm_set1_ps(1.0f);
    } else {
        simde__m128 v_pi = simde_mm_set1_ps(SPECTRAL_PI);
        simde__m128 v_inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
        simde__m128 v_width = simde_mm_set1_ps(width);
        simde__m128 v_one = simde_mm_set1_ps(1.0f);
        simde__m128 v_neg_one = simde_mm_set1_ps(-1.0f);
        simde__m128 norm = simde_mm_mul_ps(simde_mm_add_ps(rads, v_pi), v_inv_2pi);
        simde__m128 cmp = simde_mm_cmplt_ps(norm, v_width);
        wave = simde_mm_or_ps(
            simde_mm_and_ps(cmp, v_one),
            simde_mm_andnot_ps(cmp, v_neg_one));
    }
    /* Per-lane finite-rads mask: |rads| <= FLT_MAX rejects NaN (unordered
     * compare is false) and +/-Inf, forcing those lanes to 0 like the scalar. */
    simde__m128 abs_rads = simde_mm_andnot_ps(simde_mm_set1_ps(-0.0f), rads);
    simde__m128 finite = simde_mm_cmple_ps(abs_rads, simde_mm_set1_ps(FLT_MAX));
    return simde_mm_and_ps(finite, wave);
}

/* Fused single-pass synthesis: inline phase computation, waveform, envelope,
 * and accumulation with zero temp buffers. Handles fade-in, sustain, and
 * fade-out regions. */

static void osc_simd_fused_sustain(float* dst, const SegmentLoopParams* lp,
                                    WaveformFn4 wave_fn4, WaveformFn1 wave_fn1,
                                    const void* ctx) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    const size_t fade_in_end = fp.fade_len;
    const size_t fade_out_start = fp.fade_out_start;

    /* Fade-in region: scalar */
    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float amp = (amp0 + d_amp * (float)j) * fade_envelope_in(j, fp.inv_fade);
        dst[j] += amp * wave_fn1(rads, ctx);
    }

    /* Sustain region: fused vectorized path (no temp buffers) */
    {
        const simde__m128 v_alpha = simde_mm_set1_ps(alpha);
        const simde__m128 v_beta = simde_mm_set1_ps(beta);
        const simde__m128 v_phase0 = simde_mm_set1_ps(phase0);
        const simde__m128 v_amp0 = simde_mm_set1_ps(amp0);
        const simde__m128 v_d_amp = simde_mm_set1_ps(d_amp);
        const simde__m128 v_four = simde_mm_set1_ps(4.0f);

        size_t sustain_end = (fade_out_start < len) ? fade_out_start : len;
        size_t j = fade_in_end;
        simde__m128 v_j = simde_mm_set_ps((float)(j+3), (float)(j+2), (float)(j+1), (float)j);

        for (; j + 4 <= sustain_end; j += 4) {
            /* Phase: p = phase0 + j*(alpha + beta*j) */
            simde__m128 bj = simde_mm_mul_ps(v_beta, v_j);
            simde__m128 ab = simde_mm_add_ps(v_alpha, bj);
            simde__m128 raw = simde_mm_add_ps(v_phase0, simde_mm_mul_ps(v_j, ab));

            simde__m128 rads = simde_normalize_phase_ps(raw);

            /* Waveform + amplitude + accumulate */
            simde__m128 wave = wave_fn4(rads, ctx);
            simde__m128 amp = simde_mm_add_ps(v_amp0, simde_mm_mul_ps(v_d_amp, v_j));
            simde__m128 existing = simde_mm_loadu_ps(&dst[j]);
            simde_mm_storeu_ps(&dst[j],
                simde_mm_add_ps(existing, simde_mm_mul_ps(amp, wave)));

            v_j = simde_mm_add_ps(v_j, v_four);
        }
        /* Scalar tail for sustain */
        for (; j < sustain_end; j++) {
            float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
            dst[j] += (amp0 + d_amp * (float)j) * wave_fn1(rads, ctx);
        }
    }

    /* Fade-out region: scalar */
    size_t fo_start = (fade_out_start > fade_in_end) ? fade_out_start : fade_in_end;
    for (size_t j = fo_start; j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float amp = (amp0 + d_amp * (float)j) * fade_envelope_out(j, len, fp.inv_fade);
        dst[j] += amp * wave_fn1(rads, ctx);
    }
}

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_sine_4, wave_sine_1, NULL);
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_saw_4, wave_saw_1, NULL);
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_square_4, wave_square_1, NULL);
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_triangle_4, wave_triangle_1, NULL);
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_parabola_4, wave_parabola_1, NULL);
}

void osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) {
    float width = lp->width;
    osc_simd_fused_sustain(dst, lp, wave_quantized_4, wave_quantized_1, &width);
}

void osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) {
    float width = lp->width;
    osc_simd_fused_sustain(dst, lp, wave_pwm_4, wave_pwm_1, &width);
}

int osc_simd_available(SpectralTimbre timbre) {
    return timbre == TIMBRE_SINE || timbre == TIMBRE_SAW ||
           timbre == TIMBRE_SQUARE || timbre == TIMBRE_TRIANGLE ||
           timbre == TIMBRE_PARABOLA || timbre == TIMBRE_QUANTIZED ||
           timbre == TIMBRE_PWM;
}
