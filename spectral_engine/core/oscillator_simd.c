/* oscillator_simd.c - SIMD-accelerated oscillator synthesis
 *
 * Two paths: CMSIS for ARM Cortex-M embedded, SIMDe SSE for all desktop
 * platforms (macOS ARM -> NEON, Linux x86 -> native SSE, else -> scalar).
 */
#include "oscillator_dispatch.h"
#include "oscillator.h"
#include "spectral_config.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(OSC_SIMD_CMSIS)

/* CMSIS path: ARM Cortex-M embedded (DSP intrinsics, integer fixed-point) */

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t phases[4], sines[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            phases[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        for (int k = 0; k < 4; k++) sines[k] = arm_sin_f32(phases[k]);
        arm_mult_f32(sines, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = arm_sin_f32(rads);
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t rads_v[4], waves[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_scale_f32(rads_v, -SPECTRAL_INV_PI, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = rads * -SPECTRAL_INV_PI;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    for (size_t j = 0; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = rads > 0.0f ? 1.0f : -1.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t rads_v[4], abs_v[4], waves[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k)) * SPECTRAL_INV_PI;
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_abs_f32(rads_v, abs_v, 4);
        arm_scale_f32(abs_v, -2.0f, waves, 4);
        arm_offset_f32(waves, 1.0f, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = 1.0f - fabsf(rads * SPECTRAL_INV_PI) * 2.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t rads_v[4], sq_v[4], waves[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(compute_phase(phase0, alpha, beta, j + k));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_mult_f32(rads_v, rads_v, sq_v, 4);
        arm_scale_f32(sq_v, -SPECTRAL_INV_PI_SQ, waves, 4);
        arm_offset_f32(waves, 1.0f, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        float wave = 1.0f - rads * rads * SPECTRAL_INV_PI_SQ;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

/* No CMSIS SIMD for quantized/PWM - scalar fallback via dispatch */
void osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }

int osc_simd_available(SpectralTimbre timbre) {
    return (timbre <= TIMBRE_PARABOLA && timbre != TIMBRE_SQUARE);
}

#elif defined(OSC_SIMD_GENERIC)

/* SIMDe SSE path: macOS ARM -> NEON, Linux x86 -> native SSE */

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
    simde__m128 norm = simde_mm_mul_ps(phase, v_inv_2pi);
    simde__m128 floored = simde_floor_ps_portable(norm);
    return simde_mm_mul_ps(v_2pi, simde_mm_sub_ps(simde_mm_sub_ps(norm, floored), v_half));
}

/* Vectorized Padé [5/4] sine approximation — matches fast_sin() but 4-wide */
static inline simde__m128 simde_fast_sin_ps(simde__m128 x) {
    const simde__m128 inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
    const simde__m128 two_pi = simde_mm_set1_ps(SPECTRAL_TWO_PI);
    const simde__m128 half = simde_mm_set1_ps(0.5f);

    /* Canonical range reduction:
     * x = x - 2pi * floor(x * inv_2pi + 0.5) */
    simde__m128 n = simde_floor_ps_portable(
        simde_mm_add_ps(simde_mm_mul_ps(x, inv_2pi), half));
    x = simde_mm_sub_ps(x, simde_mm_mul_ps(n, two_pi));

    /* Padé [5/4]: num = x * (1 - x2*(0.16605 - x2*0.00761))
     *             den = 1 + x2*0.00766 */
    simde__m128 x2 = simde_mm_mul_ps(x, x);
    simde__m128 c1 = simde_mm_set1_ps(SPECTRAL_PADE_SIN_C2);
    simde__m128 c2 = simde_mm_set1_ps(SPECTRAL_PADE_SIN_C1);
    simde__m128 c3 = simde_mm_set1_ps(SPECTRAL_PADE_SIN_C3);
    simde__m128 one = simde_mm_set1_ps(1.0f);

    simde__m128 num_inner = simde_mm_sub_ps(c2, simde_mm_mul_ps(x2, c1));
    simde__m128 num = simde_mm_mul_ps(x, simde_mm_sub_ps(one, simde_mm_mul_ps(x2, num_inner)));
    simde__m128 den = simde_mm_add_ps(one, simde_mm_mul_ps(x2, c3));

    return simde_mm_div_ps(num, den);
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
    simde__m128 truncated = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(scaled));
    return simde_mm_mul_ps(truncated, v_inv_w);
}

static inline simde__m128 wave_pwm_4(simde__m128 rads, const void* ctx) {
    float width = *(const float*)ctx;
    if (width <= 0.0f) return simde_mm_set1_ps(1.0f);
    simde__m128 v_pi = simde_mm_set1_ps(SPECTRAL_PI);
    simde__m128 v_inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
    simde__m128 v_width = simde_mm_set1_ps(width);
    simde__m128 v_one = simde_mm_set1_ps(1.0f);
    simde__m128 v_neg_one = simde_mm_set1_ps(-1.0f);
    simde__m128 norm = simde_mm_mul_ps(simde_mm_add_ps(rads, v_pi), v_inv_2pi);
    simde__m128 cmp = simde_mm_cmplt_ps(norm, v_width);
    return simde_mm_or_ps(
        simde_mm_and_ps(cmp, v_one),
        simde_mm_andnot_ps(cmp, v_neg_one));
}

/* Fused single-pass synthesis: inline phase computation, waveform, envelope,
 * and accumulation with zero temp buffers. Handles fade-in, sustain, and
 * fade-out regions. */

static void osc_simd_fused_sustain(float* dst, const SegmentLoopParams* lp,
                                    WaveformFn4 wave_fn4, WaveformFn1 wave_fn1,
                                    const void* ctx) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_DESKTOP);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    const size_t fade_in_end = fp.fade_len;
    const size_t fade_out_start = fp.fade_out_start;

    /* Fade-in region: scalar */
    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
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
            float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
            dst[j] += (amp0 + d_amp * (float)j) * wave_fn1(rads, ctx);
        }
    }

    /* Fade-out region: scalar */
    size_t fo_start = (fade_out_start > fade_in_end) ? fade_out_start : fade_in_end;
    for (size_t j = fo_start; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
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
    return (timbre <= TIMBRE_PWM);
}

#endif /* OSC_SIMD_GENERIC */

static int g_native_backend_available = 0;
void osc_set_native_available(int available) { g_native_backend_available = available; }
int osc_native_available(void) { return g_native_backend_available; }
