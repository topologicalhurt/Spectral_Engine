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
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void build_phase_array(float* phases, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;

#if defined(OSC_SIMD_GENERIC)
    /* Vectorized: compute 4 raw phases via quadratic polynomial, then vectorized phase_to_rads */
    const simde__m128 v_inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
    const simde__m128 v_2pi = simde_mm_set1_ps(SPECTRAL_TWO_PI);
    const simde__m128 v_half = simde_mm_set1_ps(0.5f);
    const simde__m128 v_alpha = simde_mm_set1_ps(alpha);
    const simde__m128 v_beta = simde_mm_set1_ps(beta);
    const simde__m128 v_phase0 = simde_mm_set1_ps(phase0);
    const simde__m128 v_offsets = simde_mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
    const simde__m128 v_four = simde_mm_set1_ps(4.0f);

    simde__m128 v_j = v_offsets;
    size_t j = 0;
    for (; j + 4 <= len; j += 4) {
        /* p = phase0 + j*(alpha + beta*j) */
        simde__m128 bj = simde_mm_mul_ps(v_beta, v_j);
        simde__m128 ab = simde_mm_add_ps(v_alpha, bj);
        simde__m128 raw = simde_mm_add_ps(v_phase0, simde_mm_mul_ps(v_j, ab));

        /* phase_to_rads: norm = raw * inv_2pi; rads = 2pi * (norm - floor(norm + 0.5)) */
        simde__m128 norm = simde_mm_mul_ps(raw, v_inv_2pi);
        simde__m128 shifted = simde_mm_add_ps(norm, v_half);
        simde__m128 floored = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(shifted));
        /* Correct for negative: if floored > shifted, subtract 1 */
        simde__m128 corr = simde_mm_and_ps(
            simde_mm_cmpgt_ps(floored, shifted),
            simde_mm_set1_ps(1.0f));
        floored = simde_mm_sub_ps(floored, corr);
        simde__m128 rads = simde_mm_mul_ps(v_2pi, simde_mm_sub_ps(norm, floored));

        simde_mm_storeu_ps(&phases[j], rads);
        v_j = simde_mm_add_ps(v_j, v_four);
    }
    for (; j < len; j++) {
        phases[j] = phase_to_rads(compute_phase(phase0, alpha, beta, j));
    }
#else
    for (size_t j = 0; j < len; j++) {
        phases[j] = phase_to_rads(compute_phase(phase0, alpha, beta, j));
    }
#endif
}

static void build_amp_envelope(float* amps, const SegmentLoopParams* lp, const FadeParams* fp) {
    const size_t len = lp->length;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    for (size_t j = 0; j < len; j++) {
        amps[j] = (amp0 + d_amp * (float)j) * fade_envelope(j, fp, len);
    }
}

static void apply_envelope_accumulate(float* dst, const float* wave, const float* amps, size_t len) {
    for (size_t j = 0; j < len; j++) {
        dst[j] += wave[j] * amps[j];
    }
}

/* Stack-allocate temp buffers when segment length fits; fall back to malloc for large segments. */

typedef struct { float* a; float* b; float* c; size_t len; int on_heap; } TempBuffers;

static TempBuffers alloc_temp_stack(float* stack_a, float* stack_b, float* stack_c,
                                     size_t len, int count) {
    TempBuffers t = {0};
    t.len = len;
    t.on_heap = 0;
    if (len <= SPECTRAL_OSC_SIMD_STACK_MAX) {
        t.a = stack_a;
        if (count >= 2) t.b = stack_b;
        if (count >= 3) t.c = stack_c;
    } else {
        t.on_heap = 1;
        t.a = (float*)malloc(len * sizeof(float));
        if (count >= 2) t.b = (float*)malloc(len * sizeof(float));
        if (count >= 3) t.c = (float*)malloc(len * sizeof(float));
    }
    return t;
}

static int temp_valid(const TempBuffers* t, int count) {
    if (!t->a) return 0;
    if (count >= 2 && !t->b) return 0;
    if (count >= 3 && !t->c) return 0;
    return 1;
}

static void free_temp(TempBuffers* t) {
    if (t->on_heap) { free(t->a); free(t->b); free(t->c); }
}

#if defined(OSC_SIMD_CMSIS)

/* CMSIS path: ARM Cortex-M embedded (DSP intrinsics, integer fixed-point) */

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
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

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
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

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
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

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
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

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
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

/* SIMDe SSE path: macOS ARM → NEON, Linux x86 → native SSE */

/* Vectorized Padé [5/4] sine approximation — matches fast_sin() but 4-wide */
static inline simde__m128 simde_fast_sin_ps(simde__m128 x) {
    /* Range reduce to [-pi, pi] */
    simde__m128 inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
    simde__m128 two_pi  = simde_mm_set1_ps(SPECTRAL_TWO_PI);
    simde__m128 half    = simde_mm_set1_ps(0.5f);

    /* x = x - 2pi * floor(x * inv_2pi + 0.5) */
    simde__m128 n = simde_mm_add_ps(simde_mm_mul_ps(x, inv_2pi), half);
    /* Floor via truncation toward zero then adjust */
    n = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(n));
    /* Handle negative: floor(x) for negative needs adjustment */
    simde__m128 correction = simde_mm_and_ps(
        simde_mm_cmpgt_ps(simde_mm_mul_ps(n, two_pi), simde_mm_add_ps(x, simde_mm_mul_ps(half, two_pi))),
        simde_mm_set1_ps(1.0f));
    n = simde_mm_sub_ps(n, correction);
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

/* Fused sustain synthesis for simple waveforms (saw, square, triangle, parabola).
 * Eliminates 3 temp buffers and 4 data passes for the sustain region (90%+ of samples).
 * Keep multi-pass for sine (Pade benefits from tight vectorized loop) and fade regions. */

typedef simde__m128 (*WaveformFn4)(simde__m128 phase);

static inline simde__m128 wave_saw_4(simde__m128 rads) {
    return simde_mm_mul_ps(rads, simde_mm_set1_ps(-SPECTRAL_INV_PI));
}

static inline simde__m128 wave_square_4(simde__m128 rads) {
    simde__m128 zero = simde_mm_setzero_ps();
    simde__m128 mask = simde_mm_cmpgt_ps(rads, zero);
    return simde_mm_or_ps(
        simde_mm_and_ps(mask, simde_mm_set1_ps(1.0f)),
        simde_mm_andnot_ps(mask, simde_mm_set1_ps(-1.0f)));
}

static inline simde__m128 wave_triangle_4(simde__m128 rads) {
    simde__m128 scaled = simde_mm_mul_ps(rads, simde_mm_set1_ps(SPECTRAL_INV_PI));
    simde__m128 abs_scaled = simde_mm_andnot_ps(simde_mm_set1_ps(-0.0f), scaled);
    return simde_mm_add_ps(simde_mm_set1_ps(1.0f),
                           simde_mm_mul_ps(simde_mm_set1_ps(-2.0f), abs_scaled));
}

static inline simde__m128 wave_parabola_4(simde__m128 rads) {
    simde__m128 sq = simde_mm_mul_ps(rads, rads);
    return simde_mm_add_ps(simde_mm_set1_ps(1.0f),
                           simde_mm_mul_ps(sq, simde_mm_set1_ps(-SPECTRAL_INV_PI_SQ)));
}

static void osc_simd_fused_sustain(float* dst, const SegmentLoopParams* lp, WaveformFn4 wave_fn) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    const size_t fade_in_end = fp.fade_len;
    const size_t fade_out_start = fp.fade_out_start;

    /* Fade-in region: scalar (small, not worth vectorizing) */
    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        simde__m128 v_rads = simde_mm_set1_ps(rads);
        simde__m128 v_wave = wave_fn(v_rads);
        float tmp[4];
        simde_mm_storeu_ps(tmp, v_wave);
        float amp = (amp0 + d_amp * (float)j) * fade_envelope_in(j, fp.inv_fade);
        dst[j] += amp * tmp[0];
    }

    /* Sustain region: fused vectorized path (no temp buffers) */
    {
        const simde__m128 v_inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
        const simde__m128 v_2pi = simde_mm_set1_ps(SPECTRAL_TWO_PI);
        const simde__m128 v_half = simde_mm_set1_ps(0.5f);
        const simde__m128 v_one = simde_mm_set1_ps(1.0f);
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

            /* phase_to_rads */
            simde__m128 norm = simde_mm_mul_ps(raw, v_inv_2pi);
            simde__m128 shifted = simde_mm_add_ps(norm, v_half);
            simde__m128 floored = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(shifted));
            simde__m128 corr = simde_mm_and_ps(
                simde_mm_cmpgt_ps(floored, shifted), v_one);
            floored = simde_mm_sub_ps(floored, corr);
            simde__m128 rads = simde_mm_mul_ps(v_2pi, simde_mm_sub_ps(norm, floored));

            /* Waveform + amplitude + accumulate */
            simde__m128 wave = wave_fn(rads);
            simde__m128 amp = simde_mm_add_ps(v_amp0, simde_mm_mul_ps(v_d_amp, v_j));
            simde__m128 existing = simde_mm_loadu_ps(&dst[j]);
            simde_mm_storeu_ps(&dst[j],
                simde_mm_add_ps(existing, simde_mm_mul_ps(amp, wave)));

            v_j = simde_mm_add_ps(v_j, v_four);
        }
        /* Scalar tail for sustain */
        for (; j < sustain_end; j++) {
            float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
            simde__m128 v_rads = simde_mm_set1_ps(rads);
            simde__m128 v_wave = wave_fn(v_rads);
            float tmp[4];
            simde_mm_storeu_ps(tmp, v_wave);
            dst[j] += (amp0 + d_amp * (float)j) * tmp[0];
        }
    }

    /* Fade-out region: scalar */
    size_t fo_start = (fade_out_start > fade_in_end) ? fade_out_start : fade_in_end;
    for (size_t j = fo_start; j < len; j++) {
        float rads = phase_to_rads(compute_phase(phase0, alpha, beta, j));
        simde__m128 v_rads = simde_mm_set1_ps(rads);
        simde__m128 v_wave = wave_fn(v_rads);
        float tmp[4];
        simde_mm_storeu_ps(tmp, v_wave);
        float amp = (amp0 + d_amp * (float)j) * fade_envelope_out(j, len, fp.inv_fade);
        dst[j] += amp * tmp[0];
    }
}

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    float sa[SPECTRAL_OSC_SIMD_STACK_MAX], sb[SPECTRAL_OSC_SIMD_STACK_MAX], sc[SPECTRAL_OSC_SIMD_STACK_MAX];
    TempBuffers t = alloc_temp_stack(sa, sb, sc, len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }

    build_phase_array(t.a, lp);

    /* Vectorized sine via SIMDe SSE */
    size_t j = 0;
    for (; j + 4 <= len; j += 4) {
        simde__m128 phase_vec = simde_mm_loadu_ps(&t.a[j]);
        simde__m128 sin_vec = simde_fast_sin_ps(phase_vec);
        simde_mm_storeu_ps(&t.b[j], sin_vec);
    }
    for (; j < len; j++) {
        t.b[j] = fast_sin(t.a[j]);
    }

    build_amp_envelope(t.c, lp, &fp);
    apply_envelope_accumulate(dst, t.b, t.c, len);

    free_temp(&t);
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_saw_4);
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_square_4);
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_triangle_4);
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    osc_simd_fused_sustain(dst, lp, wave_parabola_4);
}

/* Quantized: wave = trunc(rads * width) / width */
void osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    float sa[SPECTRAL_OSC_SIMD_STACK_MAX], sb[SPECTRAL_OSC_SIMD_STACK_MAX], sc[SPECTRAL_OSC_SIMD_STACK_MAX];
    TempBuffers t = alloc_temp_stack(sa, sb, sc, len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }

    build_phase_array(t.a, lp);

    float width = lp->width;
    if (width <= 0.0f) {
        memset(t.b, 0, len * sizeof(float));
    } else {
        simde__m128 v_width = simde_mm_set1_ps(width);
        simde__m128 v_inv_w = simde_mm_set1_ps(1.0f / width);
        size_t j = 0;
        for (; j + 4 <= len; j += 4) {
            simde__m128 rads = simde_mm_loadu_ps(&t.a[j]);
            /* trunc(rads * width) / width */
            simde__m128 scaled = simde_mm_mul_ps(rads, v_width);
            simde__m128 truncated = simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(scaled));
            simde__m128 wave = simde_mm_mul_ps(truncated, v_inv_w);
            simde_mm_storeu_ps(&t.b[j], wave);
        }
        for (; j < len; j++) {
            float inv_w = 1.0f / width;
            t.b[j] = (float)(int)(t.a[j] * width) * inv_w;
        }
    }

    build_amp_envelope(t.c, lp, &fp);
    apply_envelope_accumulate(dst, t.b, t.c, len);
    free_temp(&t);
}

/* PWM: wave = ((rads + PI) * INV_TWO_PI < width) ? 1 : -1 */
void osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, FADE_SAMPLES_DEFAULT);
    float sa[SPECTRAL_OSC_SIMD_STACK_MAX], sb[SPECTRAL_OSC_SIMD_STACK_MAX], sc[SPECTRAL_OSC_SIMD_STACK_MAX];
    TempBuffers t = alloc_temp_stack(sa, sb, sc, len, 3);
    if (!temp_valid(&t, 3)) { free_temp(&t); return; }

    build_phase_array(t.a, lp);

    float width = lp->width;
    if (width <= 0.0f) {
        /* PWM with width=0 is DC +1 */
        for (size_t j = 0; j < len; j++) t.b[j] = 1.0f;
    } else {
        simde__m128 v_pi = simde_mm_set1_ps(SPECTRAL_PI);
        simde__m128 v_inv_2pi = simde_mm_set1_ps(SPECTRAL_INV_TWO_PI);
        simde__m128 v_width = simde_mm_set1_ps(width);
        simde__m128 v_one = simde_mm_set1_ps(1.0f);
        simde__m128 v_neg_one = simde_mm_set1_ps(-1.0f);
        size_t j = 0;
        for (; j + 4 <= len; j += 4) {
            simde__m128 rads = simde_mm_loadu_ps(&t.a[j]);
            simde__m128 norm = simde_mm_mul_ps(simde_mm_add_ps(rads, v_pi), v_inv_2pi);
            simde__m128 cmp = simde_mm_cmplt_ps(norm, v_width);
            simde__m128 wave = simde_mm_or_ps(
                simde_mm_and_ps(cmp, v_one),
                simde_mm_andnot_ps(cmp, v_neg_one));
            simde_mm_storeu_ps(&t.b[j], wave);
        }
        for (; j < len; j++) {
            t.b[j] = ((t.a[j] + SPECTRAL_PI) * SPECTRAL_INV_TWO_PI < width) ? 1.0f : -1.0f;
        }
    }

    build_amp_envelope(t.c, lp, &fp);
    apply_envelope_accumulate(dst, t.b, t.c, len);
    free_temp(&t);
}

int osc_simd_available(SpectralTimbre timbre) {
    return (timbre <= TIMBRE_PWM);
}

#endif /* OSC_SIMD_GENERIC */

static int g_native_backend_available = 0;
void osc_set_native_available(int available) { g_native_backend_available = available; }
int osc_native_available(void) { return g_native_backend_available; }
