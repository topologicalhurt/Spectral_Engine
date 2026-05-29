/* oscillator_simd.c (embedded profile) - CMSIS-DSP oscillator synthesis.
 *
 * Build-selected for ARM Cortex-M builds with CMSIS-DSP (arm_math.h). The host
 * SIMDe counterpart lives in core/port/host/oscillator_simd.c. Both implement
 * the oscillator_dispatch.h SIMD segment interface. */
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

/* Capability gate: this translation unit is build-selected for CMSIS-DSP targets
 * and uses arm_math.h intrinsics. OSC_SIMD_CMSIS is defined by oscillator_dispatch.h
 * exactly when CMSIS-DSP is present (Cortex-M class), so the body is inert if the
 * file is ever seen without that capability. */
#if defined(OSC_SIMD_CMSIS)

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t phases[4], sines[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            phases[k] = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)(j + k)));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        for (int k = 0; k < 4; k++) sines[k] = arm_sin_f32(phases[k]);
        arm_mult_f32(sines, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float wave = arm_sin_f32(rads);
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t rads_v[4], waves[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)(j + k)));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_scale_f32(rads_v, -SPECTRAL_INV_PI, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float wave = rads * -SPECTRAL_INV_PI;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    for (size_t j = 0; j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float wave = rads > 0.0f ? 1.0f : -1.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t rads_v[4], abs_v[4], waves[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)(j + k))) * SPECTRAL_INV_PI;
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_abs_f32(rads_v, abs_v, 4);
        arm_scale_f32(abs_v, -2.0f, waves, 4);
        arm_offset_f32(waves, 1.0f, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float wave = 1.0f - fabsf(rads * SPECTRAL_INV_PI) * 2.0f;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;
    const float beta = lp->beta;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;

    size_t j = 0;
    float32_t rads_v[4], sq_v[4], waves[4], amps_v[4], results[4];

    for (; j + 4 <= len; j += 4) {
        for (int k = 0; k < 4; k++) {
            rads_v[k] = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)(j + k)));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        arm_mult_f32(rads_v, rads_v, sq_v, 4);
        arm_scale_f32(sq_v, -SPECTRAL_INV_PI_SQ, waves, 4);
        arm_offset_f32(waves, 1.0f, waves, 4);
        arm_mult_f32(waves, amps_v, results, 4);
        arm_add_f32(dst + j, results, dst + j, 4);
    }

    for (; j < len; j++) {
        float rads = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)j));
        float wave = 1.0f - rads * rads * SPECTRAL_INV_PI_SQ;
        float amp = (amp0 + d_amp * (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
}

/* No CMSIS SIMD for quantized/PWM - scalar fallback via dispatch */
void osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }

int osc_simd_available(SpectralTimbre timbre) {
    return timbre == TIMBRE_SINE || timbre == TIMBRE_SAW ||
           timbre == TIMBRE_TRIANGLE || timbre == TIMBRE_PARABOLA;
}

#endif /* OSC_SIMD_CMSIS */
