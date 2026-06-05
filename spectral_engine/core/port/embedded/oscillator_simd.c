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

/* Phase model: quadratic only, by design.
 *
 * The host SIMDe oscillator also evaluates the cubic McAulay-Quatieri phase
 * polynomial under SPECTRAL_PRECISE_PHASE, but cubic phase is unreachable on
 * this profile: the cubic coefficients (c2, c3) are carried in the 64-byte
 * desktop/GPU Segment's spare pad words, and the 32-byte embedded Segment has no
 * room for them (see spectral_common.h). So spectral_segment_has_cubic() is
 * structurally false here, SegmentLoopParams always arrives with c2 == beta and
 * c3 == 0, and the quadratic helper spectral_segment_phase_at_f32() is exact and
 * complete -- a cubic Horner would only add per-sample cost on the Cortex-M for
 * coefficients that can never be non-trivial. If a future embedded build gains a
 * wider Segment, switch these calls to spectral_segment_phase_at_cubic_f32(lp->c2,
 * lp->c3) to mirror the host.
 *
 * Structure: phases and amplitudes are filled with a short scalar pre-pass (the
 * phase recurrence and the fade envelope are inherently per-sample), then the
 * heavy waveform/scale/accumulate work is handed to CMSIS-DSP vector kernels
 * (arm_*_f32) over 4-sample blocks. A scalar tail finishes the final <4 samples. */

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
        /* Scalar pre-pass: wrapped phase and enveloped amplitude per lane. */
        for (int k = 0; k < 4; k++) {
            phases[k] = phase_to_rads(spectral_segment_phase_at_f32(phase0, alpha, beta, (float)(j + k)));
            amps_v[k] = (amp0 + d_amp * (float)(j + k)) * fade_envelope(j + k, &fp, len);
        }
        /* Vector body: sin, then amp*sin, then accumulate onto the output. */
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

/* Sawtooth: scale each wrapped phase by -1/pi (arm_scale_f32) -> ramp in (-1, 1]. */
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

/* Square: no useful CMSIS vector primitive for the sign test, so this stays a
 * plain scalar loop (+1 where rads > 0, else -1), matching spectral_osc_square(). */
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

/* Triangle: |rads|/pi (arm_abs after pre-scaling), then 1 - 2*|..| via
 * scale(-2)+offset(1). Peak +1 at rads 0, troughs -1 at +/-pi. */
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

/* Parabola: rads^2 (arm_mult by itself), scaled by -1/pi^2, offset +1 -> an
 * inverted parabola, +1 at rads 0 down to ~-1 at +/-pi. */
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

/* Quantized and PWM have no efficient CMSIS vector form (their domain guards are
 * per-lane branches), so they are intentionally not advertised by
 * osc_simd_available() below and dispatch routes them to the scalar oscillator.
 * These no-op stubs exist only to satisfy the link-time symbol contract; they
 * are never called for these timbres. */
void osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }
void osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) { (void)dst; (void)lp; }

/* Timbres this profile renders with CMSIS SIMD. Square (scalar sign test),
 * asin, quantized, and pwm are excluded and fall back to the scalar oscillator. */
int osc_simd_available(SpectralTimbre timbre) {
    return timbre == TIMBRE_SINE || timbre == TIMBRE_SAW ||
           timbre == TIMBRE_TRIANGLE || timbre == TIMBRE_PARABOLA;
}

#endif /* OSC_SIMD_CMSIS */
