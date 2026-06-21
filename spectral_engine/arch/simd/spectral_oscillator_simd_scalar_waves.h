/* spectral_oscillator_simd_scalar_waves.h - width-independent scalar waveform lanes.
 *
 * The fade-in / fade-out / SIMD-tail regions of the fused oscillator render one
 * sample at a time (per-sample envelope branching makes a SIMD broadcast
 * wasteful there). These wave_*_1 forward to the canonical spectral_osc_*() so
 * the scalar regions are numerically identical to the scalar oscillator and to
 * the vector sustain lanes they border. They are scalar, so they do NOT depend
 * on the SIMD lane width and live here, shared by oscillator_simd.c and the
 * width-parity test, rather than inside the width-templated kernel. */
#ifndef SPECTRAL_OSCILLATOR_SIMD_SCALAR_WAVES_H
#define SPECTRAL_OSCILLATOR_SIMD_SCALAR_WAVES_H

#include "spectral_osc_formulas.h"

/* Scalar single-sample waveform fn: takes a phase already wrapped to [-pi, pi)
 * plus an opaque per-segment context (quantized levels / pwm duty), NULL else. */
typedef float (*WaveformFn1)(float rads, const void* ctx);

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

#endif /* SPECTRAL_OSCILLATOR_SIMD_SCALAR_WAVES_H */
