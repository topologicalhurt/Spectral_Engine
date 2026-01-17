/* oscillator.h - Oscillator module */
#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_lut.h"
#include "oscillator_dispatch.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define OSC_HOT __attribute__((hot))
#else
#define OSC_HOT
#endif

/* CPU oscillator - fast_sin() is canonical, GPU shaders must match */
OSC_HOT float fast_sin(float x);

/* SegmentLoopParams defined in spectral_synth_internal.h */
struct SegmentLoopParams;

OSC_HOT float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width);

/* Segment synthesis with backend dispatch */
OSC_HOT void timbre_synth_segment(float* __restrict__ dst, const struct SegmentLoopParams* lp, SpectralTimbre timbre);

void osc_set_dispatch(OscDispatchWord dispatch);
OscDispatchWord osc_get_dispatch(void);

/* CUDA oscillator - must be in header for __device__ visibility */
#ifdef __CUDACC__

__device__ __forceinline__ float oscillator_normalize_phase_cuda(float p) {
    float norm = p * SPECTRAL_INV_TWO_PI;
    return SPECTRAL_TWO_PI * (norm - floorf(norm) - 0.5f);
}

/* Padé [5/4] sine approximation - single divide, max error ~1e-5 */
__device__ __forceinline__ float oscillator_fast_sin_cuda(float x) {
    x = x - SPECTRAL_TWO_PI * floorf(x * SPECTRAL_INV_TWO_PI + 0.5f);
    float x2 = x * x;
    float num = x * (1.0f - x2 * (0.16605f - x2 * 0.00761f));
    float den = 1.0f + x2 * 0.00766f;
    return num / den;
}

__device__ __forceinline__ float oscillator_cuda(float phase, int timbre) {
    float rads = oscillator_normalize_phase_cuda(phase);
    switch (timbre) {
        case TIMBRE_SINE:     return oscillator_fast_sin_cuda(phase);
        case TIMBRE_SAW:      return rads * -SPECTRAL_INV_PI;
        case TIMBRE_SQUARE:   return (rads > 0.0f) ? 1.0f : -1.0f;
        case TIMBRE_TRIANGLE: return (1.0f - fabsf(rads) * SPECTRAL_TWO_INV_PI) * 2.0f - 1.0f;
        case TIMBRE_ASIN:     return asinf(rads * SPECTRAL_INV_PI);
        case TIMBRE_PARABOLA: return 1.0f - rads * rads * SPECTRAL_INV_PI_SQ;
        default:              return oscillator_fast_sin_cuda(phase);
    }
}

#endif

#if defined(__APPLE__) && !defined(__CUDACC__)
extern const char* oscillator_metal_source;
#endif

#ifdef __cplusplus
}
#endif

#endif
