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

/* SegmentLoopParams defined in spectral_synth_internal.h */
struct SegmentLoopParams;

OSC_HOT float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width);

/* Segment synthesis with backend dispatch */
OSC_HOT void timbre_synth_segment(float* __restrict__ dst, const struct SegmentLoopParams* lp, SpectralTimbre timbre);

void osc_set_dispatch(OscDispatchWord dispatch);
OscDispatchWord osc_get_dispatch(void);

/* CUDA oscillator — delegates to canonical formulas in spectral_osc_formulas.h */
#ifdef __CUDACC__

#include "spectral_osc_formulas.h"

#define oscillator_normalize_phase_cuda spectral_normalize_phase
#define oscillator_fast_sin_cuda        spectral_fast_sin_inline

__device__ __forceinline__ float oscillator_cuda(float phase, int timbre) {
    float rads = spectral_normalize_phase(phase);
    switch (timbre) {
        case TIMBRE_SINE:     return spectral_osc_sine(rads, 0);
        case TIMBRE_SAW:      return spectral_osc_saw(rads, 0);
        case TIMBRE_SQUARE:   return spectral_osc_square(rads, 0);
        case TIMBRE_TRIANGLE: return spectral_osc_triangle(rads, 0);
        case TIMBRE_ASIN:     return spectral_osc_asin(rads, 0);
        case TIMBRE_PARABOLA: return spectral_osc_parabola(rads, 0);
        default:              return spectral_osc_sine(rads, 0);
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
