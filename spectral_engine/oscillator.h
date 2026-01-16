/* oscillator.h - Oscillator Functions and LUT-Based Waveform Generation
 * 
 * This module provides:
 *   - Q15 sine LUT initialization and lookup with linear interpolation
 *   - Segment loop parameter extraction for synthesis
 *   - Multi-timbre oscillator for float synthesis (desktop)
 * 
 * LUT Configuration:
 *   SPECTRAL_OSC_LUT_BITS controls table size (default 12 = 4096 entries)
 *   Lookup uses 16-bit phase accumulator with interpolation for smooth output
 * 
 * The Q15 LUT functions are used by embedded synthesis, while
 * timbre_oscillator() provides the full timbre set for desktop.
 */
#ifndef OSCILLATOR_H
#define OSCILLATOR_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_q15.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SPECTRAL_OSC_LUT_BITS
#define SPECTRAL_OSC_LUT_BITS   12
#endif

#define SPECTRAL_OSC_LUT_SIZE   (1u << SPECTRAL_OSC_LUT_BITS)
#define SPECTRAL_OSC_LUT_MASK   (SPECTRAL_OSC_LUT_SIZE - 1u)
#define SPECTRAL_OSC_FRAC_BITS  (16u - SPECTRAL_OSC_LUT_BITS)
#define SPECTRAL_OSC_FRAC_MASK  ((1u << SPECTRAL_OSC_FRAC_BITS) - 1u)

#define SPECTRAL_SIN_LUT_BITS   SPECTRAL_OSC_LUT_BITS
#define SPECTRAL_SIN_LUT_SIZE   SPECTRAL_OSC_LUT_SIZE
#define SPECTRAL_SIN_LUT_MASK   SPECTRAL_OSC_LUT_MASK

#if defined(__GNUC__)
#define OSC_HOT __attribute__((hot))
#else
#define OSC_HOT
#endif

void spectral_osc_lut_init_sine(q15_t* lut);
#define spectral_init_sin_lut spectral_osc_lut_init_sine

OSC_HOT q15_t spectral_osc_lut_lookup(uq16_t phase_u16, const q15_t* lut);
#define spectral_sin_q15 spectral_osc_lut_lookup

OSC_HOT q15_t spectral_osc_lut_lookup_cos(uq16_t phase_u16, const q15_t* lut);
#define spectral_cos_q15 spectral_osc_lut_lookup_cos

typedef struct {
    size_t start_idx;
    size_t length;
    float alpha;
    float beta;
    float d_amp;
    float phase;
    float amp;
    float width;
    int valid;
} SegmentLoopParams;

SegmentLoopParams segment_loop_params_init(const Segment* s, const SynthParams* p, size_t out_len);

OSC_HOT float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width);

/* SIMD-optimized segment synthesis - renders entire segment with inlined waveform math */
OSC_HOT void timbre_synth_segment(float* __restrict__ dst, const SegmentLoopParams* lp, SpectralTimbre timbre);

#ifdef __cplusplus
}
#endif

#endif /* OSCILLATOR_H */
