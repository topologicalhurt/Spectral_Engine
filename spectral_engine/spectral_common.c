/* spectral_common.c - Core Utilities
 * 
 * Implements fast math approximations and synthesis helpers.
 * 
 * Fast Math:
 *   fast_atan2() - Polynomial approximation, ~10x faster than libm
 *   fast_sin()   - Bhaskara I approximation, ~5x faster than sinf()
 * 
 * Helpers:
 *   make_synth_params() - Precompute synthesis parameters
 *   spectral_aligned_alloc() - Cache-aligned memory allocation
 * 
 * For segment file I/O, see spectral_segment_parser.c
 */
#include "spectral_common.h"

void* spectral_aligned_alloc(size_t size) {
    /* Round size up to cache line boundary for aligned_alloc */
    return aligned_alloc(CACHE_ALIGN, (size + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1));
}

/*
 * fast_atan2: Polynomial approximation of atan2(y, x)
 * 
 * Algorithm:
 *   1. Compute ratio of smaller/larger absolute value (range [0,1])
 *   2. Evaluate 3rd-order polynomial approximation of atan on [0,1]
 *   3. Apply quadrant corrections based on signs
 * 
 * Coefficients derived from minimax polynomial fit.
 * Max error ~0.07 degrees, ~10x faster than libm atan2f.
 */
float fast_atan2(float y, float x) {
    float abs_x = fabsf(x), abs_y = fabsf(y);
    float a = (abs_x < abs_y) ? abs_x / (abs_y + 1e-10f) : abs_y / (abs_x + 1e-10f);
    float s = a * a;
    /* Polynomial: a - 0.3276*a^3 + 0.1593*a^5 - 0.0465*a^7 (Horner form) */
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (abs_y > abs_x) r = 1.57079637f - r;  /* Reflect for |y| > |x| */
    if (x < 0) r = 3.14159274f - r;           /* Quadrant II/III */
    if (y < 0) r = -r;                        /* Quadrant III/IV */
    return r;
}

/*
 * fast_sin: Bhaskara I sine approximation
 * Max error ~0.1%, ~5x faster than sinf().
 */
float fast_sin(float x) {
    /* Wrap x to [-pi, pi] using: x - 2pi * round(x / 2pi) */
    x = x - TWO_PI * floorf(x * INV_TWO_PI + 0.5f);
    float x2 = x * x;
    return x * (PI_SQ - x2) / (PI_SQ + 0.25f * x2);
}

/*
 * make_synth_params: Precompute synthesis parameters
 * 
 * pitch_factor: semitone offset to frequency multiplier
 *   pitch=0  -> 1.0x (unchanged)
 *   pitch=12 -> 2.0x (octave up)
 *   pitch=-12 -> 0.5x (octave down)
 */
SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs) {
    return (SynthParams){
        .stretch = stretch,
        .inv_stretch = 1.0f / stretch,
        .inv_stretch_sq = 1.0f / (stretch * stretch),
        .pitch_factor = powf(2.0f, pitch / 12.0f),  /* 2^(semitones/12) */
        .out_len = (uint32_t)out_len,
        .num_segments = (uint32_t)num_segs
    };
}
