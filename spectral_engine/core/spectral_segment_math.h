/* spectral_segment_math.h - Canonical segment math helpers shared by backends */
#ifndef SPECTRAL_SEGMENT_MATH_H
#define SPECTRAL_SEGMENT_MATH_H

#include <stddef.h>

/* Bump when any formula below changes.  Metal shader (spectral_synth_metal.m)
 * duplicates these formulas as MSL strings and checks this version at
 * compile time.  A mismatch means the Metal copy is stale. */
#define SPECTRAL_SEGMENT_MATH_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__CUDACC__)
#define SPECTRAL_SEGMENT_MATH_INLINE static __host__ __device__ __forceinline__
#elif defined(__GNUC__) || defined(__clang__)
#define SPECTRAL_SEGMENT_MATH_INLINE static inline __attribute__((always_inline))
#else
#define SPECTRAL_SEGMENT_MATH_INLINE static inline
#endif

SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_alpha_f32(
    float omega, float pitch_factor, float inv_stretch)
{
    return omega * pitch_factor * inv_stretch;
}

SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_beta_f32(
    float df, float pitch_factor, float inv_stretch_sq)
{
    return df * pitch_factor * inv_stretch_sq;
}

SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_d_amp_f32(float da, float inv_stretch) {
    return da * inv_stretch;
}

SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_phase_at_f32(
    float phase0, float alpha, float beta, float sample_offset)
{
    return phase0 + sample_offset * (alpha + beta * sample_offset);
}

SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_amp_at_f32(
    float amp0, float d_amp, float sample_offset)
{
    return amp0 + d_amp * sample_offset;
}

float spectral_segment_phase_at_index_f32(float phase0, float alpha, float beta, size_t sample_index);
float spectral_segment_amp_at_index_f32(float amp0, float d_amp, size_t sample_index);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SEGMENT_MATH_H */
