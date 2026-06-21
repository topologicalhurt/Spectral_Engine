/* spectral_segment_math.h - Canonical segment math helpers shared by backends */
#ifndef SPECTRAL_SEGMENT_MATH_H
#define SPECTRAL_SEGMENT_MATH_H

#include <stddef.h>

/* Bump when any formula below changes.  metal_osc.py regenerates these into
 * drivers/metal/spectral_osc_metal_generated.h (stamped segment_math_version);
 * verify_metal_osc fails the build if the committed MSL drifts from this contract. */
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

/* Cubic (McAulay-Quatieri 1986, eq. 33-37) phase interpolation.
 * theta(t) = phase0 + t*(alpha + t*(c2 + t*c3)), Horner form.
 * Reduces exactly to the quadratic helper when c2==beta and c3==0:
 * IEEE float multiply commutes, so beta*offset == offset*beta and the extra
 * (c3==0) term contributes a hard +0.0f, leaving the result bit-identical. */
SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_phase_at_cubic_f32(
    float phase0, float alpha, float c2, float c3, float sample_offset)
{
    return phase0 + sample_offset * (alpha + sample_offset * (c2 + sample_offset * c3));
}

SPECTRAL_SEGMENT_MATH_INLINE float spectral_segment_amp_at_f32(
    float amp0, float d_amp, float sample_offset)
{
    return amp0 + d_amp * sample_offset;
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SEGMENT_MATH_H */
