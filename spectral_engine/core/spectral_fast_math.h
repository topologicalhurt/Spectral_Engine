/* spectral_fast_math.h - Fast Math Approximations
 *
 * Provides shared exact-by-default math wrappers for oscillator and analysis
 * modules. Approximation flags are intentionally centralized here so peak,
 * vector and synth code do not grow local copies of the same bit hacks.
 * GPU shaders must match fast_sin.
 */
#ifndef SPECTRAL_FAST_MATH_H
#define SPECTRAL_FAST_MATH_H

#include "spectral_config.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define FAST_MATH_HOT __attribute__((hot))
#else
#define FAST_MATH_HOT
#endif

FAST_MATH_HOT float fast_sin(float x);
FAST_MATH_HOT float fast_inv_sqrt(float x);
FAST_MATH_HOT float fast_sqrt(float x);
FAST_MATH_HOT float fast_peak_log(float x);
float fast_atan2(float y, float x);
float phase_to_rads(float p);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_FAST_MATH_H */
