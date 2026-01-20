/* spectral_lut.h - Q15 LUT Oscillator */
#ifndef SPECTRAL_LUT_H
#define SPECTRAL_LUT_H

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

void spectral_lut_init_sine(q15_t* lut);
q15_t spectral_lut_sin(uq16_t phase_u16, const q15_t* lut);
q15_t spectral_lut_cos(uq16_t phase_u16, const q15_t* lut);

#ifdef __cplusplus
}
#endif

#endif
