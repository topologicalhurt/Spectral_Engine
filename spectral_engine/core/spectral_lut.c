/* spectral_lut.c - Q15 LUT Oscillator */
#include "spectral_lut.h"
#include "spectral_config.h"
#include <math.h>

void spectral_lut_init_sine(q15_t* lut) {
    if (!lut) return;
    
    const float scale = 32700.0f;
    for (uint32_t i = 0; i <= SPECTRAL_OSC_LUT_SIZE; i++) {
        float phase = (float)i / (float)SPECTRAL_OSC_LUT_SIZE * SPECTRAL_TWO_PI;
        lut[i] = (q15_t)(sinf(phase) * scale);
    }
    
    /* Force exact cardinal values */
    lut[0] = 0;
    lut[SPECTRAL_OSC_LUT_SIZE / 2] = 0;
    lut[SPECTRAL_OSC_LUT_SIZE / 4] = (q15_t)scale;
    lut[3 * SPECTRAL_OSC_LUT_SIZE / 4] = (q15_t)(-scale);
}

q15_t spectral_lut_sin(uq16_t phase_u16, const q15_t* lut) {
    uint32_t idx = phase_u16 >> (16 - SPECTRAL_OSC_LUT_BITS);
    uint32_t frac_raw = phase_u16 & SPECTRAL_OSC_FRAC_MASK;
    uint32_t frac = (SPECTRAL_OSC_FRAC_BITS >= 8) 
        ? (frac_raw >> (SPECTRAL_OSC_FRAC_BITS - 8)) 
        : (frac_raw << (8 - SPECTRAL_OSC_FRAC_BITS));
    
    q15_t s0 = lut[idx];
    q15_t s1 = lut[idx + 1];
    return (q15_t)(s0 + ((((q31_t)s1 - (q31_t)s0) * (int32_t)frac) >> 8));
}

q15_t spectral_lut_cos(uq16_t phase_u16, const q15_t* lut) {
    return spectral_lut_sin(phase_u16 + 16384, lut);
}
