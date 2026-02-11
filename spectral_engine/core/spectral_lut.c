/* spectral_lut.c - Q15 LUT Oscillator */
#include "spectral_lut.h"
#include "spectral_config.h"
#include <math.h>

/*
 * Flash-resident LUT option: when SPECTRAL_LUT_IN_FLASH is set,
 * the LUT is pre-computed at build time (by tools/gen_lut.py) and
 * placed in flash via linker section, saving ~8KB of SRAM.
 *
 * When not set, spectral_lut_init_sine() computes the table at runtime.
 */
#if SPECTRAL_LUT_IN_FLASH
#include "spectral_lut_data.h"
#else

void spectral_lut_init_sine(q15_t* lut) {
    if (!lut) return;

    /* Q15_MAX (32767) minus headroom to prevent linear interpolation overflow */
    const float scale = SPECTRAL_LUT_AMP_SCALE;
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

#endif /* !SPECTRAL_LUT_IN_FLASH */

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
