/* spectral_lut.h - Q15 LUT Oscillator */
#ifndef SPECTRAL_LUT_H
#define SPECTRAL_LUT_H

#include "spectral_q.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SPECTRAL_OSC_LUT_BITS
#define SPECTRAL_OSC_LUT_BITS   12
#endif

#if SPECTRAL_OSC_LUT_BITS < 1 || SPECTRAL_OSC_LUT_BITS > 16
#error "SPECTRAL_OSC_LUT_BITS must be in [1,16] for uq16_t phase lookup"
#endif

#define SPECTRAL_OSC_LUT_SIZE   (1u << SPECTRAL_OSC_LUT_BITS)
#define SPECTRAL_OSC_FRAC_BITS  (16u - SPECTRAL_OSC_LUT_BITS)
#define SPECTRAL_OSC_FRAC_MASK  ((1u << SPECTRAL_OSC_FRAC_BITS) - 1u)

/*
 * Flash-resident LUT: when SPECTRAL_LUT_IN_FLASH is defined, the LUT is
 * pre-computed at build time (tools/spectral_tools/generators/lut_generator.py)
 * and stored in flash.
 * Use SPECTRAL_DEFAULT_LUT to get a pointer to the appropriate table.
 */
#ifndef SPECTRAL_LUT_IN_FLASH
#define SPECTRAL_LUT_IN_FLASH   0
#endif

#if SPECTRAL_LUT_IN_FLASH
extern const q15_t spectral_lut_flash[SPECTRAL_OSC_LUT_SIZE + 1];
#define SPECTRAL_DEFAULT_LUT    spectral_lut_flash
#else
void spectral_lut_init_sine(q15_t* lut);
/* Caller must provide their own q15_t array and call spectral_lut_init_sine() */
#endif

/* Inline LUT lookup — eliminates function-call overhead in hot loops */
static inline q15_t spectral_lut_sin(uq16_t phase_u16, const q15_t* lut) {
    if (!lut) return Q15_ZERO;

    uint32_t idx = ((uint32_t)phase_u16) >> (16u - SPECTRAL_OSC_LUT_BITS);
    uint32_t frac_raw = ((uint32_t)phase_u16) & SPECTRAL_OSC_FRAC_MASK;
    uint32_t frac = (SPECTRAL_OSC_FRAC_BITS >= 8u)
        ? (frac_raw >> (SPECTRAL_OSC_FRAC_BITS - 8u))
        : (frac_raw << (8u - SPECTRAL_OSC_FRAC_BITS));
    q15_t s0 = lut[idx];
    q15_t s1 = lut[idx + 1u];
    return (q15_t)(s0 + ((((q31_t)s1 - (q31_t)s0) * (int32_t)frac) >> 8));
}

static inline q15_t spectral_lut_cos(uq16_t phase_u16, const q15_t* lut) {
    /* cos(x) = sin(x + pi/2); 16384 = 65536/4 is a quarter turn in the uint16 phase. */
    return spectral_lut_sin((uq16_t)(phase_u16 + (uq16_t)16384u), lut);
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_LUT_H */
