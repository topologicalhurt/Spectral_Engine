/* spectral_consts.h - Mathematical and Physical Constants */
#ifndef SPECTRAL_CONSTS_H
#define SPECTRAL_CONSTS_H

/* Math constants (float precision) */
#define SPECTRAL_PI             3.14159265358979323846f
#define SPECTRAL_TWO_PI         6.283185307179586f
#define SPECTRAL_INV_PI         0.31830988618379067f
#define SPECTRAL_INV_TWO_PI     0.159154943091895f
#define SPECTRAL_TWO_INV_PI     0.6366197723675814f
#define SPECTRAL_PI_SQ          9.8696044f
#define SPECTRAL_INV_PI_SQ      0.10132118364233778f

/* IEEE-754 exact float values for atan2/trig fast paths */
#define SPECTRAL_HALF_PI        1.57079637f
#define SPECTRAL_PI_F           3.14159274f

/* Pade [5/4] sine coefficients -- GPU shaders MUST use these same values */
#define SPECTRAL_PADE_SIN_C1    0.16605f
#define SPECTRAL_PADE_SIN_C2    0.00761f
#define SPECTRAL_PADE_SIN_C3    0.00766f

/* Polynomial atan2 coefficients */
#define SPECTRAL_ATAN2_A0       (-0.0464964749f)
#define SPECTRAL_ATAN2_A1       0.15931422f
#define SPECTRAL_ATAN2_A2       (-0.327622764f)
#define SPECTRAL_ATAN2_EPS      1e-10f

/* Q30 scale (2^30) for Q15*Q15 accumulator conversion */
#define SPECTRAL_Q30_SCALE      1073741824.0
#define SPECTRAL_INV_Q30_SCALE  (1.0 / SPECTRAL_Q30_SCALE)

/* LUT amplitude scale */
#define SPECTRAL_LUT_AMP_SCALE  32700.0f

/* Q15/Q31 conversion constants */
#define SPECTRAL_Q15_SCALE      32768.0f
#define SPECTRAL_INV_Q15_SCALE  3.0517578125e-5f
#define SPECTRAL_Q31_SCALE      2147483648.0f
#define SPECTRAL_INV_Q31_SCALE  4.6566128730773926e-10f

/* Q31 phase conversion: radians to Q31 fixed-point increment
 * Q31 uses full 32-bit range: 2^32 steps per 2*pi radians.
 * Used for high-precision phase accumulators in embedded synth. */
#define SPECTRAL_Q31_PER_RAD    (4294967296.0 / SPECTRAL_TWO_PI)  /* ~683565275.6 */

#endif /* SPECTRAL_CONSTS_H */
