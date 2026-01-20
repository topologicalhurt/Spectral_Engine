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

/* Q15/Q31 conversion constants */
#define SPECTRAL_Q15_SCALE      32768.0f
#define SPECTRAL_INV_Q15_SCALE  3.0517578125e-5f
#define SPECTRAL_Q31_SCALE      2147483648.0f
#define SPECTRAL_INV_Q31_SCALE  4.6566128730773926e-10f

/* Q31 phase conversion: radians to Q31 fixed-point increment
 * Q31 uses full 32-bit range: 2^32 steps per 2*pi radians.
 * Used for high-precision phase accumulators in embedded synth. */
#define SPECTRAL_Q31_PER_RAD    (4294967296.0 / SPECTRAL_TWO_PI)  /* ~683565275.6 */

/* Short aliases for convenience (guarded to avoid conflicts) */
#ifndef PI
#define PI          SPECTRAL_PI
#endif
#ifndef TWO_PI
#define TWO_PI      SPECTRAL_TWO_PI
#endif
#ifndef INV_TWO_PI
#define INV_TWO_PI  SPECTRAL_INV_TWO_PI
#endif
#ifndef PI_SQ
#define PI_SQ       SPECTRAL_PI_SQ
#endif

#endif /* SPECTRAL_CONSTS_H */
