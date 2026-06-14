/* spectral_consts.h - Mathematical and Physical Constants */
#ifndef SPECTRAL_CONSTS_H
#define SPECTRAL_CONSTS_H

/* Math constants (float precision) */
#define SPECTRAL_PI             3.14159265358979323846f
/* Double-precision pi for init-time table construction (IFFT motif etc.). */
#define SPECTRAL_PI_D           3.14159265358979323846
#define SPECTRAL_TWO_PI         6.283185307179586f
#define SPECTRAL_INV_PI         0.31830988618379067f
#define SPECTRAL_INV_TWO_PI     0.159154943091895f
#define SPECTRAL_INV_PI_SQ      0.10132118364233778f

/* IEEE-754 exact float values for atan2/trig fast paths */
#define SPECTRAL_HALF_PI        1.57079637f
#define SPECTRAL_PI_F           3.14159274f

/* Minimax atan coefficients (Horner in s = a^2, ratio a in [0,1]). Consumed only
 * by spectral_atan2_poly (spectral_fast_math.h), which documents the argument
 * reduction, the odd-polynomial form, and the octant reconstruction. GPU/SIMD
 * copies MUST use these same values (parity-tested against atan2f). */
#define SPECTRAL_ATAN2_A0       (-0.0464964749f)
#define SPECTRAL_ATAN2_A1       0.15931422f
#define SPECTRAL_ATAN2_A2       (-0.327622764f)
#define SPECTRAL_ATAN2_EPS      1e-10f

/* Peak-tracker interpolation constants */
#define SPECTRAL_TRACK_LOG_FLOOR            1e-30f
#define SPECTRAL_TRACK_PARABOLIC_DENOM_EPS  1e-20f

/* Q15 sine-LUT amplitude scale. Held below full scale (32767) by ~-0.02 dB to
 * leave overflow headroom for linear interpolation between table entries -- see
 * spectral_osc_q15.h. */
#define SPECTRAL_LUT_AMP_SCALE  32700.0f

/* Q15/Q31 conversion constants */
#define SPECTRAL_Q15_SCALE      32768.0f
#define SPECTRAL_INV_Q15_SCALE  3.0517578125e-5f
#define SPECTRAL_Q31_SCALE      2147483648.0f
#define SPECTRAL_INV_Q31_SCALE  4.6566128730773926e-10f

/* Degree-9 odd minimax polynomial coefficients for sin(x) folded to [-pi/2, pi/2]:
 *   sin(x) ~ x * (1 + x^2*(C3 + x^2*(C5 + x^2*(C7 + x^2*C9)))).
 * SINGLE SOURCE for the scalar fast_sin (spectral_osc_formulas.h) and its SIMD twin
 * (arch/simd/spectral_oscillator_simd_kernel.inc); ~1.4 ULP vs libm over the oscillator's
 * operating range. Re-tune in ONE place so the two paths cannot drift. */
#define SPECTRAL_MINIMAX_SIN_C3  (-0.16666647791862488f)
#define SPECTRAL_MINIMAX_SIN_C5  ( 0.00833289884030819f)
#define SPECTRAL_MINIMAX_SIN_C7  (-0.0001980086526600644f)
#define SPECTRAL_MINIMAX_SIN_C9  ( 0.0000025904300855472684f)

/* Generalized-cosine window coefficients (portable path; vDSP carries its own).
 *   Hamming:  w[n] = A0 - A1*cos(2*pi*n/(N-1))
 *   Blackman: w[n] = B0 - B1*cos(theta) + B2*cos(2*theta) */
#define SPECTRAL_HAMMING_A0   0.54f
#define SPECTRAL_HAMMING_A1   0.46f
#define SPECTRAL_BLACKMAN_B0  0.42f
#define SPECTRAL_BLACKMAN_B1  0.5f
#define SPECTRAL_BLACKMAN_B2  0.08f

/* A sub-bin peak-frequency estimate is clamped to +/- half a bin. */
#define SPECTRAL_PEAK_BIN_OFFSET_MAX  0.5f

#endif /* SPECTRAL_CONSTS_H */
