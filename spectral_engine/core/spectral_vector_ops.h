/* spectral_vector_ops.h - Platform-Agnostic SIMD Vector Operations
 *
 * Provides vectorized math operations using SIMDe SSE intrinsics.
 * On macOS ARM -> NEON, Linux x86 -> native SSE, elsewhere -> scalar fallback.
 *
 * Replaces platform-specific vDSP calls and scalar fallback loops in
 * spectral_analysis.c, spectral_out.c, and spectral_synth_cpu.c.
 */
#ifndef SPECTRAL_VECTOR_OPS_H
#define SPECTRAL_VECTOR_OPS_H

#include "spectral_config.h"
#include <stddef.h>

/* Only use SIMD vector ops on desktop/simulation (not bare-metal CMSIS targets) */
#if (!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && !defined(ARM_MATH_CM4) && !defined(ARM_MATH_CM7)

#ifdef __cplusplus
extern "C" {
#endif

void spectral_vmul(const float* a, const float* b, float* dst, size_t len);
void spectral_vadd(const float* a, const float* b, float* dst, size_t len);
void spectral_vsq(const float* src, float* dst, size_t len);
void spectral_vmax(const float* src, float* out, size_t len);
void spectral_vmaxmgv(const float* src, float* out, size_t len);
void spectral_vsmul(const float* src, float scalar, float* dst, size_t len);
void spectral_vatan2(const float* y, const float* x, float* dst, size_t len);
void spectral_deinterleave(const float* interleaved,
                            float* re, float* im, size_t count);
void spectral_magsq_phase(const float* interleaved,
                           float* magsq, float* phase,
                           float* max_magsq, size_t count);
void spectral_magsq_only(const float* interleaved,
                          float* magsq, float* max_magsq, size_t count);
void spectral_magsq_split(const float* re, const float* im,
                           float* dst, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* (!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && !CMSIS */

#endif /* SPECTRAL_VECTOR_OPS_H */
