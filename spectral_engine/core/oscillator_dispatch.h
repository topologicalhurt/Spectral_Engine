#ifndef OSCILLATOR_DISPATCH_H
#define OSCILLATOR_DISPATCH_H

#include <stdint.h>
#include "spectral_config.h"

/* =====================================================================
 * OSCILLATOR BACKEND x COMPUTE-DOMAIN x TARGET MATRIX  (authoritative)
 * ---------------------------------------------------------------------
 * North star: float is the DEFAULT compute domain everywhere.  Q15's
 * home is embedded; on the desktop Q15 is OPT-IN (--q15) and the default
 * render stays byte-identical when Q15 is not requested.
 *
 *   backend  | float | Q15            | runs on        | source
 *   ---------+-------+----------------+----------------+--------------------------
 *   Scalar   |  yes  | yes            | host + embedded| spectral_osc_formulas.h (f32)
 *            |       |                |                | spectral_osc_q15.h      (Q15)
 *   SIMDe    |  yes  | yes (pack8 8xQ15)| HOST ONLY    | arch/simd/oscillator_simd.c
 *   CMSIS    |  yes  | yes (Ph.2)+    | EMBEDDED ONLY  | arch/arm/oscillator_simd.c
 *   GPU      |  yes  | float-only*     | host (Metal) / | Metal MSL + CUDA (codegen
 *            |       |                | CUDA dev       | from spectral_osc_formulas.h)
 *   vDSP     |  n/a  | n/a            | host (Apple)   | NOT an oscillator backend --
 *            |       |                |                | analysis/FFT + math-accel only
 *
 *   * GPU Q15 double-pack (Metal half / CUDA __half2) is a measure-first
 *     Phase-5 investigation; default stays float.
 *   + CMSIS-Q15 (osc_simd_q15_segment under OSC_SIMD_CMSIS) shares the canonical
 *     spectral_osc_q15.h evaluators (parity by construction).  It is BUILT only on
 *     real Cortex-M and is NOT yet wired into the live embedded dispatch -- see the
 *     prohibit/embedded notes below.
 *
 * WHICH PATHS PROHIBIT Q15:
 *   - vDSP   : not an oscillator at all (FFT/window/math-accel only, Phase 4).
 *   - GPU    : float-only by default (Q15 double-pack is gated on a proven win).
 *   - CMSIS  : has BOTH float (arm_*_f32) and Q15 (Phase 2, shared canonical contract);
 *              the Q15 body is hardware-gated (OSC_SIMD_CMSIS = real Cortex-M, not the
 *              host sim, and not compile-checked locally -- no arm_math.h here).
 *
 * WHAT AN EMBEDDED BUILD CAN TAKE (oscillator paths on real Cortex-M):
 *   - Scalar  : float OR Q15 (spectral_osc_q15.h evaluators).  Universal.
 *   - CMSIS   : float (live); Q15 kernel present (Phase 2) but not yet dispatched --
 *               oscillator.c's Q15 path is deliberately #if !SPECTRAL_EMBEDDED, so
 *               today embedded Q15 oscillator synthesis is owned by arm32 below.
 *               Promoting CMSIS-Q15 into the live dispatch changes shipped embedded
 *               behavior and needs explicit sign-off.  Cortex-M4/M7/ARMv8-M only.
 *   - arm32   : Q15 fixed-point synth (arch/arm/spectral_synth_arm32.c),
 *               sine-only via spectral_lut_sin over the production 32700-scale LUT.
 *   SIMDe is HOST-ONLY: it is the desktop default CPU path and also runs in the
 *   embedded-*simulation* build (itself a host build), but on real Cortex-M the
 *   dispatch below selects CMSIS in its place (OSC_SIMD_CMSIS).  So every embedded
 *   oscillator path can run pure Q15 (Scalar-Q15 or arm32-Q15 today; CMSIS-Q15 Ph.2).
 *
 * The Q15 WAVEFORM contract is spectral_osc_q15.h (versioned: SPECTRAL_OSC_Q15_VERSION);
 * the float WAVEFORM contract is spectral_osc_formulas.h (SPECTRAL_OSC_FORMULAS_VERSION).
 * ===================================================================== */

/* SIMD backend: CMSIS on ARM embedded, SIMDe SSE on desktop (maps to NEON/SSE/scalar) */
#if defined(ARM_MATH_CM4) || defined(ARM_MATH_CM7) || defined(ARM_MATH_ARMV8MML)
    #define OSC_SIMD_CMSIS 1
    #include "arm_math.h"
#else
    #define OSC_SIMD_GENERIC 1
    /* SIMDe: write SSE2 intrinsics, compiles to native NEON/SSE/scalar */
    #include "simde/x86/sse2.h"
    #ifdef __AVX2__
        #include "simde/x86/avx2.h"
    #endif
#endif

#ifndef __CUDACC__
/* Q15 leaf evaluators for spectral_osc_q15_eval below. Included AFTER the CMSIS
 * arm_math.h above (not with the other top includes): arm_math_types.h defines
 * the Q15/Q31 range macros UNGUARDED, so CMSIS must precede the #ifndef-guarded
 * copies in spectral_q.h or the bare-metal build redefines them under -Werror. */
#include "spectral_osc_q15.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __CUDACC__
/* Canonical Q15 timbre dispatch: map a timbre to its spectral_osc_q15.h leaf
 * evaluator. One definition shared by every Q15 backend's scalar/fade path (the
 * core scalar oracle, the SIMDe pack8 fade tail, the CMSIS embedded sibling), so
 * the timbre->waveform map cannot drift between them. Timbres with no Q15
 * evaluator (asin/quantized/pwm) return silence. The version pin makes this a
 * contract consumer: a SPECTRAL_OSC_Q15_VERSION bump fails the build here until
 * the dispatch is re-validated. */
_Static_assert(SPECTRAL_OSC_Q15_VERSION == 1,
               "spectral_osc_q15.h contract changed; re-validate spectral_osc_q15_eval");
static inline q15_t spectral_osc_q15_eval(q15_t pq, SpectralTimbre timbre, const q15_t* lut) {
    switch (timbre) {
    case TIMBRE_SINE:     return spectral_osc_q15_sine(pq, lut);
    case TIMBRE_SAW:      return spectral_osc_q15_saw(pq);
    case TIMBRE_SQUARE:   return spectral_osc_q15_square(pq);
    case TIMBRE_TRIANGLE: return spectral_osc_q15_triangle(pq);
    case TIMBRE_PARABOLA: return spectral_osc_q15_parabola(pq);
    default:              return Q15_ZERO;
    }
}
#endif /* !__CUDACC__ */

/* Dispatch mode: 2 bits per timbre */
typedef enum {
    OSC_MODE_CPU_SCALAR = 0,
    OSC_MODE_CPU_SIMD   = 1,
    OSC_MODE_FALLBACK   = 3,
} OscDispatchMode;

/* Explicit bitfield for per-timbre dispatch (8 timbres × 2 bits = 16 bits) */
typedef union {
    struct {
        uint16_t sine     : 2;
        uint16_t saw      : 2;
        uint16_t square   : 2;
        uint16_t triangle : 2;
        uint16_t asin     : 2;
        uint16_t parabola : 2;
        uint16_t quantized: 2;
        uint16_t pwm      : 2;
    };
    uint16_t word;
} OscDispatchWord;

/* Accessor macros for generic timbre index access */
#define OSC_GET_MODE(dispatch, timbre) \
    ((OscDispatchMode)(((dispatch).word >> ((timbre) * 2)) & 0x3))

#define OSC_SET_MODE(dispatch, timbre, mode) \
    ((dispatch).word = ((dispatch).word & ~(0x3u << ((timbre) * 2))) | ((uint16_t)(mode) << ((timbre) * 2)))

/* Presets */
#define OSC_DISPATCH_ALL_SCALAR   ((OscDispatchWord){ .word = 0x0000 })
#define OSC_DISPATCH_ALL_SIMD     ((OscDispatchWord){ .word = 0x5555 })
#define OSC_DISPATCH_ALL_FALLBACK ((OscDispatchWord){ .word = 0xFFFF })

/* SIMD vector width (floats per vector register) */
#if defined(OSC_SIMD_CMSIS)
    #define OSC_SIMD_WIDTH 4  /* Cortex-M4/M7 with FPU */
#elif defined(__AVX2__) || defined(__AVX__)
    #define OSC_SIMD_WIDTH 8
#else
    #define OSC_SIMD_WIDTH 4  /* SSE2/NEON via SIMDe */
#endif

/* Forward declarations */
struct SegmentLoopParams;

/* SIMD segment synthesis - platform-specific implementations in oscillator_simd.c */
void osc_simd_segment_sine(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_saw(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_square(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_triangle(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_parabola(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_quantized(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_pwm(float* dst, const struct SegmentLoopParams* lp);

/* Query if SIMD is available for a timbre (some timbres may not have SIMD paths) */
int osc_simd_available(SpectralTimbre timbre);

#if defined(OSC_SIMD_GENERIC)
/* Packed 8xQ15 SIMD oscillator - host/SIMDe only, opt-in via the Q15
 * compute domain. osc_simd_q15_available() advertises the 4 algebraic timbres
 * plus sine (B1): sine's eval is a serial LUT gather using sine_lut, the algebraic
 * timbres ignore it but keep the param for a uniform call site. */
int osc_simd_q15_available(SpectralTimbre timbre);
void osc_simd_q15_segment(float* dst, const struct SegmentLoopParams* lp,
                          SpectralTimbre timbre, const int16_t* sine_lut);
#endif

#if defined(OSC_SIMD_CMSIS)
/* CMSIS-Q15 oscillator (Phase 2) - the EMBEDDED Q15 compute path. Same signature
 * as the SIMDe pack8 twin above so the Q15 dispatch is backend-uniform: it renders
 * the waveform through the SAME canonical spectral_osc_q15.h evaluators (one
 * versioned contract, no drift), with float quadratic phase + CMSIS-DSP f32 widen/
 * amp/accumulate. Verification is hardware-gated (real Cortex-M); it has no LIVE
 * caller yet (oscillator.c's Q15 dispatch is host-only by design) -- wiring it
 * into the embedded dispatch changes shipped behavior and needs explicit
 * sign-off. See arch/arm/oscillator_simd.c. */
int osc_simd_q15_available(SpectralTimbre timbre);
void osc_simd_q15_segment(float* dst, const struct SegmentLoopParams* lp,
                          SpectralTimbre timbre, const int16_t* sine_lut);
#endif

#ifdef __cplusplus
}
#endif

#endif /* OSCILLATOR_DISPATCH_H */
