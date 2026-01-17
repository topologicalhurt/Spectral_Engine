#ifndef OSCILLATOR_DISPATCH_H
#define OSCILLATOR_DISPATCH_H

#include "spectral_config.h"
#include <stdint.h>

/* Dispatch mode: 2 bits per timbre */
typedef enum {
    OSC_MODE_CPU_SCALAR = 0,
    OSC_MODE_CPU_SIMD   = 1,
    OSC_MODE_NATIVE     = 2,
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
#define OSC_DISPATCH_ALL_NATIVE   ((OscDispatchWord){ .word = 0xAAAA })
#define OSC_DISPATCH_ALL_FALLBACK ((OscDispatchWord){ .word = 0xFFFF })

/* SIMD vector width (floats per vector register) */
#if defined(OSC_SIMD_AVX)
    #define OSC_SIMD_WIDTH 8
#elif defined(OSC_SIMD_SSE) || defined(OSC_SIMD_NEON) || defined(OSC_SIMD_VDSP)
    #define OSC_SIMD_WIDTH 4
#elif defined(OSC_SIMD_CMSIS)
    #define OSC_SIMD_WIDTH 4  /* Cortex-M4/M7 with FPU */
#else
    #define OSC_SIMD_WIDTH 1
#endif

/* Forward declarations */
struct SegmentLoopParams;

/* SIMD segment synthesis - platform-specific implementations in oscillator_simd.c */
void osc_simd_segment_sine(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_saw(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_square(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_triangle(float* dst, const struct SegmentLoopParams* lp);
void osc_simd_segment_parabola(float* dst, const struct SegmentLoopParams* lp);

/* Query if SIMD is available for a timbre (some timbres may not have SIMD paths) */
int osc_simd_available(SpectralTimbre timbre);

/* Query if native backend is available */
int osc_native_available(void);

#endif /* OSCILLATOR_DISPATCH_H */
