/* spectral_common.h - Core Types and Shared Segment Contracts */
#ifndef SPECTRAL_COMMON_H
#define SPECTRAL_COMMON_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#include "spectral_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Half-precision storage for the STFT complex spectrum (phase-at-peaks). re/im
 * are consumed only as atan2f(im,re) at tracked peak bins, where fp16's ~3e-4
 * rad angle error is ~-99.6 dBFS (inaudible, parity-passing). Storing them as
 * fp16 makes the magsq+re+im layout memory- AND bandwidth-neutral vs the old
 * magsq+phase (8 bytes/bin either way), so phase-at-peaks costs no extra RAM or
 * store traffic while still retiring the all-bins producer atan2. Falls back to
 * float on compilers without _Float16 (correct, but +50% on the full-matrix
 * path there). */
#if defined(__FLT16_MANT_DIG__)
typedef _Float16 SpectralHalf;
#define SPECTRAL_HALF_IS_FP16 1
#else
typedef float SpectralHalf;
#define SPECTRAL_HALF_IS_FP16 0
#endif

/* Convert semitones to pitch multiplier: 2^(semitones/12) */
static inline float spectral_pitch_factor(float semitones) {
    return powf(2.0f, semitones / 12.0f);
}

/* 64-byte segment (desktop) - cache-line aligned. One sinusoidal partial over a
 * time span: the analysis output and the synthesis input. Field units are the
 * engine-wide contract (AI_CANON 9) and must not drift across backends. */
typedef struct __attribute__((aligned(64))) {
    float start;   /* onset, samples */
    float length;  /* duration, samples */
    float phase;   /* initial phase, radians */
    float omega;   /* angular frequency, radians/sample */
    float df;      /* chirp rate d(omega)/dn, radians/sample^2 */
    float amp;     /* initial amplitude, linear [-1, 1] */
    float da;      /* amplitude slope, linear/sample */
    union {
        float _pad[9];
        /* width: timbre shape param (e.g. PWM duty) in [0,1]; _pad_w: cubic-phase
         * annotation words, see spectral_segment_set_cubic below. */
        struct { float width; float _pad_w[8]; };
    };
} Segment;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(Segment) == 64, "Segment size");
#endif

/* Cubic MQ phase annotation lives in the 64-byte Segment's spare pad words.
 * Layout (only the desktop/GPU 64B Segment has room; embedded 32B does not):
 *   _pad_w[0] = c2 (quadratic phase coeff, == beta when no cross-frame linkage)
 *   _pad_w[1] = c3 (cubic phase coeff, 0 when no cross-frame linkage)
 *   _pad_w[2] = annotation flag (non-zero => c2/c3 valid). finalize() memsets
 *               _pad_w to 0, so an un-annotated segment reads back as absent. */
static inline int spectral_segment_has_cubic(const Segment* seg) {
    return seg && seg->_pad_w[2] != 0.0f;
}
static inline void spectral_segment_set_cubic(Segment* seg, float c2, float c3) {
    if (!seg) return;
    seg->_pad_w[0] = c2;
    seg->_pad_w[1] = c3;
    seg->_pad_w[2] = 1.0f;
}
static inline float spectral_segment_cubic_c2(const Segment* seg) {
    return seg->_pad_w[0];
}
static inline float spectral_segment_cubic_c3(const Segment* seg) {
    return seg->_pad_w[1];
}

/* 32-byte segment (embedded) - see Segment for field documentation */
typedef struct __attribute__((aligned(4))) {
    float start, length, phase, omega, df, amp, da;
    union {
        float _pad[1];
        struct { float width; };
    };
} SegmentCompact;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SegmentCompact) == 32, "SegmentCompact size");
#endif

/* 32-byte GPU segment — packs only the 7 active fields used by GPU kernels
 * (start/length/phase/omega/df/amp/da; units mirror Segment, AI_CANON 9).
 * Stored in the segment cache for zero-copy GPU upload via mmap. The flat scalar
 * body is parsed by the Metal struct-mirror codegen (metal_osc.py) — keep it a
 * flat scalar list so the generated MSL mirror stays byte-equivalent. */
typedef struct __attribute__((aligned(4))) {
    float start, length, phase, omega, df, amp, da;
    float _pad;
} SegmentGpu;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SegmentGpu) == 32, "SegmentGpu size");
#endif

/* Pack a full Segment into the 32-byte GPU layout used by CUDA/Metal. */
static inline SegmentGpu spectral_segment_pack_gpu(const Segment* seg)
{
    SegmentGpu out = {0};
    if (!seg) return out;
    out.start  = seg->start;
    out.length = seg->length;
    out.phase  = seg->phase;
    out.omega  = seg->omega;
    out.df     = seg->df;
    out.amp    = seg->amp;
    out.da     = seg->da;
    out._pad   = 0.0f;
    return out;
}

static inline void spectral_segment_pack_gpu_array(
    const Segment* src, uint32_t count, SegmentGpu* dst)
{
    if (!src || !dst) return;
    for (uint32_t i = 0; i < count; i++) {
        dst[i] = spectral_segment_pack_gpu(&src[i]);
    }
}

#if SPECTRAL_COMPACT_SEG
typedef SegmentCompact SegmentActive;
#define SEGMENT_SIZE 32
#else
typedef Segment SegmentActive;
#define SEGMENT_SIZE 64
#endif

typedef struct SegmentArray {
    Segment* segs;
    uint32_t count;
    uint32_t capacity;
} SegmentArray;

#define SEGMENT_ARRAY_EMPTY  {NULL, 0, 0}

typedef struct {
    float stretch, inv_stretch, inv_stretch_sq, pitch_factor;
    size_t out_len;
    uint32_t num_segments;
} SynthParams;

#ifndef __CUDACC__
void* spectral_aligned_alloc(size_t size);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_COMMON_H */
