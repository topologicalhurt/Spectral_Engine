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

/* Convert semitones to pitch multiplier: 2^(semitones/12) */
#define SPECTRAL_PITCH_FACTOR(semitones) powf(2.0f, (semitones) / 12.0f)

/* 64-byte segment (desktop) - cache-line aligned */
typedef struct __attribute__((aligned(64))) {
    float start, length, phase, omega, df, amp, da;
    union {
        float _pad[9];
        struct { float width; float _pad_w[8]; };
    };
} Segment;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(Segment) == 64, "Segment size");
#endif

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
