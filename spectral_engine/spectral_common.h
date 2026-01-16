/* spectral_common.h - Core Types and Fast Math */
#ifndef SPECTRAL_COMMON_H
#define SPECTRAL_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "spectral_config.h"

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

/* 64-byte segment (desktop) - cache-line aligned */
typedef struct __attribute__((aligned(64))) {
    float start, length, phase, freq_hz, df, amp, da;
    union {
        float _pad[9];
        struct { float width; float _pad_w[8]; };
    };
} Segment;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(Segment) == 64, "Segment size");
#endif

/* 32-byte segment (embedded) */
typedef struct __attribute__((aligned(4))) {
    float start, length, phase, freq_hz, df, amp, da;
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

#if !SPECTRAL_EMBEDDED
#define PREFETCH_READ(addr)  __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#else
#define PREFETCH_READ(addr)  ((void)0)
#define PREFETCH_WRITE(addr) ((void)0)
#endif

typedef struct SegmentArray {
    Segment* segs;
    uint32_t count;
    uint32_t capacity;
} SegmentArray;

typedef struct {
    float stretch, inv_stretch, inv_stretch_sq, pitch_factor;
    size_t out_len;
    uint32_t num_segments;
} SynthParams;

#ifndef __CUDACC__
void* spectral_aligned_alloc(size_t size);
float fast_atan2(float y, float x);
float fast_sin(float x);
SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs);
#endif

#ifdef __CUDACC__
__device__ __forceinline__ float fast_sin_device(float x);
#endif

#if !SPECTRAL_NO_PERF
#include "spectral_perf.h"
#endif

#endif
