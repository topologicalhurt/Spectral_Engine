/* spectral_common.h - Core Types and Fast Math */
#ifndef SPECTRAL_COMMON_H
#define SPECTRAL_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include "spectral_config.h"

/* Convert semitones to pitch multiplier: 2^(semitones/12) */
#define SPECTRAL_PITCH_FACTOR(semitones) powf(2.0f, (semitones) / 12.0f)

/* Synthesis output cleanup on error */
static inline void synth_fail_cleanup(float* buf, size_t len, double* t_synth) {
    if (buf) memset(buf, 0, len * sizeof(float));
    if (t_synth) *t_synth = 0.0;
}

/* 64-byte segment (desktop) - cache-line aligned
 * 
 * Fields:
 *   start  - Start sample index (pre-stretch)
 *   length - Duration in samples (pre-stretch)
 *   phase  - Initial phase in radians
 *   omega  - Angular frequency (radians per sample, pre-pitch/stretch)
 *   df     - Frequency delta per sample (chirp rate)
 *   amp    - Amplitude [0, 1]
 *   da     - Amplitude delta per sample
 *   width  - Timbre-specific parameter (PWM duty, quantization level)
 */
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

typedef struct {
    float stretch, inv_stretch, inv_stretch_sq, pitch_factor;
    size_t out_len;
    uint32_t num_segments;
} SynthParams;

#ifndef __CUDACC__
void* spectral_aligned_alloc(size_t size);
float fast_atan2(float y, float x);
float phase_to_rads(float p);
SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs);
#endif

#if !SPECTRAL_NO_PERF
#include "spectral_perf.h"
#endif

#endif
