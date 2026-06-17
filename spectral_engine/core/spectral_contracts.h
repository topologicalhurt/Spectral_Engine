/* spectral_contracts.h - Canonical kernel validation contracts */
#ifndef SPECTRAL_CONTRACTS_H
#define SPECTRAL_CONTRACTS_H

#include "spectral_common.h"
#include "spectral_utils.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include <string.h>

static inline int spectral_f32_span_finite(const float* values, size_t count)
{
    if (count > 0u && !values) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!spectral_is_finite_f32(values[i])) return 0;
    }
    return 1;
}

static inline int spectral_f32_span_finite_nonnegative(const float* values, size_t count)
{
    if (count > 0u && !values) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!spectral_is_finite_f32(values[i]) || values[i] < 0.0f) return 0;
    }
    return 1;
}

static inline int spectral_segment_payload_valid(const Segment* s)
{
    if (!s) return 0;
    if (!spectral_is_finite_f32(s->start) || s->start < 0.0f) return 0;
    if (!spectral_is_finite_f32(s->length) || s->length < 0.0f) return 0;
    if (!spectral_is_finite_f32(s->phase)) return 0;
    if (!spectral_is_finite_f32(s->omega) || s->omega < 0.0f) return 0;
    if (!spectral_is_finite_f32(s->df)) return 0;
    if (!spectral_is_finite_f32(s->amp)) return 0;
    if (!spectral_is_finite_f32(s->da)) return 0;
    if (!spectral_is_finite_f32(s->width) || fabsf(s->width) > SPECTRAL_SEGMENT_WIDTH_MAX) return 0;
    return 1;
}

static inline int spectral_segment_valid_for_synth(const Segment* s)
{
    if (!spectral_segment_payload_valid(s)) return 0;
    if (s->length <= 0.0f) return 0;
    if (s->amp < 0.0f) return 0;
    return 1;
}

static inline int spectral_segment_array_payload_valid(const Segment* segs,
                                                       size_t count,
                                                       uint32_t* bad_idx)
{
    if (count > 0u && !segs) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!spectral_segment_payload_valid(&segs[i])) {
            if (bad_idx) *bad_idx = (i > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)i;
            return 0;
        }
    }
    return 1;
}

static inline int spectral_segment_array_valid_for_synth(const SegmentArray* sa)
{
    if (!sa) return 0;
    if ((size_t)sa->count > 0u && !sa->segs) return 0;
    for (size_t i = 0; i < (size_t)sa->count; i++) {
        if (!spectral_segment_valid_for_synth(&sa->segs[i])) return 0;
    }
    return 1;
}

static inline int spectral_segment_gpu_payload_valid(const SegmentGpu* s)
{
    if (!s) return 0;
    if (!spectral_is_finite_f32(s->start) || s->start < 0.0f) return 0;
    if (!spectral_is_finite_f32(s->length) || s->length < 0.0f) return 0;
    if (!spectral_is_finite_f32(s->phase)) return 0;
    if (!spectral_is_finite_f32(s->omega) || s->omega < 0.0f) return 0;
    if (!spectral_is_finite_f32(s->df)) return 0;
    if (!spectral_is_finite_f32(s->amp)) return 0;
    if (!spectral_is_finite_f32(s->da)) return 0;
    return 1;
}

static inline int spectral_segment_gpu_matches_segment(const Segment* seg,
                                                       const SegmentGpu* gpu)
{
    if (!spectral_segment_payload_valid(seg) ||
        !spectral_segment_gpu_payload_valid(gpu)) {
        return 0;
    }

    return gpu->start == seg->start &&
           gpu->length == seg->length &&
           gpu->phase == seg->phase &&
           gpu->omega == seg->omega &&
           gpu->df == seg->df &&
           gpu->amp == seg->amp &&
           gpu->da == seg->da;
}

static inline int spectral_segment_gpu_array_matches_segments(const Segment* segs,
                                                              const SegmentGpu* gpu_segs,
                                                              size_t count,
                                                              uint32_t* bad_idx)
{
    if (count > 0u && (!segs || !gpu_segs)) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!spectral_segment_gpu_matches_segment(&segs[i], &gpu_segs[i])) {
            if (bad_idx) *bad_idx = (i > (size_t)UINT32_MAX) ? UINT32_MAX : (uint32_t)i;
            return 0;
        }
    }
    return 1;
}

static inline int spectral_gpu_tile_layout_words_valid(const void* tile_ranges,
                                                       const uint32_t* tile_segment_ids,
                                                       uint32_t tile_count,
                                                       uint32_t tile_total_refs,
                                                       uint32_t segment_count)
{
    const char* range_src = (const char*)tile_ranges;
    uint32_t running_refs = 0;

    if (tile_count == 0u || tile_total_refs == 0u) {
        return tile_count == 0u && tile_total_refs == 0u;
    }
    if (!range_src || !tile_segment_ids || segment_count == 0u) return 0;

    for (uint32_t i = 0; i < tile_count; i++) {
        uint32_t words[2] = {0, 0};
        uint32_t start = 0;
        uint32_t count = 0;

        memcpy(words, range_src + ((size_t)i * sizeof(words)), sizeof(words));
        start = words[0];
        count = words[1];

        if (start != running_refs) return 0;
        if (count > UINT32_MAX - running_refs) return 0;
        running_refs += count;
        if (running_refs > tile_total_refs) return 0;
    }
    if (running_refs != tile_total_refs) return 0;

    for (uint32_t i = 0; i < tile_total_refs; i++) {
        if (tile_segment_ids[i] >= segment_count) return 0;
    }
    return 1;
}

static inline int spectral_u64_add_checked(uint64_t a, uint64_t b, uint64_t* out)
{
    if (!out || b > UINT64_MAX - a) return 0;
    *out = a + b;
    return 1;
}

static inline int spectral_double_accumulate_nonnegative_checked(double current,
                                                                 double delta,
                                                                 double* out)
{
    double next = 0.0;

    if (!out || !spectral_is_finite_f32(current) || current < 0.0 ||
        !spectral_is_finite_f32(delta) || delta < 0.0) {
        return 0;
    }

    next = current + delta;
    if (!spectral_is_finite_f32(next) || next < current) return 0;
    *out = next;
    return 1;
}

/* COLA / WOLA overlap-add reconstruction envelope.
 *
 * For an STFT with analysis window a[] and synthesis window s[] (each `length`
 * samples) advanced by `hop` samples, the per-phase reconstruction envelope is
 *   env[n] = sum_k a[n + k*hop] * s[n + k*hop],   n in [0, hop)
 * Overlap-add resynthesis is gain-flat exactly when env[n] is the same constant
 * for every phase position n: the Constant-OverLap-Add (COLA) condition for a
 * single window, or its Weighted form (WOLA) when an analysis AND a synthesis
 * window are both present. See Allen (1977), Allen & Rabiner (1977), and Griffin
 * & Lim (1984) "Signal Estimation from Modified Short-Time Fourier Transform".
 *
 * `hop` must satisfy 0 < hop <= length and divide length (integer overlap
 * factor). Pass synthesis == NULL to test plain COLA on the analysis window
 * (synthesis identically 1). Returns 1 and writes the envelope min/max/mean (any
 * out pointer may be NULL); returns 0 on bad arguments or a non-finite sample. */
static inline int spectral_overlap_add_envelope_stats(const float* analysis,
                                                       const float* synthesis,
                                                       size_t length, size_t hop,
                                                       double* out_min,
                                                       double* out_max,
                                                       double* out_mean)
{
    double env_min = 0.0;
    double env_max = 0.0;
    double env_sum = 0.0;

    if (!analysis || length == 0u || hop == 0u || hop > length) return 0;
    if (length % hop != 0u) return 0;

    for (size_t n = 0; n < hop; n++) {
        double e = 0.0;
        for (size_t idx = n; idx < length; idx += hop) {
            double a = (double)analysis[idx];
            double s = synthesis ? (double)synthesis[idx] : 1.0;
            if (!spectral_is_finite_f32(a) || !spectral_is_finite_f32(s)) return 0;
            e += a * s;
        }
        if (!spectral_is_finite_f32(e)) return 0;
        if (n == 0u) {
            env_min = e;
            env_max = e;
        } else {
            if (e < env_min) env_min = e;
            if (e > env_max) env_max = e;
        }
        env_sum += e;
    }

    if (out_min) *out_min = env_min;
    if (out_max) *out_max = env_max;
    if (out_mean) *out_mean = env_sum / (double)hop;
    return 1;
}

/* COLA/WOLA invariant predicate: the overlap-add envelope (see
 * spectral_overlap_add_envelope_stats) is constant within `rel_tol` of its mean,
 * with a strictly positive mean. Returns 1 when the window/hop combination
 * reconstructs gain-flat to tolerance, else 0. On success *out_gain (if non-NULL)
 * receives the constant overlap gain (the envelope mean). A negative or non-finite
 * rel_tol is treated as 0 (exact). */
static inline int spectral_overlap_add_is_constant(const float* analysis,
                                                    const float* synthesis,
                                                    size_t length, size_t hop,
                                                    double rel_tol, double* out_gain)
{
    double env_min = 0.0;
    double env_max = 0.0;
    double env_mean = 0.0;

    if (!spectral_overlap_add_envelope_stats(analysis, synthesis, length, hop,
                                             &env_min, &env_max, &env_mean)) {
        return 0;
    }
    if (!spectral_is_finite_f32(env_mean) || env_mean <= 0.0) return 0;
    if (!spectral_is_finite_f32(rel_tol) || rel_tol < 0.0) rel_tol = 0.0;
    if ((env_max - env_min) > rel_tol * env_mean) return 0;
    if (out_gain) *out_gain = env_mean;
    return 1;
}

#endif /* SPECTRAL_CONTRACTS_H */
