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
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

static inline int spectral_f32_span_finite_nonnegative(const float* values, size_t count)
{
    if (count > 0u && !values) return 0;
    for (size_t i = 0; i < count; i++) {
        if (!isfinite(values[i]) || values[i] < 0.0f) return 0;
    }
    return 1;
}

static inline int spectral_segment_payload_valid(const Segment* s)
{
    if (!s) return 0;
    if (!isfinite(s->start) || s->start < 0.0f) return 0;
    if (!isfinite(s->length) || s->length < 0.0f) return 0;
    if (!isfinite(s->phase)) return 0;
    if (!isfinite(s->omega) || s->omega < 0.0f) return 0;
    if (!isfinite(s->df)) return 0;
    if (!isfinite(s->amp)) return 0;
    if (!isfinite(s->da)) return 0;
    if (!isfinite(s->width)) return 0;
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
    if (!isfinite(s->start) || s->start < 0.0f) return 0;
    if (!isfinite(s->length) || s->length < 0.0f) return 0;
    if (!isfinite(s->phase)) return 0;
    if (!isfinite(s->omega) || s->omega < 0.0f) return 0;
    if (!isfinite(s->df)) return 0;
    if (!isfinite(s->amp)) return 0;
    if (!isfinite(s->da)) return 0;
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

    if (!out || !isfinite(current) || current < 0.0 ||
        !isfinite(delta) || delta < 0.0) {
        return 0;
    }

    next = current + delta;
    if (!isfinite(next) || next < current) return 0;
    *out = next;
    return 1;
}

#endif /* SPECTRAL_CONTRACTS_H */
