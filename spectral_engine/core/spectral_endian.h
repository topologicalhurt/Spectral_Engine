/* spectral_endian.h — Shared little-endian byte-swap primitives.
 *
 * All binary file formats in this project are stored little-endian.
 * On big-endian hosts these helpers perform byte-swapping; on LE they no-op.
 */
#ifndef SPECTRAL_ENDIAN_H
#define SPECTRAL_ENDIAN_H

#include "spectral_common.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline int spectral_is_big_endian(void)
{
    union { uint32_t i; uint8_t c[4]; } u = { .i = 0x01020304 };
    return u.c[0] == 0x01;
}

static inline uint32_t spectral_swap_u32(uint32_t x)
{
    return ((x >> 24) & 0x000000FFu) |
           ((x >>  8) & 0x0000FF00u) |
           ((x <<  8) & 0x00FF0000u) |
           ((x << 24) & 0xFF000000u);
}

static inline uint64_t spectral_swap_u64(uint64_t x)
{
    return ((x >> 56) & 0x00000000000000FFull) |
           ((x >> 40) & 0x000000000000FF00ull) |
           ((x >> 24) & 0x0000000000FF0000ull) |
           ((x >>  8) & 0x00000000FF000000ull) |
           ((x <<  8) & 0x000000FF00000000ull) |
           ((x << 24) & 0x0000FF0000000000ull) |
           ((x << 40) & 0x00FF000000000000ull) |
           ((x << 56) & 0xFF00000000000000ull);
}

static inline float spectral_swap_float(float f)
{
    union { float f; uint32_t u; } u;
    u.f = f;
    u.u = spectral_swap_u32(u.u);
    return u.f;
}

/* Unconditional byte-swap of ALL float words of the serialized 64-byte Segment.
 * The full payload must be covered: _pad_w[0..2] carry the cubic-MQ phase
 * annotation (c2/c3/flag, spectral_common.h) which is serialized to the segment
 * .bin / seg cache when SPECTRAL_PRECISE_PHASE is enabled — skipping them
 * corrupts cubic phase on a cross-endian exchange. The remaining _pad_w words
 * are reserved/zero and swapping zero is a no-op, so covering the whole union
 * is always correct. Exposed (not endian-gated) so the round-trip test can
 * exercise it on a little-endian host; production callers use the gated wrapper
 * below. The operation is its own inverse. */
static inline void spectral_segment_swap_words(Segment* seg)
{
    seg->start  = spectral_swap_float(seg->start);
    seg->length = spectral_swap_float(seg->length);
    seg->phase  = spectral_swap_float(seg->phase);
    seg->omega  = spectral_swap_float(seg->omega);
    seg->df     = spectral_swap_float(seg->df);
    seg->amp    = spectral_swap_float(seg->amp);
    seg->da     = spectral_swap_float(seg->da);
    seg->width  = spectral_swap_float(seg->width);
    for (int i = 0; i < 8; i++) {
        seg->_pad_w[i] = spectral_swap_float(seg->_pad_w[i]);
    }
}

/* Swap a Segment between native and LE — a no-op on LE hosts. */
static inline void spectral_segment_swap_endian(Segment* seg)
{
    if (!spectral_is_big_endian()) return;
    spectral_segment_swap_words(seg);
}

static inline void spectral_segment_gpu_swap_endian(SegmentGpu* seg)
{
    if (!spectral_is_big_endian()) return;
    seg->start  = spectral_swap_float(seg->start);
    seg->length = spectral_swap_float(seg->length);
    seg->phase  = spectral_swap_float(seg->phase);
    seg->omega  = spectral_swap_float(seg->omega);
    seg->df     = spectral_swap_float(seg->df);
    seg->amp    = spectral_swap_float(seg->amp);
    seg->da     = spectral_swap_float(seg->da);
    seg->_pad   = spectral_swap_float(seg->_pad);
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_ENDIAN_H */
