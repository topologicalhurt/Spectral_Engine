/* spectral_segment_parser.h - Segment file I/O
 *
 * File format: SegmentFileHeader followed by Segment array in little-endian.
 * Version 2 adds width field and padding changes.
 */
#ifndef SPECTRAL_SEGMENT_PARSER_H
#define SPECTRAL_SEGMENT_PARSER_H

#include "spectral_common.h"
#include "spectral_error.h"
#include "spectral_endian.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t sr;
    float stretch;
    float pitch;
    uint32_t count;
    uint32_t reserved;
} SegmentFileHeader;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SegmentFileHeader) == 28, "SegmentFileHeader must be 28 bytes");
#endif

/* Swap a SegmentFileHeader between native and little-endian on-disk order.
 * The .bin (SPEC) format is always stored little-endian; the swap is symmetric
 * (its own inverse) and a no-op on little-endian hosts.  Single source of truth
 * shared by segments_save/segments_load and the convert_segments tool (which
 * does not link spectral_segment_parser.c, so the helper must be header-only).
 * The char[4] magic is endian-independent and intentionally left untouched. */
static inline void spectral_segment_file_header_swap_le(SegmentFileHeader* hdr)
{
    if (!hdr || !spectral_is_big_endian()) return;
    hdr->version  = spectral_swap_u32(hdr->version);
    hdr->sr       = spectral_swap_u32(hdr->sr);
    hdr->stretch  = spectral_swap_float(hdr->stretch);
    hdr->pitch    = spectral_swap_float(hdr->pitch);
    hdr->count    = spectral_swap_u32(hdr->count);
    hdr->reserved = spectral_swap_u32(hdr->reserved);
}

#define SEGMENT_FILE_MAGIC "SPEC"
#define SEGMENT_FILE_VERSION 2

/*
 * .spq File Format (Q15 embedded segments)
 * 
 * Binary format for pre-converted Q15 segments used on embedded targets.
 * Created by tools/convert_segments from desktop .bin files.
 */
#define SPQ_FILE_MAGIC  0x31515053  /* little-endian ASCII "SPQ1" */
#define SPQ_FILE_VERSION 1

typedef struct __attribute__((packed, aligned(4))) {
    uint32_t magic;
    uint32_t version;
    uint32_t sample_rate;
    uint32_t num_segments;
    uint32_t output_length;
    uint32_t flags;
    uint32_t reserved[2];
} SpqFileHeader;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(SpqFileHeader) == 32, "SpqFileHeader must be 32 bytes");
#endif

/* File I/O functions — desktop/simulation only (uses stdio, malloc) */
/* Save segments to binary file.
 * Returns SPECTRAL_OK on success, SPECTRAL_ERR_* on failure. */
SpectralError segments_save(const char* path, const SegmentArray* sa, int sr, float stretch, float pitch);

/* Load segments from binary file.
 * Allocates memory for sa->segs; caller must free() when done.
 * Returns SPECTRAL_OK on success, SPECTRAL_ERR_* on failure. */
SpectralError segments_load(const char* path, SegmentArray* sa, int* out_sr, float* out_stretch, float* out_pitch);
#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SEGMENT_PARSER_H */
