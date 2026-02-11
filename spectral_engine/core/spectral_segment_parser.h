/* spectral_segment_parser.h - Segment file I/O
 *
 * File format: SegmentFileHeader followed by Segment array in little-endian.
 * Version 2 adds width field and padding changes.
 */
#ifndef SPECTRAL_SEGMENT_PARSER_H
#define SPECTRAL_SEGMENT_PARSER_H

#include "spectral_common.h"

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

#define SEGMENT_FILE_MAGIC "SPEC"
#define SEGMENT_FILE_VERSION 2

/*
 * .spq File Format (Q15 embedded segments)
 * 
 * Binary format for pre-converted Q15 segments used on embedded targets.
 * Created by tools/convert_segments from desktop .bin files.
 */
#define SPQ_FILE_MAGIC  0x31515053
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

/* File I/O functions — desktop/emulator only (uses stdio, malloc) */
#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMULATOR

/* Save segments to binary file.
 * Returns SPECTRAL_OK on success, SPECTRAL_ERR_* on failure. */
SpectralError segments_save(const char* path, const SegmentArray* sa, int sr, float stretch, float pitch);

/* Load segments from binary file.
 * Allocates memory for sa->segs; caller must free() when done.
 * Returns SPECTRAL_OK on success, SPECTRAL_ERR_* on failure. */
SpectralError segments_load(const char* path, SegmentArray* sa, int* out_sr, float* out_stretch, float* out_pitch);

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMULATOR */

#endif /* SPECTRAL_SEGMENT_PARSER_H */
