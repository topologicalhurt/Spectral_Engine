/* spectral_segment_parser.h - Segment File I/O
 * 
 * Binary segment file format for storing/loading analyzed audio data.
 * Used to transfer spectral analysis results between desktop (analysis)
 * and embedded (synthesis) builds.
 * 
 * File Format:
 *   - Header: SegmentFileHeader (magic, version, metadata)
 *   - Body: Array of Segment structs (count specified in header)
 *   - Endianness: Files stored in little-endian, auto-converted on load
 * 
 * Version History:
 *   - Version 1: Initial format
 *   - Version 2: Added width field, changed padding
 * 
 * Functions:
 *   segments_save() - Write analyzed segments to binary file
 *   segments_load() - Read segments from binary file (allocates memory)
 * 
 * Error Codes:
 *   0: Success
 *  -1: File I/O error (open, read, or write failed)
 *  -2: Invalid file format (bad magic bytes)
 *  -3: Memory allocation failed
 *  -4: Version mismatch (file too old or too new)
 *  -5: Corrupt data (NaN/inf in critical fields)
 */
#ifndef SPECTRAL_SEGMENT_PARSER_H
#define SPECTRAL_SEGMENT_PARSER_H

#include "spectral_common.h"

typedef enum {
    SEGMENT_PARSER_OK          =  0,
    SEGMENT_PARSER_ERR_IO      = -1,
    SEGMENT_PARSER_ERR_FORMAT  = -2,
    SEGMENT_PARSER_ERR_MEMORY  = -3,
    SEGMENT_PARSER_ERR_VERSION = -4,
    SEGMENT_PARSER_ERR_CORRUPT = -5  /* Invalid float values (NaN/inf) */
} SegmentParserResult;

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

/* Save segments to binary file
 * 
 * Parameters:
 *   path    - Output file path
 *   sa      - Segment array to save
 *   sr      - Sample rate in Hz
 *   stretch - Time stretch factor used during analysis
 *   pitch   - Pitch shift in semitones used during analysis
 * 
 * Returns: 0 on success, negative error code on failure
 */
int segments_save(const char* path, const SegmentArray* sa, int sr, float stretch, float pitch);

/* Load segments from binary file
 * 
 * Allocates memory for sa->segs; caller must free() when done.
 * 
 * Parameters:
 *   path       - Input file path
 *   sa         - Output segment array (segs pointer will be allocated)
 *   out_sr     - Output: sample rate from file
 *   out_stretch - Output: time stretch factor from file
 *   out_pitch  - Output: pitch shift from file
 * 
 * Returns: 0 on success, negative error code on failure
 */
int segments_load(const char* path, SegmentArray* sa, int* out_sr, float* out_stretch, float* out_pitch);

#endif /* SPECTRAL_SEGMENT_PARSER_H */
