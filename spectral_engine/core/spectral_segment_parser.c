/* spectral_segment_parser.c - Segment File I/O Implementation
 *
 * Binary segment file operations for saving and loading analyzed audio.
 * See spectral_segment_parser.h for file format documentation.
 *
 * Endianness: Files are always stored in little-endian format.
 * On big-endian hosts, byte-swapping is performed during load/save.
 *
 * Desktop/emulator only — bare-metal embedded loads segments from flash/SDRAM.
 */
#include "spectral_segment_parser.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMULATOR

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/*
 * Endianness detection and byte-swap utilities
 */

static inline int is_big_endian(void) {
    union { uint32_t i; uint8_t c[4]; } u = { .i = 0x01020304 };
    return u.c[0] == 0x01;
}

static inline uint32_t swap_u32(uint32_t x) {
    return ((x >> 24) & 0x000000FF) |
           ((x >>  8) & 0x0000FF00) |
           ((x <<  8) & 0x00FF0000) |
           ((x << 24) & 0xFF000000);
}

static inline float swap_float(float f) {
    union { float f; uint32_t u; } u;
    u.f = f;
    u.u = swap_u32(u.u);
    return u.f;
}

/* Swap header fields to/from little-endian */
static void header_to_le(SegmentFileHeader* hdr) {
    if (!is_big_endian()) return;
    hdr->version = swap_u32(hdr->version);
    hdr->sr = swap_u32(hdr->sr);
    hdr->stretch = swap_float(hdr->stretch);
    hdr->pitch = swap_float(hdr->pitch);
    hdr->count = swap_u32(hdr->count);
    hdr->reserved = swap_u32(hdr->reserved);
}

static void header_from_le(SegmentFileHeader* hdr) {
    header_to_le(hdr);  /* Symmetric operation */
}

/* Swap segment fields to/from little-endian */
static void segment_to_le(Segment* seg) {
    if (!is_big_endian()) return;
    seg->start = swap_float(seg->start);
    seg->length = swap_float(seg->length);
    seg->phase = swap_float(seg->phase);
    seg->omega = swap_float(seg->omega);
    seg->df = swap_float(seg->df);
    seg->amp = swap_float(seg->amp);
    seg->da = swap_float(seg->da);
    seg->width = swap_float(seg->width);
}

static void segment_from_le(Segment* seg) {
    segment_to_le(seg);  /* Symmetric operation */
}

/* Returns 1 if segment values are finite and in valid range, 0 if corrupt */
static int segment_validate(const Segment* seg) {
    if (!isfinite(seg->omega) || seg->omega < 0.0f) return 0;
    if (!isfinite(seg->amp)) return 0;
    if (!isfinite(seg->start) || seg->start < 0.0f) return 0;
    if (!isfinite(seg->length) || seg->length < 0.0f) return 0;
    if (!isfinite(seg->df)) return 0;
    if (!isfinite(seg->da)) return 0;
    if (!isfinite(seg->phase)) return 0;
    if (!isfinite(seg->width)) return 0;
    return 1;
}

/* Validate entire segment array, report first corrupt segment index */
static int segments_validate_all(const Segment* segs, uint32_t count, uint32_t* bad_idx) {
    for (uint32_t i = 0; i < count; i++) {
        if (!segment_validate(&segs[i])) {
            if (bad_idx) *bad_idx = i;
            return 0;
        }
    }
    return 1;
}

SpectralError segments_save(const char* path, const SegmentArray* sa, int sr, float stretch, float pitch) {
    if (!path || !sa) return SPECTRAL_ERR_PARAM;
    
    FILE* f = fopen(path, "wb");
    if (!f) return SPECTRAL_ERR_FILE_OPEN;
    
    SegmentFileHeader hdr = {
        .magic = {SEGMENT_FILE_MAGIC[0], SEGMENT_FILE_MAGIC[1], 
                  SEGMENT_FILE_MAGIC[2], SEGMENT_FILE_MAGIC[3]},
        .version = SEGMENT_FILE_VERSION,
        .sr = (uint32_t)sr,
        .stretch = stretch,
        .pitch = pitch,
        .count = sa->count,
        .reserved = 0
    };
    
    header_to_le(&hdr);
    
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return SPECTRAL_ERR_FILE_WRITE; }
    
    /* Write segments, converting each to little-endian if needed */
    if (is_big_endian()) {
        for (uint32_t i = 0; i < sa->count; i++) {
            Segment seg = sa->segs[i];
            segment_to_le(&seg);
            if (fwrite(&seg, sizeof(Segment), 1, f) != 1) { fclose(f); return SPECTRAL_ERR_FILE_WRITE; }
        }
    } else {
        if (fwrite(sa->segs, sizeof(Segment), sa->count, f) != sa->count) { fclose(f); return SPECTRAL_ERR_FILE_WRITE; }
    }
    
    fclose(f);
    return SPECTRAL_OK;
}

SpectralError segments_load(const char* path, SegmentArray* sa, int* out_sr, float* out_stretch, float* out_pitch) {
    if (!path || !sa || !out_sr || !out_stretch || !out_pitch) return SPECTRAL_ERR_PARAM;
    
    FILE* f = fopen(path, "rb");
    if (!f) return SPECTRAL_ERR_FILE_OPEN;
    
    SegmentFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return SPECTRAL_ERR_FILE_READ; }
    
    /* Magic is not affected by endianness (char array) */
    if (memcmp(hdr.magic, SEGMENT_FILE_MAGIC, 4) != 0) { 
        fprintf(stderr, "Error: Invalid segment file format (bad magic)\n");
        fclose(f); 
        return SPECTRAL_ERR_FILE_FORMAT; 
    }
    
    /* Convert header from little-endian file format */
    header_from_le(&hdr);
    
    /* Version compatibility check */
    if (hdr.version != SEGMENT_FILE_VERSION) {
        if (hdr.version > SEGMENT_FILE_VERSION) {
                 fprintf(stderr, "Error: Segment file version %u is newer than supported version %u\n"
                          "       Please update spectral tools.\n",
                    hdr.version, SEGMENT_FILE_VERSION);
        } else {
            fprintf(stderr, "Error: Segment file version %u is older than current version %u\n"
                           "       Re-export segments with current desktop build.\n",
                    hdr.version, SEGMENT_FILE_VERSION);
        }
        fclose(f); 
        return SPECTRAL_ERR_FILE_VERSION;
    }
    
    sa->count = hdr.count;
    sa->capacity = hdr.count;
    sa->segs = NULL;

    if (hdr.count == 0) {
        *out_sr = (int)hdr.sr;
        *out_stretch = hdr.stretch;
        *out_pitch = hdr.pitch;
        fclose(f);
        return SPECTRAL_OK;
    }

/* Overflow check - tautologically false on 64-bit but needed for 32-bit portability */
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-constant-out-of-range-compare"
#endif
    if (hdr.count > SIZE_MAX / sizeof(Segment)) { fclose(f); return SPECTRAL_ERR_MEMORY; }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
    sa->segs = (Segment*)malloc(hdr.count * sizeof(Segment));
    if (!sa->segs) { fclose(f); return SPECTRAL_ERR_MEMORY; }
    
    if (fread(sa->segs, sizeof(Segment), hdr.count, f) != hdr.count) {
        free(sa->segs);
        sa->segs = NULL;
        fclose(f);
        return SPECTRAL_ERR_FILE_READ;
    }
    
    /* Convert segments from little-endian if needed */
    if (is_big_endian()) {
        for (uint32_t i = 0; i < hdr.count; i++) {
            segment_from_le(&sa->segs[i]);
        }
    }
    
    /* Validate segment data integrity */
    uint32_t bad_idx = 0;
    if (!segments_validate_all(sa->segs, hdr.count, &bad_idx)) {
        fprintf(stderr, "Error: Corrupt segment data at index %u (NaN/inf detected)\n"
                       "       Re-export segments or check source audio.\n", bad_idx);
        free(sa->segs);
        sa->segs = NULL;
        fclose(f);
        return SPECTRAL_ERR_FILE_CORRUPT;
    }
    
    *out_sr = (int)hdr.sr;
    *out_stretch = hdr.stretch;
    *out_pitch = hdr.pitch;
    fclose(f);
    return SPECTRAL_OK;
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMULATOR */
