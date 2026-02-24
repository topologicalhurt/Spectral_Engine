/* spectral_segment_parser.c - Segment File I/O Implementation
 *
 * Binary segment file operations for saving and loading analyzed audio.
 * See spectral_segment_parser.h for file format documentation.
 *
 * Endianness: Files are always stored in little-endian format.
 * On big-endian hosts, byte-swapping is performed during load/save.
 *
 * Desktop/simulation only — bare-metal embedded loads segments from flash/SDRAM.
 */
#include "spectral_segment_parser.h"
#include "spectral_fs.h"
#include "spectral_endian.h"
#include "spectral_utils.h"
#include "spectral_log.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* Swap header fields to/from little-endian (symmetric). */
static void header_to_le(SegmentFileHeader* hdr) {
    if (!spectral_is_big_endian()) return;
    hdr->version  = spectral_swap_u32(hdr->version);
    hdr->sr       = spectral_swap_u32(hdr->sr);
    hdr->stretch  = spectral_swap_float(hdr->stretch);
    hdr->pitch    = spectral_swap_float(hdr->pitch);
    hdr->count    = spectral_swap_u32(hdr->count);
    hdr->reserved = spectral_swap_u32(hdr->reserved);
}

static void header_from_le(SegmentFileHeader* hdr) {
    header_to_le(hdr);
}

/* Returns 1 if segment values are finite and in valid range, 0 if corrupt */
static int segment_validate(const Segment* seg) {
    if (!spectral_is_finite_f32(seg->omega) || seg->omega < 0.0f) return 0;
    if (!spectral_is_finite_f32(seg->amp)) return 0;
    if (!spectral_is_finite_f32(seg->start) || seg->start < 0.0f) return 0;
    if (!spectral_is_finite_f32(seg->length) || seg->length < 0.0f) return 0;
    if (!spectral_is_finite_f32(seg->df)) return 0;
    if (!spectral_is_finite_f32(seg->da)) return 0;
    if (!spectral_is_finite_f32(seg->phase)) return 0;
    if (!spectral_is_finite_f32(seg->width)) return 0;
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
    SpectralError err = SPECTRAL_OK;
    FILE* f = NULL;
    int needs_swap = spectral_is_big_endian();
    size_t seg_bytes = 0;

    if (spectral_is_empty_string(path) || !sa) return SPECTRAL_ERR_PARAM;
    if (sa->count > 0 && !sa->segs) return SPECTRAL_ERR_PARAM;

    err = spectral_fs_open(&f, path, "wb");
    if (err != SPECTRAL_OK) return err;

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

    err = spectral_fs_write_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_WRITE);
    if (err != SPECTRAL_OK) goto cleanup;

    /* Write segments, converting each to little-endian if needed */
    if (needs_swap) {
        for (uint32_t i = 0; i < sa->count; i++) {
            Segment seg = sa->segs[i];
            spectral_segment_swap_endian(&seg);
            err = spectral_fs_write_exact(f, &seg, sizeof(Segment), SPECTRAL_ERR_FILE_WRITE);
            if (err != SPECTRAL_OK) goto cleanup;
        }
    } else {
        if (!spectral_array_bytes((size_t)sa->count, sizeof(Segment), &seg_bytes)) {
            err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }
        err = spectral_fs_write_exact(f, sa->segs, seg_bytes, SPECTRAL_ERR_FILE_WRITE);
        if (err != SPECTRAL_OK) goto cleanup;
    }

cleanup:
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_WRITE);
        if (err == SPECTRAL_OK && close_err != SPECTRAL_OK) err = close_err;
    }
    return err;
}

SpectralError segments_load(const char* path, SegmentArray* sa, int* out_sr, float* out_stretch, float* out_pitch) {
    SpectralError err = SPECTRAL_OK;
    FILE* f = NULL;
    SegmentFileHeader hdr;
    Segment* loaded_segs = NULL;

    if (spectral_is_empty_string(path) || !sa || !out_sr || !out_stretch || !out_pitch) return SPECTRAL_ERR_PARAM;

    err = spectral_fs_open(&f, path, "rb");
    if (err != SPECTRAL_OK) return err;

    err = spectral_fs_read_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) goto cleanup;

    /* Magic is not affected by endianness (char array) */
    if (memcmp(hdr.magic, SEGMENT_FILE_MAGIC, 4) != 0) {
        SPECTRAL_LOG_ERROR_STDERR("Error: Invalid segment file format (bad magic)");
        err = SPECTRAL_ERR_FILE_FORMAT;
        goto cleanup;
    }

    /* Convert header from little-endian file format */
    header_from_le(&hdr);

    /* Version compatibility check */
    if (hdr.version != SEGMENT_FILE_VERSION) {
        if (hdr.version > SEGMENT_FILE_VERSION) {
            SPECTRAL_LOG_ERROR_STDERR(
                "Error: Segment file version %u is newer than supported version %u\n"
                "       Please update spectral tools.",
                hdr.version, SEGMENT_FILE_VERSION);
        } else {
            SPECTRAL_LOG_ERROR_STDERR(
                "Error: Segment file version %u is older than current version %u\n"
                "       Re-export segments with current desktop build.",
                hdr.version, SEGMENT_FILE_VERSION);
        }
        err = SPECTRAL_ERR_FILE_VERSION;
        goto cleanup;
    }

    if (hdr.count == 0) {
        sa->count = 0;
        sa->capacity = 0;
        sa->segs = NULL;
        *out_sr = (int)hdr.sr;
        *out_stretch = hdr.stretch;
        *out_pitch = hdr.pitch;
        err = SPECTRAL_OK;
        goto cleanup;
    }

    {
        size_t alloc_bytes = 0;
        if (!spectral_size_mul((size_t)hdr.count, sizeof(Segment), &alloc_bytes)) {
            err = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }
        loaded_segs = (Segment*)malloc(alloc_bytes);
    }
    if (!loaded_segs) {
        err = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }

    {
        size_t seg_bytes = 0;
        if (!spectral_array_bytes((size_t)hdr.count, sizeof(Segment), &seg_bytes)) {
            err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }
        err = spectral_fs_read_exact(f, loaded_segs, seg_bytes, SPECTRAL_ERR_FILE_READ);
        if (err != SPECTRAL_OK) goto cleanup;
    }

    /* Convert segments from little-endian if needed */
    if (spectral_is_big_endian()) {
        for (uint32_t i = 0; i < hdr.count; i++) {
            spectral_segment_swap_endian(&loaded_segs[i]);
        }
    }
    
    /* Validate segment data integrity */
    uint32_t bad_idx = 0;
    if (!segments_validate_all(loaded_segs, hdr.count, &bad_idx)) {
        SPECTRAL_LOG_ERROR_STDERR(
            "Error: Corrupt segment data at index %u (NaN/inf detected)\n"
            "       Re-export segments or check source audio.",
            bad_idx);
        err = SPECTRAL_ERR_FILE_CORRUPT;
        goto cleanup;
    }

    sa->count = hdr.count;
    sa->capacity = hdr.count;
    sa->segs = loaded_segs;
    loaded_segs = NULL;

    *out_sr = (int)hdr.sr;
    *out_stretch = hdr.stretch;
    *out_pitch = hdr.pitch;
    err = SPECTRAL_OK;

cleanup:
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_READ);
        if (err == SPECTRAL_OK && close_err != SPECTRAL_OK) err = close_err;
    }
    free(loaded_segs);
    return err;
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */
