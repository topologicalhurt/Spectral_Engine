/* spectral_seg_cache.c — Centralized xxhash-indexed segment cache.
 *
 * See spectral_seg_cache.h for file format overview.
 *
 * The index is kept small (thousands of entries max) so we reload it entirely
 * for each operation. Entries are sorted by key for O(log n) lookup.
 * On store, segment data is appended first (orphan bytes are harmless on
 * crash), then the index is rewritten with the new entry in sorted position.
 */
#include "spectral_seg_cache.h"
#include "spectral_seg_cache_fs.h"
#include "spectral_endian.h"
#include "spectral_hash_xx32_xx3.h"
#include "spectral_utils.h"
#include "spectral_log.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* --- Key computation ----------------------------------------------------- */

static int seg_cache_scale_to_i32(float value, double scale, int* out)
{
    double scaled = 0.0;

    if (!out || !spectral_is_finite_f32(value) || !isfinite(scale)) return 0;

    scaled = (double)value * scale;
    if (!isfinite(scaled) || scaled < (double)INT_MIN || scaled > (double)INT_MAX) {
        return 0;
    }

    *out = (int)scaled;
    return 1;
}

uint64_t spectral_seg_cache_key(const char* input_id,
                                int n_fft, int hop,
                                float db_thresh,
                                float start_sec, float end_sec,
                                float stretch,
                                uint32_t tile_size)
{
    /* Hash text form of input identity + analysis/render params for deterministic
     * keys.  The caller-provided input_id must already identify the audio input
     * content, not just its basename. */
    char buf[512];
    int len = 0;
    int db_thresh_i = 0;
    int start_ms_i = 0;
    int end_ms_i = 0;
    int stretch_ppm_i = 0;

    if (spectral_is_empty_string(input_id) ||
        n_fft <= 0 || hop <= 0 || tile_size == 0u ||
        !spectral_is_finite_positive_f32(stretch) ||
        stretch > SPECTRAL_MAX_STRETCH ||
        !seg_cache_scale_to_i32(db_thresh, 10.0, &db_thresh_i) ||
        !seg_cache_scale_to_i32(start_sec, 1000.0, &start_ms_i) ||
        !seg_cache_scale_to_i32(end_sec, 1000.0, &end_ms_i) ||
        !seg_cache_scale_to_i32(stretch, 1000000.0, &stretch_ppm_i)) {
        return 0;
    }

    len = snprintf(buf, sizeof(buf),
                   "%s|%d|%d|%d|%d|%d|%d|%u",
                   input_id,
                   n_fft, hop,
                   db_thresh_i,
                   start_ms_i,
                   end_ms_i,
                   stretch_ppm_i,
                   tile_size);
    if (len <= 0 || (size_t)len >= sizeof(buf)) {
        return 0;
    }
    return (uint64_t)spectral_hash_oneshot(buf, (size_t)len);
}



/* --- Binary search (Java binarySearch convention: ~insertion_point) ----- */

static int64_t seg_cache_bsearch(const SpectralSegCacheEntry* entries,
                                 uint32_t count, uint64_t key)
{
    int64_t lo = 0;
    int64_t hi = (int64_t)count - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (entries[mid].key == key) return mid;
        if (entries[mid].key < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return ~lo;
}

static int seg_cache_tile_blob_bytes(uint32_t tile_count,
                                     uint32_t tile_total_refs,
                                     size_t* out_bytes)
{
    size_t ranges_bytes = 0;
    size_t ids_bytes = 0;
    size_t sum = 0;

    if (!out_bytes) return 0;
    *out_bytes = 0;

    if (tile_count == 0 || tile_total_refs == 0) {
        return 1;
    }

    if (!spectral_array_bytes((size_t)tile_count, sizeof(uint32_t) * 2u, &ranges_bytes) ||
        !spectral_array_bytes((size_t)tile_total_refs, sizeof(uint32_t), &ids_bytes) ||
        !spectral_size_add(sizeof(SpectralSegCacheTileHeader), ranges_bytes, &sum) ||
        !spectral_size_add(sum, ids_bytes, out_bytes)) {
        return 0;
    }
    return 1;
}

static int seg_cache_entry_metadata_valid(const SpectralSegCacheEntry* e)
{
    if (!e) return 0;

    /* All fields below come from disk.  Validate before narrowing uint32_t to
     * int, uint64_t to size_t, or trusting tile metadata to describe a blob. */
    if (e->sample_rate < (uint32_t)SPECTRAL_MIN_SAMPLE_RATE ||
        e->sample_rate > (uint32_t)SPECTRAL_MAX_SAMPLE_RATE) {
        return 0;
    }
    if (!spectral_is_finite_positive_f32(e->stretch) ||
        e->stretch > SPECTRAL_MAX_STRETCH) {
        return 0;
    }
    if (!spectral_is_finite_f32(e->pitch) ||
        e->pitch < SPECTRAL_MIN_PITCH ||
        e->pitch > SPECTRAL_MAX_PITCH) {
        return 0;
    }
    if ((uint64_t)(size_t)e->output_length != e->output_length) {
        return 0;
    }
    if (e->seg_count == 0 && (e->tile_count != 0 || e->tile_total_refs != 0)) {
        return 0;
    }
    if ((e->tile_count == 0) != (e->tile_total_refs == 0)) {
        return 0;
    }
    return 1;
}

static SpectralError seg_cache_validate_data_extent(const char* cache_dir,
                                                    uint64_t data_offset,
                                                    size_t total_data_bytes)
{
    uint64_t data_file_size = 0;
    uint64_t total_data_bytes_u64 = 0;
    SpectralError err = SPECTRAL_OK;

    if (!cache_dir) return SPECTRAL_ERR_PARAM;
    if (total_data_bytes == 0u) return SPECTRAL_OK;

    total_data_bytes_u64 = (uint64_t)total_data_bytes;
    if ((size_t)total_data_bytes_u64 != total_data_bytes) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    err = spectral_seg_cache_fs_data_file_size(cache_dir, &data_file_size);
    if (err != SPECTRAL_OK) return err;

    if (total_data_bytes_u64 > UINT64_MAX - data_offset ||
        data_file_size < data_offset + total_data_bytes_u64) {
        return SPECTRAL_ERR_FILE_CORRUPT;
    }

    return SPECTRAL_OK;
}


static int seg_cache_validate_tile_blob(const SpectralSegCacheEntry* e,
                                        const char* tile_base,
                                        size_t tile_data_bytes,
                                        SpectralSegCacheLookupResult* result)
{
    SpectralSegCacheTileHeader th;
    size_t expected_bytes = 0;

    if (!e || !tile_base || !result) return 0;
    if (e->tile_count == 0 || e->tile_total_refs == 0) return 0;
    if (tile_data_bytes < sizeof(SpectralSegCacheTileHeader)) return 0;

    memcpy(&th, tile_base, sizeof(th));

    if (!seg_cache_tile_blob_bytes(th.num_tiles, th.total_refs, &expected_bytes)) {
        return 0;
    }
    if (expected_bytes != tile_data_bytes) {
        SPECTRAL_LOG_WARN("Segment cache tile blob size mismatch — skipping tile cache data");
        return 0;
    }

    if (th.tile_size != SPECTRAL_GPU_TILE_SIZE ||
        th.num_tiles != e->tile_count ||
        th.total_refs != e->tile_total_refs) {
        SPECTRAL_LOG_WARN("Segment cache tile metadata mismatch — skipping tile cache data");
        return 0;
    }

    result->tile_size = th.tile_size;
    result->tile_count = th.num_tiles;
    result->tile_total_refs = th.total_refs;
    result->tile_ranges = (void*)(tile_base + sizeof(SpectralSegCacheTileHeader));
    result->tile_segment_ids = (uint32_t*)(tile_base + sizeof(SpectralSegCacheTileHeader)
                                           + (size_t)th.num_tiles * (sizeof(uint32_t) * 2u));
    return 1;
}

/* --- Public API ---------------------------------------------------------- */

SpectralError spectral_seg_cache_lookup(const char* cache_dir,
                                        uint64_t key,
                                        SpectralSegCacheLookupResult* result)
{
    SpectralSegCacheEntry* entries = NULL;
    uint32_t count = 0;
    SpectralError err;
    int64_t idx;

    if (!cache_dir || !result) return SPECTRAL_ERR_PARAM;
    *result = (SpectralSegCacheLookupResult){0};

    err = spectral_seg_cache_fs_index_load(cache_dir, &entries, &count);
    if (err != SPECTRAL_OK) return err;

    if (count == 0) {
        free(entries);
        return SPECTRAL_OK;
    }

    idx = seg_cache_bsearch(entries, count, key);
    if (idx < 0) {
        free(entries);
        return SPECTRAL_OK;
    }

    /* Hit — load segment + optional tile data from the data file. */
    {
        const SpectralSegCacheEntry* e = &entries[idx];
        size_t seg_bytes = 0;
        size_t gpu_seg_bytes = 0;
        size_t tile_data_bytes = 0;
        size_t total_data_bytes = 0;

        if (!seg_cache_entry_metadata_valid(e)) {
            free(entries);
            return SPECTRAL_ERR_FILE_CORRUPT;
        }

        result->sample_rate = (int)e->sample_rate;
        result->stretch = e->stretch;
        result->pitch = e->pitch;
        result->seg_count = e->seg_count;
        result->output_length = (size_t)e->output_length;

        if (e->seg_count == 0) {
            result->hit = 1;
            free(entries);
            return SPECTRAL_OK;
        }

        if (!spectral_array_bytes((size_t)e->seg_count, sizeof(Segment), &seg_bytes) ||
            !spectral_array_bytes((size_t)e->seg_count, sizeof(SegmentGpu), &gpu_seg_bytes) ||
            !seg_cache_tile_blob_bytes(e->tile_count, e->tile_total_refs, &tile_data_bytes) ||
            !spectral_size_add(seg_bytes, gpu_seg_bytes, &total_data_bytes) ||
            !spectral_size_add(total_data_bytes, tile_data_bytes, &total_data_bytes)) {
            free(entries);
            return SPECTRAL_ERR_OVERFLOW;
        }

        err = seg_cache_validate_data_extent(cache_dir, e->data_offset, total_data_bytes);
        if (err != SPECTRAL_OK) {
            free(entries);
            return err;
        }

        /* Fast path: mmap the data region read-only (little-endian hosts). */
        if (!spectral_is_big_endian() && spectral_seg_cache_fs_has_mmap()) {
            SpectralSegCacheFsMapView mv = {0};
            SpectralError map_err = spectral_seg_cache_fs_data_map_ro(
                cache_dir, e->data_offset, total_data_bytes, &mv);
            if (map_err == SPECTRAL_OK) {
                char* data_ptr = (char*)mv.base + mv.page_offset;

                result->segments.segs = (Segment*)data_ptr;
                result->segments.count = e->seg_count;
                result->segments.capacity = 0; /* mmap-owned */

                /* Pre-packed GPU segments follow Segment data. */
                result->gpu_segs = (const SegmentGpu*)(data_ptr + seg_bytes);

                if (tile_data_bytes > 0) {
                    const char* tile_base = data_ptr + seg_bytes + gpu_seg_bytes;
                    (void)seg_cache_validate_tile_blob(e, tile_base, tile_data_bytes, result);
                }

                result->_mmap_base = mv.base;
                result->_mmap_len = mv.len;
                result->hit = 1;
                free(entries);
                return SPECTRAL_OK;
            }

            if (map_err == SPECTRAL_ERR_FILE_CORRUPT || map_err == SPECTRAL_ERR_FILE_READ) {
                free(entries);
                return map_err;
            }
            /* mmap unavailable/failed — fall through to heap path. */
        }

        /* Fallback: heap allocation + read (big-endian or mmap unavailable). */
        {
            Segment* segs = (Segment*)spectral_malloc_array((size_t)e->seg_count, sizeof(Segment));
            if (!segs) {
                free(entries);
                return SPECTRAL_ERR_MEMORY;
            }

            err = spectral_seg_cache_fs_data_read_exact(cache_dir, e->data_offset, segs, seg_bytes);
            if (err != SPECTRAL_OK) {
                free(segs);
                free(entries);
                return err;
            }

            if (spectral_is_big_endian()) {
                for (uint32_t i = 0; i < e->seg_count; i++) {
                    spectral_segment_swap_endian(&segs[i]);
                }
            }

            result->hit = 1;
            result->segments.segs = segs;
            result->segments.count = e->seg_count;
            result->segments.capacity = e->seg_count;
            result->tile_size = 0;
            result->tile_count = 0;
            result->tile_total_refs = 0;
            result->tile_ranges = NULL;
            result->tile_segment_ids = NULL;
            result->gpu_segs = NULL;
        }
    }

    free(entries);
    return SPECTRAL_OK;
}

SegmentArray spectral_seg_cache_result_take_segments(
    SpectralSegCacheLookupResult* r)
{
    SegmentArray taken = {0};
    if (!r) return taken;
    /* Note: for mmap-backed hits, taken.segs points into r->_mmap_base and
     * remains valid only until spectral_seg_cache_result_free(r). */
    taken = r->segments;
    r->segments = (SegmentArray){0};
    return taken;
}

void spectral_seg_cache_result_free(SpectralSegCacheLookupResult* r)
{
    if (!r) return;

    if (r->_mmap_base) {
        SpectralSegCacheFsMapView mv = {
            .base = r->_mmap_base,
            .len = r->_mmap_len,
            .page_offset = 0
        };
        spectral_seg_cache_fs_data_unmap(&mv);
        memset(r, 0, sizeof(*r));
        return;
    }

    if (r->segments.capacity > 0) free(r->segments.segs);
    free(r->tile_ranges);
    free(r->tile_segment_ids);
    memset(r, 0, sizeof(*r));
}

static int seg_cache_store_metadata_valid(uint64_t key,
                                          const SegmentArray* sa,
                                          int sample_rate,
                                          float stretch,
                                          float pitch,
                                          size_t output_length)
{
    uint64_t output_length_u64 = 0;

    if (key == 0u || !sa) return 0;
    if (sa->count > 0 && !sa->segs) return 0;
    if (sample_rate < SPECTRAL_MIN_SAMPLE_RATE || sample_rate > SPECTRAL_MAX_SAMPLE_RATE) return 0;
    if (!spectral_is_finite_positive_f32(stretch) || stretch > SPECTRAL_MAX_STRETCH) return 0;
    if (!spectral_is_finite_f32(pitch) ||
        pitch < SPECTRAL_MIN_PITCH ||
        pitch > SPECTRAL_MAX_PITCH) {
        return 0;
    }

    output_length_u64 = (uint64_t)output_length;
    if ((size_t)output_length_u64 != output_length) {
        return 0;
    }

    return 1;
}

SpectralError spectral_seg_cache_store(const char* cache_dir,
                                       uint64_t key,
                                       const SegmentArray* sa,
                                       int sample_rate,
                                       float stretch, float pitch,
                                       size_t output_length,
                                       const void* tile_ranges,
                                       const uint32_t* tile_segment_ids,
                                       uint32_t tile_count,
                                       uint32_t tile_total_refs)
{
    SpectralSegCacheEntry* entries = NULL;
    uint32_t count = 0;
    SpectralError err;
    uint64_t data_offset = 0;
    int needs_swap = spectral_is_big_endian();
    int has_tile_blob = (tile_count > 0u && tile_total_refs > 0u &&
                         tile_ranges && tile_segment_ids);

    if (!cache_dir || !sa) return SPECTRAL_ERR_PARAM;
    if (!seg_cache_store_metadata_valid(key, sa, sample_rate, stretch, pitch, output_length)) {
        return SPECTRAL_ERR_PARAM;
    }

    /* 1. Append segment data (+ optional tile data) to data file. */
    {
        SpectralSegCacheFsAppendWriter w = {0};
        size_t seg_bytes = 0;
        size_t gpu_seg_bytes = 0;
        size_t tile_ranges_bytes = 0;
        size_t tile_refs_bytes = 0;

        if (sa->count > 0) {
            if (!spectral_array_bytes((size_t)sa->count, sizeof(Segment), &seg_bytes) ||
                !spectral_array_bytes((size_t)sa->count, sizeof(SegmentGpu), &gpu_seg_bytes)) {
                return SPECTRAL_ERR_OVERFLOW;
            }
        }

        if (has_tile_blob) {
            if (!spectral_array_bytes((size_t)tile_count, sizeof(uint32_t) * 2u, &tile_ranges_bytes) ||
                !spectral_array_bytes((size_t)tile_total_refs, sizeof(uint32_t), &tile_refs_bytes)) {
                return SPECTRAL_ERR_OVERFLOW;
            }
        }

        err = spectral_seg_cache_fs_data_append_begin(cache_dir, &w);
        if (err != SPECTRAL_OK) return err;

        data_offset = w.data_offset;

        /* Write segments. */
        if (sa->count > 0) {
            if (needs_swap) {
                Segment* scratch = (Segment*)spectral_malloc_array((size_t)sa->count, sizeof(Segment));
                if (scratch) {
                    for (uint32_t i = 0; i < sa->count; i++) {
                        scratch[i] = sa->segs[i];
                        spectral_segment_swap_endian(&scratch[i]);
                    }
                    err = spectral_seg_cache_fs_data_append_write(&w, scratch, seg_bytes);
                    free(scratch);
                    if (err != SPECTRAL_OK) goto append_error;
                } else {
                    for (uint32_t i = 0; i < sa->count; i++) {
                        Segment seg = sa->segs[i];
                        spectral_segment_swap_endian(&seg);
                        err = spectral_seg_cache_fs_data_append_write(&w, &seg, sizeof(seg));
                        if (err != SPECTRAL_OK) goto append_error;
                    }
                }
            } else {
                err = spectral_seg_cache_fs_data_append_write(&w, sa->segs, seg_bytes);
                if (err != SPECTRAL_OK) goto append_error;
            }
        }

        /* Write pre-packed GPU segments (SegmentGpu, 32 bytes each). */
        if (sa->count > 0) {
            SegmentGpu* scratch = (SegmentGpu*)spectral_malloc_array((size_t)sa->count, sizeof(SegmentGpu));
            if (scratch) {
                spectral_segment_pack_gpu_array(sa->segs, sa->count, scratch);
                if (needs_swap) {
                    for (uint32_t i = 0; i < sa->count; i++) {
                        spectral_segment_gpu_swap_endian(&scratch[i]);
                    }
                }
                err = spectral_seg_cache_fs_data_append_write(&w, scratch, gpu_seg_bytes);
                free(scratch);
                if (err != SPECTRAL_OK) goto append_error;
            } else {
                for (uint32_t i = 0; i < sa->count; i++) {
                    SegmentGpu gs = spectral_segment_pack_gpu(&sa->segs[i]);
                    if (needs_swap) {
                        spectral_segment_gpu_swap_endian(&gs);
                    }
                    err = spectral_seg_cache_fs_data_append_write(&w, &gs, sizeof(gs));
                    if (err != SPECTRAL_OK) goto append_error;
                }
            }
        }

        /* Write tile data (header + ranges + segment IDs). */
        if (has_tile_blob) {
            SpectralSegCacheTileHeader th = {0};
            th.tile_size = SPECTRAL_GPU_TILE_SIZE;
            th.num_tiles = tile_count;
            th.total_refs = tile_total_refs;
            th.reserved = 0;
            if (needs_swap) {
                th.tile_size = spectral_swap_u32(th.tile_size);
                th.num_tiles = spectral_swap_u32(th.num_tiles);
                th.total_refs = spectral_swap_u32(th.total_refs);
            }
            err = spectral_seg_cache_fs_data_append_write(&w, &th, sizeof(th));
            if (err != SPECTRAL_OK) goto append_error;

            if (needs_swap) {
                uint32_t* ranges_scratch = (uint32_t*)spectral_malloc_array((size_t)tile_count, sizeof(uint32_t) * 2u);
                uint32_t* refs_scratch = (uint32_t*)spectral_malloc_array((size_t)tile_total_refs, sizeof(uint32_t));
                if (ranges_scratch && refs_scratch) {
                    const char* range_src = (const char*)tile_ranges;
                    uint32_t* range_dst = ranges_scratch;

                    for (uint32_t i = 0; i < tile_count; i++) {
                        memcpy(range_dst, range_src, sizeof(uint32_t) * 2u);
                        range_dst[0] = spectral_swap_u32(range_dst[0]);
                        range_dst[1] = spectral_swap_u32(range_dst[1]);
                        range_dst += 2u;
                        range_src += sizeof(uint32_t) * 2u;
                    }
                    for (uint32_t i = 0; i < tile_total_refs; i++) {
                        refs_scratch[i] = spectral_swap_u32(tile_segment_ids[i]);
                    }
                    err = spectral_seg_cache_fs_data_append_write(&w, ranges_scratch, tile_ranges_bytes);
                    if (err == SPECTRAL_OK) {
                        err = spectral_seg_cache_fs_data_append_write(&w, refs_scratch, tile_refs_bytes);
                    }
                    free(ranges_scratch);
                    free(refs_scratch);
                    if (err != SPECTRAL_OK) goto append_error;
                } else {
                    const char* range_src = (const char*)tile_ranges;

                    free(ranges_scratch);
                    free(refs_scratch);
                    for (uint32_t i = 0; i < tile_count; i++) {
                        uint32_t buf[2];
                        memcpy(buf, range_src, sizeof(buf));
                        range_src += sizeof(buf);
                        buf[0] = spectral_swap_u32(buf[0]);
                        buf[1] = spectral_swap_u32(buf[1]);
                        err = spectral_seg_cache_fs_data_append_write(&w, buf, sizeof(buf));
                        if (err != SPECTRAL_OK) goto append_error;
                    }
                    for (uint32_t i = 0; i < tile_total_refs; i++) {
                        uint32_t val = spectral_swap_u32(tile_segment_ids[i]);
                        err = spectral_seg_cache_fs_data_append_write(&w, &val, sizeof(val));
                        if (err != SPECTRAL_OK) goto append_error;
                    }
                }
            } else {
                err = spectral_seg_cache_fs_data_append_write(&w, tile_ranges, tile_ranges_bytes);
                if (err != SPECTRAL_OK) goto append_error;

                err = spectral_seg_cache_fs_data_append_write(&w, tile_segment_ids, tile_refs_bytes);
                if (err != SPECTRAL_OK) goto append_error;
            }
        }

        err = spectral_seg_cache_fs_data_append_end(&w, &data_offset);
        if (err != SPECTRAL_OK) goto append_error;

        goto append_success;

    append_error:
        spectral_seg_cache_fs_data_append_abort(&w);
        return err;

    append_success: ;
    }

    /* 2. Load existing index. */
    err = spectral_seg_cache_fs_index_load(cache_dir, &entries, &count);
    if (err != SPECTRAL_OK) {
        spectral_log_warn_codef(SPECTRAL_ERROR_DOMAIN_CORE, err,
                                "Segment cache index load failed; rebuilding index");
        entries = NULL;
        count = 0;
    }

    /* 3. Insert or update entry. */
    {
        int64_t pos = (count > 0) ? seg_cache_bsearch(entries, count, key) : ~(int64_t)0;
        uint32_t tc = has_tile_blob ? tile_count : 0;
        uint32_t tr = has_tile_blob ? tile_total_refs : 0;

        if (pos >= 0) {
            entries[pos].sample_rate = (uint32_t)sample_rate;
            entries[pos].stretch = stretch;
            entries[pos].pitch = pitch;
            entries[pos].seg_count = sa->count;
            entries[pos].data_offset = data_offset;
            entries[pos].output_length = (uint64_t)output_length;
            entries[pos].tile_count = tc;
            entries[pos].tile_total_refs = tr;
        } else {
            uint32_t ins = (uint32_t)(~pos);
            size_t new_count = 0;
            size_t new_entries_bytes = 0;
            size_t prefix_bytes = 0;
            size_t suffix_count = 0;
            size_t suffix_bytes = 0;
            SpectralSegCacheEntry* new_entries = NULL;

            if (ins > count) {
                free(entries);
                return SPECTRAL_ERR_FILE_CORRUPT;
            }

            /* Index insertion changes the persistent entry count.  This must be
             * checked in size_t and against the uint32_t on-disk field before
             * allocating or copying.  `(uint32_t)(count + 1)` would wrap at
             * UINT32_MAX and create an undersized index buffer. */
            if (!spectral_size_add((size_t)count, 1u, &new_count) ||
                new_count > (size_t)UINT32_MAX ||
                !spectral_array_bytes(new_count, sizeof(SpectralSegCacheEntry), &new_entries_bytes) ||
                !spectral_array_bytes((size_t)ins, sizeof(SpectralSegCacheEntry), &prefix_bytes)) {
                free(entries);
                return SPECTRAL_ERR_OVERFLOW;
            }

            suffix_count = (size_t)count - (size_t)ins;
            if (!spectral_array_bytes(suffix_count, sizeof(SpectralSegCacheEntry), &suffix_bytes)) {
                free(entries);
                return SPECTRAL_ERR_OVERFLOW;
            }

            new_entries = (SpectralSegCacheEntry*)spectral_malloc_array(
                new_count, sizeof(SpectralSegCacheEntry));
            if (!new_entries) {
                free(entries);
                return SPECTRAL_ERR_MEMORY;
            }

            if (prefix_bytes > 0u && entries) {
                memcpy(new_entries, entries, prefix_bytes);
            }

            new_entries[ins] = (SpectralSegCacheEntry){
                .key = key,
                .sample_rate = (uint32_t)sample_rate,
                .stretch = stretch,
                .pitch = pitch,
                .seg_count = sa->count,
                .data_offset = data_offset,
                .output_length = (uint64_t)output_length,
                .tile_count = tc,
                .tile_total_refs = tr
            };

            if (suffix_bytes > 0u && entries) {
                memcpy(&new_entries[(size_t)ins + 1u], &entries[ins], suffix_bytes);
            }

            free(entries);
            entries = new_entries;
            count = (uint32_t)new_count;
        }
    }

    /* 4. Write updated index. */
    err = spectral_seg_cache_fs_index_write(cache_dir, entries, count);
    free(entries);
    return err;
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */
