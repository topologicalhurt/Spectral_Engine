/* spectral_seg_cache_fs.c — Host filesystem I/O for segment cache. */
#include "spectral_seg_cache_fs.h"
#include "spectral_endian.h"
#include "spectral_utils.h"

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

#include <stdlib.h>
#include <string.h>

/* --- Endian helpers for cache-specific structs --------------------------- */

static void seg_cache_header_swap(SpectralSegCacheHeader* h)
{
    if (!spectral_is_big_endian()) return;
    h->version  = spectral_swap_u32(h->version);
    h->count    = spectral_swap_u32(h->count);
    h->reserved = spectral_swap_u32(h->reserved);
}

static void seg_cache_entry_swap(SpectralSegCacheEntry* e)
{
    if (!spectral_is_big_endian()) return;
    e->key             = spectral_swap_u64(e->key);
    e->sample_rate     = spectral_swap_u32(e->sample_rate);
    e->stretch         = spectral_swap_float(e->stretch);
    e->pitch           = spectral_swap_float(e->pitch);
    e->seg_count       = spectral_swap_u32(e->seg_count);
    e->data_offset     = spectral_swap_u64(e->data_offset);
    e->output_length   = spectral_swap_u64(e->output_length);
    e->tile_count      = spectral_swap_u32(e->tile_count);
    e->tile_total_refs = spectral_swap_u32(e->tile_total_refs);
}

static int seg_cache_index_path(char* out, size_t out_size, const char* cache_dir)
{
    return spectral_path_join(out, out_size, cache_dir, SPECTRAL_SEG_CACHE_INDEX_FILE);
}

static int seg_cache_data_path(char* out, size_t out_size, const char* cache_dir)
{
    return spectral_path_join(out, out_size, cache_dir, SPECTRAL_SEG_CACHE_DATA_FILE);
}

/* --- Index file ---------------------------------------------------------- */

SpectralError spectral_seg_cache_fs_index_load(const char* cache_dir,
                                               SpectralSegCacheEntry** out_entries,
                                               uint32_t* out_count)
{
    char path[SPECTRAL_PIPELINE_PATH_CAPACITY];
    FILE* f = NULL;
    int missing = 0;
    SpectralSegCacheHeader hdr;
    SpectralSegCacheEntry* entries = NULL;
    size_t entries_bytes = 0;
    SpectralError err = SPECTRAL_OK;

    if (!cache_dir || !out_entries || !out_count) return SPECTRAL_ERR_PARAM;
    *out_entries = NULL;
    *out_count   = 0;

    if (!seg_cache_index_path(path, sizeof(path), cache_dir))
        return SPECTRAL_ERR_PARAM;

    err = spectral_fs_open_optional(&f, &missing, path, "rb");
    if (err != SPECTRAL_OK) return err;
    if (missing) return SPECTRAL_OK; /* no index yet */

    err = spectral_fs_read_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) goto cleanup;

    if (memcmp(hdr.magic, SPECTRAL_SEG_CACHE_INDEX_MAGIC, 4) != 0) {
        err = SPECTRAL_ERR_FILE_FORMAT;
        goto cleanup;
    }
    seg_cache_header_swap(&hdr);

    if (hdr.version != SPECTRAL_SEG_CACHE_INDEX_VERSION) {
        err = SPECTRAL_ERR_FILE_VERSION;
        goto cleanup;
    }
    if (hdr.count == 0) {
        uint64_t file_size = 0;
        err = spectral_fs_file_size(f, &file_size);
        if (err != SPECTRAL_OK) goto cleanup;
        if (file_size != (uint64_t)sizeof(SpectralSegCacheHeader)) {
            err = SPECTRAL_ERR_FILE_CORRUPT;
            goto cleanup;
        }
        err = SPECTRAL_OK;
        goto cleanup;
    }

    if (!spectral_array_bytes((size_t)hdr.count, sizeof(SpectralSegCacheEntry), &entries_bytes)) {
        err = SPECTRAL_ERR_OVERFLOW;
        goto cleanup;
    }

    {
        uint64_t file_size = 0;
        size_t expected_index_bytes = 0;

        if (!spectral_size_add(sizeof(SpectralSegCacheHeader), entries_bytes, &expected_index_bytes)) {
            err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }
        err = spectral_fs_file_size(f, &file_size);
        if (err != SPECTRAL_OK) goto cleanup;

        /* The index file is rewritten with "wb", so its exact length is part
         * of the persistence contract. Validate the count-derived byte length
         * before allocating; otherwise a corrupt header can request a huge
         * allocation even when the file cannot contain that many entries. */
        if (file_size != (uint64_t)expected_index_bytes) {
            err = SPECTRAL_ERR_FILE_CORRUPT;
            goto cleanup;
        }
    }

    entries = (SpectralSegCacheEntry*)spectral_malloc_array(
        (size_t)hdr.count, sizeof(SpectralSegCacheEntry));
    if (!entries) {
        err = SPECTRAL_ERR_MEMORY;
        goto cleanup;
    }

    err = spectral_fs_read_exact(
        f, entries, entries_bytes,
        SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) {
        free(entries);
        entries = NULL;
        goto cleanup;
    }

    for (uint32_t i = 0; i < hdr.count; i++) {
        seg_cache_entry_swap(&entries[i]);
    }

    *out_entries = entries;
    *out_count   = hdr.count;
    err = SPECTRAL_OK;

cleanup:
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_READ);
        if (err == SPECTRAL_OK && close_err != SPECTRAL_OK) err = close_err;
    }
    return err;
}

SpectralError spectral_seg_cache_fs_index_write(const char* cache_dir,
                                                const SpectralSegCacheEntry* entries,
                                                uint32_t count)
{
    char path[SPECTRAL_PIPELINE_PATH_CAPACITY];
    FILE* f = NULL;
    SpectralSegCacheHeader hdr;
    SpectralSegCacheEntry* tmp = NULL;
    size_t entries_bytes = 0;
    SpectralError err = SPECTRAL_OK;

    if (!cache_dir) return SPECTRAL_ERR_PARAM;
    if (count > 0 && !entries) return SPECTRAL_ERR_PARAM;

    if (!seg_cache_index_path(path, sizeof(path), cache_dir))
        return SPECTRAL_ERR_PARAM;

    err = spectral_fs_open(&f, path, "wb");
    if (err != SPECTRAL_OK) return err;

    memcpy(hdr.magic, SPECTRAL_SEG_CACHE_INDEX_MAGIC, 4);
    hdr.version  = SPECTRAL_SEG_CACHE_INDEX_VERSION;
    hdr.count    = count;
    hdr.reserved = 0;
    seg_cache_header_swap(&hdr);

    err = spectral_fs_write_exact(f, &hdr, sizeof(hdr), SPECTRAL_ERR_FILE_WRITE);
    if (err != SPECTRAL_OK) goto cleanup;

    if (count > 0) {
        if (!spectral_array_bytes((size_t)count, sizeof(SpectralSegCacheEntry), &entries_bytes)) {
            err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }
        tmp = (SpectralSegCacheEntry*)spectral_malloc_array(
            (size_t)count, sizeof(SpectralSegCacheEntry));
        if (!tmp) {
            err = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }
        memcpy(tmp, entries, entries_bytes);
        for (uint32_t i = 0; i < count; i++) {
            seg_cache_entry_swap(&tmp[i]);
        }

        err = spectral_fs_write_exact(
            f, tmp, entries_bytes,
            SPECTRAL_ERR_FILE_WRITE);
        if (err != SPECTRAL_OK) goto cleanup;
    }

cleanup:
    free(tmp);
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_WRITE);
        if (err == SPECTRAL_OK && close_err != SPECTRAL_OK) err = close_err;
    }
    return err;
}

/* --- Data file append/read ----------------------------------------------- */

SpectralError spectral_seg_cache_fs_data_append_begin(const char* cache_dir,
                                                      SpectralSegCacheFsAppendWriter* out_writer)
{
    char path[SPECTRAL_PIPELINE_PATH_CAPACITY];
    FILE* f = NULL;
    SpectralError err = SPECTRAL_OK;
    uint64_t pos = 0;

    if (!cache_dir || !out_writer) return SPECTRAL_ERR_PARAM;
    *out_writer = (SpectralSegCacheFsAppendWriter){0};

    if (!seg_cache_data_path(path, sizeof(path), cache_dir))
        return SPECTRAL_ERR_PARAM;

    err = spectral_fs_open(&f, path, "ab");
    if (err != SPECTRAL_OK) return err;

    err = spectral_fs_seek(f, 0, SEEK_END, SPECTRAL_ERR_FILE_WRITE);
    if (err != SPECTRAL_OK) {
        spectral_fs_close(&f, SPECTRAL_ERR_FILE_WRITE);
        return err;
    }
    err = spectral_fs_tell(f, &pos, SPECTRAL_ERR_FILE_WRITE);
    if (err != SPECTRAL_OK) {
        spectral_fs_close(&f, SPECTRAL_ERR_FILE_WRITE);
        return err;
    }

    out_writer->file = f;
    out_writer->data_offset = pos;
    return SPECTRAL_OK;
}

SpectralError spectral_seg_cache_fs_data_append_write(SpectralSegCacheFsAppendWriter* writer,
                                                      const void* data,
                                                      size_t bytes)
{
    if (!writer || !writer->file) return SPECTRAL_ERR_PARAM;
    return spectral_fs_write_exact(writer->file, data, bytes, SPECTRAL_ERR_FILE_WRITE);
}

SpectralError spectral_seg_cache_fs_data_append_end(SpectralSegCacheFsAppendWriter* writer,
                                                    uint64_t* out_data_offset)
{
    SpectralError err = SPECTRAL_OK;

    if (!writer || !writer->file) return SPECTRAL_ERR_PARAM;

    if (out_data_offset) *out_data_offset = writer->data_offset;
    err = spectral_fs_close(&writer->file, SPECTRAL_ERR_FILE_WRITE);
    *writer = (SpectralSegCacheFsAppendWriter){0};
    return err;
}

void spectral_seg_cache_fs_data_append_abort(SpectralSegCacheFsAppendWriter* writer)
{
    if (!writer) return;
    (void)spectral_fs_close(&writer->file, SPECTRAL_ERR_FILE_WRITE);
    *writer = (SpectralSegCacheFsAppendWriter){0};
}

SpectralError spectral_seg_cache_fs_data_file_size(const char* cache_dir,
                                                   uint64_t* out_size)
{
    char path[SPECTRAL_PIPELINE_PATH_CAPACITY];
    FILE* f = NULL;
    SpectralError err = SPECTRAL_OK;

    if (!cache_dir || !out_size) return SPECTRAL_ERR_PARAM;
    *out_size = 0;

    if (!seg_cache_data_path(path, sizeof(path), cache_dir))
        return SPECTRAL_ERR_PARAM;

    err = spectral_fs_open(&f, path, "rb");
    if (err != SPECTRAL_OK) return err;

    err = spectral_fs_file_size(f, out_size);
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_READ);
        if (err == SPECTRAL_OK && close_err != SPECTRAL_OK) err = close_err;
    }
    return err;
}

SpectralError spectral_seg_cache_fs_data_read_exact(const char* cache_dir,
                                                    uint64_t offset,
                                                    void* dst,
                                                    size_t bytes)
{
    char path[SPECTRAL_PIPELINE_PATH_CAPACITY];

    if (!cache_dir || !dst) return SPECTRAL_ERR_PARAM;
    if (bytes == 0) return SPECTRAL_OK;

    if (!seg_cache_data_path(path, sizeof(path), cache_dir))
        return SPECTRAL_ERR_PARAM;

    return spectral_fs_read_exact_path(path, offset, dst, bytes);
}

/* --- mmap view ----------------------------------------------------------- */

int spectral_seg_cache_fs_has_mmap(void)
{
    return spectral_fs_has_mmap();
}

SpectralError spectral_seg_cache_fs_data_map_ro(const char* cache_dir,
                                                uint64_t offset,
                                                size_t bytes,
                                                SpectralSegCacheFsMapView* out_view)
{
    char path[SPECTRAL_PIPELINE_PATH_CAPACITY];

    if (!cache_dir || !out_view) return SPECTRAL_ERR_PARAM;
    *out_view = (SpectralSegCacheFsMapView){0};

    if (!seg_cache_data_path(path, sizeof(path), cache_dir))
        return SPECTRAL_ERR_PARAM;

    return spectral_fs_map_ro_path(path, offset, bytes, out_view);
}

void spectral_seg_cache_fs_data_unmap(SpectralSegCacheFsMapView* view)
{
    spectral_fs_unmap(view);
}

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */
