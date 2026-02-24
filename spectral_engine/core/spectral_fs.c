/* spectral_fs.c — Shared host binary file I/O helpers. */
#include "spectral_fs.h"

#include <errno.h>

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

#include <string.h>
#include <limits.h>

#include <sys/types.h>
typedef off_t spectral_fs_off_t;
#define internal_fs_seek fseeko
#define internal_fs_tell ftello

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#define SPECTRAL_FS_HAS_MMAP 1
#else
#define SPECTRAL_FS_HAS_MMAP 0
#endif

static int spectral_fs_u64_to_off(uint64_t value, spectral_fs_off_t* out)
{
    spectral_fs_off_t tmp = 0;
    if (!out) return 0;
    tmp = (spectral_fs_off_t)value;
    if ((uint64_t)tmp != value) return 0;
    if (tmp < 0) return 0;
    *out = tmp;
    return 1;
}

static SpectralError spectral_fs_seek_u64(FILE* file,
                                                 uint64_t offset,
                                                 int whence,
                                                 SpectralError on_error)
{
    spectral_fs_off_t fs_off = 0;
    if (!file) return SPECTRAL_ERR_PARAM;
    if (!spectral_fs_u64_to_off(offset, &fs_off)) {
        return SPECTRAL_ERR_OVERFLOW;
    }
    return (internal_fs_seek(file, fs_off, whence) == 0) ? SPECTRAL_OK : on_error;
}

SpectralError spectral_fs_open(FILE** out_file,
                                      const char* path,
                                      const char* mode)
{
    if (!out_file || !path || !mode) return SPECTRAL_ERR_PARAM;
    *out_file = fopen(path, mode);
    return *out_file ? SPECTRAL_OK : SPECTRAL_ERR_FILE_OPEN;
}

SpectralError spectral_fs_open_optional(FILE** out_file,
                                               int* out_missing,
                                               const char* path,
                                               const char* mode)
{
    if (!out_file || !out_missing || !path || !mode) return SPECTRAL_ERR_PARAM;
    *out_file = NULL;
    *out_missing = 0;

    errno = 0;
    *out_file = fopen(path, mode);
    if (*out_file) return SPECTRAL_OK;
    if (errno == ENOENT) {
        *out_missing = 1;
        return SPECTRAL_OK;
    }
    return SPECTRAL_ERR_FILE_OPEN;
}

SpectralError spectral_fs_close(FILE** file, SpectralError on_error)
{
    if (!file || !*file) return SPECTRAL_OK;
    if (fclose(*file) != 0) {
        *file = NULL;
        return on_error;
    }
    *file = NULL;
    return SPECTRAL_OK;
}

SpectralError spectral_fs_seek(FILE* file, int64_t offset, int whence, SpectralError on_error)
{
    if (offset < 0) return SPECTRAL_ERR_PARAM;
    return spectral_fs_seek_u64(file, (uint64_t)offset, whence, on_error);
}

SpectralError spectral_fs_tell(FILE* file, uint64_t* out_pos, SpectralError on_error)
{
    spectral_fs_off_t pos = 0;
    if (!file || !out_pos) return SPECTRAL_ERR_PARAM;
    pos = internal_fs_tell(file);
    if (pos < 0) return on_error;
    *out_pos = (uint64_t)pos;
    return SPECTRAL_OK;
}

SpectralError spectral_fs_read_exact(FILE* file, void* dst, size_t bytes, SpectralError on_error)
{
    if (!file || (bytes > 0 && !dst)) return SPECTRAL_ERR_PARAM;
    if (bytes == 0) return SPECTRAL_OK;
    return (fread(dst, 1, bytes, file) == bytes) ? SPECTRAL_OK : on_error;
}

SpectralError spectral_fs_write_exact(FILE* file, const void* src, size_t bytes, SpectralError on_error)
{
    if (!file || (bytes > 0 && !src)) return SPECTRAL_ERR_PARAM;
    if (bytes == 0) return SPECTRAL_OK;
    return (fwrite(src, 1, bytes, file) == bytes) ? SPECTRAL_OK : on_error;
}

SpectralError spectral_fs_file_size(FILE* file, uint64_t* out_size)
{
    uint64_t cur = 0;
    uint64_t end = 0;
    SpectralError err = SPECTRAL_OK;

    if (!file || !out_size) return SPECTRAL_ERR_PARAM;
    *out_size = 0;

    err = spectral_fs_tell(file, &cur, SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) return err;
    err = spectral_fs_seek(file, 0, SEEK_END, SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) return err;
    err = spectral_fs_tell(file, &end, SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) return err;
    err = spectral_fs_seek_u64(file, cur, SEEK_SET, SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) return err;

    *out_size = end;
    return SPECTRAL_OK;
}

SpectralError spectral_fs_read_exact_path(const char* path,
                                                 uint64_t offset,
                                                 void* dst,
                                                 size_t bytes)
{
    FILE* f = NULL;
    SpectralError err = SPECTRAL_OK;
    uint64_t size = 0;

    if (!path || (bytes > 0 && !dst)) return SPECTRAL_ERR_PARAM;
    if (bytes == 0) return SPECTRAL_OK;

    err = spectral_fs_open(&f, path, "rb");
    if (err != SPECTRAL_OK) return err;

    err = spectral_fs_file_size(f, &size);
    if (err != SPECTRAL_OK) goto cleanup;

    if (bytes > UINT64_MAX - offset || size < offset + bytes) {
        err = SPECTRAL_ERR_FILE_CORRUPT;
        goto cleanup;
    }

    err = spectral_fs_seek_u64(f, offset, SEEK_SET, SPECTRAL_ERR_FILE_READ);
    if (err != SPECTRAL_OK) goto cleanup;

    err = spectral_fs_read_exact(f, dst, bytes, SPECTRAL_ERR_FILE_READ);

cleanup:
    {
        SpectralError close_err = spectral_fs_close(&f, SPECTRAL_ERR_FILE_READ);
        if (err == SPECTRAL_OK && close_err != SPECTRAL_OK) err = close_err;
    }
    return err;
}

size_t spectral_fs_read(void* ptr, size_t size, size_t count, FILE* stream)
{
    if (!ptr || !stream || size == 0 || count == 0) return 0;
    return fread(ptr, size, count, stream);
}

size_t spectral_fs_write(const void* ptr, size_t size, size_t count, FILE* stream)
{
    if (!ptr || !stream || size == 0 || count == 0) return 0;
    return fwrite(ptr, size, count, stream);
}

char* spectral_fs_gets(char* str, int n, FILE* stream)
{
    if (!str || n <= 0 || !stream) return NULL;
    return fgets(str, n, stream);
}

int spectral_fs_has_mmap(void)
{
#if SPECTRAL_FS_HAS_MMAP
    return 1;
#else
    return 0;
#endif
}

SpectralError spectral_fs_map_ro_path(const char* path,
                                             uint64_t offset,
                                             size_t bytes,
                                             SpectralFsMapView* out_view)
{
    if (!path || !out_view) return SPECTRAL_ERR_PARAM;
    *out_view = (SpectralFsMapView){0};
    if (bytes == 0) return SPECTRAL_OK;

#if SPECTRAL_FS_HAS_MMAP
    {
        int fd = open(path, O_RDONLY);
        if (fd < 0) return SPECTRAL_ERR_BACKEND_UNAVAIL;

        {
            struct stat st;
            if (fstat(fd, &st) != 0) {
                close(fd);
                return SPECTRAL_ERR_FILE_READ;
            }
            if (bytes > UINT64_MAX - offset ||
                (uint64_t)st.st_size < offset + bytes) {
                close(fd);
                return SPECTRAL_ERR_FILE_CORRUPT;
            }
        }

        {
            long page_sz = sysconf(_SC_PAGESIZE);
            uint64_t page_off_u64 = 0;
            uint64_t map_start_u64 = 0;
            spectral_fs_off_t map_start = 0;
            size_t page_off = 0;
            size_t map_len = 0;
            void* base = NULL;

            if (page_sz <= 0) {
                close(fd);
                return SPECTRAL_ERR_BACKEND_UNAVAIL;
            }

            page_off_u64 = offset % (uint64_t)page_sz;
            map_start_u64 = offset - page_off_u64;
            page_off = (size_t)page_off_u64;
            if (bytes > SIZE_MAX - page_off) {
                close(fd);
                return SPECTRAL_ERR_OVERFLOW;
            }
            if (!spectral_fs_u64_to_off(map_start_u64, &map_start)) {
                close(fd);
                return SPECTRAL_ERR_OVERFLOW;
            }

            map_len = page_off + bytes;
            base = mmap(NULL, map_len, PROT_READ, MAP_PRIVATE, fd, map_start);
            close(fd);
            if (base == MAP_FAILED) return SPECTRAL_ERR_BACKEND_UNAVAIL;
#ifdef MADV_SEQUENTIAL
            madvise(base, map_len, MADV_SEQUENTIAL);
#endif
            out_view->base = base;
            out_view->len = map_len;
            out_view->page_offset = page_off;
            return SPECTRAL_OK;
        }
    }
#else
    (void)path;
    (void)offset;
    (void)bytes;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
#endif
}

void spectral_fs_unmap(SpectralFsMapView* view)
{
#if SPECTRAL_FS_HAS_MMAP
    if (!view) return;
    if (view->base) {
        munmap(view->base, view->len);
    }
#else
    (void)view;
#endif
    if (view) *view = (SpectralFsMapView){0};
}

#else /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */

SpectralError spectral_fs_open(FILE** out_file,
                                      const char* path,
                                      const char* mode)
{
    (void)out_file;
    (void)path;
    (void)mode;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_open_optional(FILE** out_file,
                                               int* out_missing,
                                               const char* path,
                                               const char* mode)
{
    (void)out_file;
    (void)out_missing;
    (void)path;
    (void)mode;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_close(FILE** file, SpectralError on_error)
{
    (void)file;
    (void)on_error;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_seek(FILE* file, int64_t offset, int whence, SpectralError on_error)
{
    (void)file;
    (void)offset;
    (void)whence;
    (void)on_error;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_tell(FILE* file, uint64_t* out_pos, SpectralError on_error)
{
    (void)file;
    (void)out_pos;
    (void)on_error;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_read_exact(FILE* file, void* dst, size_t bytes, SpectralError on_error)
{
    (void)file;
    (void)dst;
    (void)bytes;
    (void)on_error;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_write_exact(FILE* file, const void* src, size_t bytes, SpectralError on_error)
{
    (void)file;
    (void)src;
    (void)bytes;
    (void)on_error;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_file_size(FILE* file, uint64_t* out_size)
{
    (void)file;
    (void)out_size;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

SpectralError spectral_fs_read_exact_path(const char* path,
                                                 uint64_t offset,
                                                 void* dst,
                                                 size_t bytes)
{
    (void)path;
    (void)offset;
    (void)dst;
    (void)bytes;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

size_t spectral_fs_read(void* ptr, size_t size, size_t count, FILE* stream)
{
    (void)ptr; (void)size; (void)count; (void)stream;
    return 0;
}

size_t spectral_fs_write(const void* ptr, size_t size, size_t count, FILE* stream)
{
    (void)ptr; (void)size; (void)count; (void)stream;
    return 0;
}

char* spectral_fs_gets(char* str, int n, FILE* stream)
{
    (void)str; (void)n; (void)stream;
    return NULL;
}

int spectral_fs_has_mmap(void)
{
    return 0;
}

SpectralError spectral_fs_map_ro_path(const char* path,
                                             uint64_t offset,
                                             size_t bytes,
                                             SpectralFsMapView* out_view)
{
    (void)path;
    (void)offset;
    (void)bytes;
    (void)out_view;
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}

void spectral_fs_unmap(SpectralFsMapView* view)
{
    (void)view;
}

#endif
