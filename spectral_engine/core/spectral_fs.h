/* spectral_fs.h — Shared host binary file I/O helpers.
 *
 * Centralizes low-level file and mmap primitives for modules that persist
 * little-endian binary data.  Callers own format/endianness policy.
 */
#ifndef SPECTRAL_FS_H
#define SPECTRAL_FS_H

#include "spectral_error.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void*  base;
    size_t len;
    size_t page_offset;
} SpectralFsMapView;

SpectralError spectral_fs_open(FILE** out_file,
                                      const char* path,
                                      const char* mode);

/* Like spectral_fs_open(), but missing files are not an error.
 * Returns SPECTRAL_OK in all success cases; out_missing is 1 on ENOENT. */
SpectralError spectral_fs_open_optional(FILE** out_file,
                                               int* out_missing,
                                               const char* path,
                                               const char* mode);

SpectralError spectral_fs_close(FILE** file, SpectralError on_error);
/* 64-bit seek offset; returns SPECTRAL_ERR_OVERFLOW when offset is unrepresentable. */
SpectralError spectral_fs_seek(FILE* file, int64_t offset, int whence, SpectralError on_error);
SpectralError spectral_fs_tell(FILE* file, uint64_t* out_pos, SpectralError on_error);
SpectralError spectral_fs_read_exact(FILE* file, void* dst, size_t bytes, SpectralError on_error);
SpectralError spectral_fs_write_exact(FILE* file, const void* src, size_t bytes, SpectralError on_error);
SpectralError spectral_fs_file_size(FILE* file, uint64_t* out_size);

/* Standard I/O wrappers for general-purpose reading/writing (e.g. text/binary blocks). */
size_t spectral_fs_read(void* ptr, size_t size, size_t count, FILE* stream);
size_t spectral_fs_write(const void* ptr, size_t size, size_t count, FILE* stream);
char* spectral_fs_gets(char* str, int n, FILE* stream);

/* Read bytes from a path at offset after bounds-checking against file size. */
SpectralError spectral_fs_read_exact_path(const char* path,
                                                 uint64_t offset,
                                                 void* dst,
                                                 size_t bytes);

int spectral_fs_has_mmap(void);
SpectralError spectral_fs_map_ro_path(const char* path,
                                             uint64_t offset,
                                             size_t bytes,
                                             SpectralFsMapView* out_view);
void spectral_fs_unmap(SpectralFsMapView* view);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_FS_H */
