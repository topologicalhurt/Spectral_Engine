/* spectral_seg_cache_fs.h — Host filesystem I/O for segment cache.
 *
 * Keeps cache file operations behind a single host-only boundary so the cache
 * policy code remains portable across build targets.
 */
#ifndef SPECTRAL_SEG_CACHE_FS_H
#define SPECTRAL_SEG_CACHE_FS_H

#include "spectral_fs.h"
#include "spectral_seg_cache.h"
#include <stddef.h>
#include <stdint.h>

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM

#include <stdio.h>

typedef struct {
    FILE*    file;
    uint64_t data_offset;
} SpectralSegCacheFsAppendWriter;

typedef SpectralFsMapView SpectralSegCacheFsMapView;

SpectralError spectral_seg_cache_fs_index_load(const char* cache_dir,
                                               SpectralSegCacheEntry** out_entries,
                                               uint32_t* out_count);

SpectralError spectral_seg_cache_fs_index_write(const char* cache_dir,
                                                const SpectralSegCacheEntry* entries,
                                                uint32_t count);

SpectralError spectral_seg_cache_fs_data_append_begin(const char* cache_dir,
                                                      SpectralSegCacheFsAppendWriter* out_writer);
SpectralError spectral_seg_cache_fs_data_append_write(SpectralSegCacheFsAppendWriter* writer,
                                                      const void* data,
                                                      size_t bytes);
SpectralError spectral_seg_cache_fs_data_append_end(SpectralSegCacheFsAppendWriter* writer,
                                                    uint64_t* out_data_offset);
void spectral_seg_cache_fs_data_append_abort(SpectralSegCacheFsAppendWriter* writer);

SpectralError spectral_seg_cache_fs_data_file_size(const char* cache_dir,
                                                   uint64_t* out_size);
SpectralError spectral_seg_cache_fs_data_read_exact(const char* cache_dir,
                                                    uint64_t offset,
                                                    void* dst,
                                                    size_t bytes);

int spectral_seg_cache_fs_has_mmap(void);
SpectralError spectral_seg_cache_fs_data_map_ro(const char* cache_dir,
                                                uint64_t offset,
                                                size_t bytes,
                                                SpectralSegCacheFsMapView* out_view);
void spectral_seg_cache_fs_data_unmap(SpectralSegCacheFsMapView* view);

#endif /* !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM */

#endif /* SPECTRAL_SEG_CACHE_FS_H */
