/* spectral_seg_cache.h — Centralized hash-indexed segment cache.
 *
 * Maintains a sorted binary index (seg_cache_index.bin) mapping hash keys to
 * segment data offsets in a companion append-only file (seg_cache_data.bin).
 * All on-disk data is little-endian.  Lookup is O(log n) via binary search.
 *
 * On hit, segment and GPU tile data are mmap'd read-only: zero-copy load,
 * instant cleanup (no dirty pages).  Use spectral_seg_cache_result_free().
 *
 * The cache key is produced by spectral_hash_oneshot() over the text
 * representation of analysis/render parameters (basename, n_fft, hop,
 * db threshold, window range, stretch, tile size).
 */
#ifndef SPECTRAL_SEG_CACHE_H
#define SPECTRAL_SEG_CACHE_H

#include "spectral_common.h"
#include "spectral_error.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* File format constants */

#define SPECTRAL_SEG_CACHE_INDEX_MAGIC   "SCIX"
#define SPECTRAL_SEG_CACHE_INDEX_VERSION 1u
#define SPECTRAL_SEG_CACHE_INDEX_FILE    "seg_cache_index.bin"
#define SPECTRAL_SEG_CACHE_DATA_FILE     "seg_cache_data.bin"

/* On-disk structures (all fields stored LE) */

typedef struct {
    char     magic[4];
    uint32_t version;
    uint32_t count;
    uint32_t reserved;
} SpectralSegCacheHeader;

typedef struct {
    uint64_t key;              /* hash of analysis parameter blob */
    uint32_t sample_rate;
    float    stretch;
    float    pitch;
    uint32_t seg_count;
    uint64_t data_offset;      /* byte offset into data file      */
    uint64_t output_length;    /* pre-computed max(start+length)   */
    uint32_t tile_count;       /* GPU tiles stored; 0 = none       */
    uint32_t tile_total_refs;  /* total segment-ID refs in tiles   */
} SpectralSegCacheEntry;

/* On-disk tile header (immediately after segment data in data file) */
typedef struct {
    uint32_t tile_size;    /* SPECTRAL_GPU_TILE_SIZE at build time */
    uint32_t num_tiles;
    uint32_t total_refs;
    uint32_t reserved;
} SpectralSegCacheTileHeader;

typedef struct {
    int          hit;             /* 1 = found, 0 = miss */
    int          sample_rate;
    float        stretch;
    float        pitch;
    uint32_t     seg_count;
    size_t       output_length;
    SegmentArray segments;        /* use spectral_seg_cache_result_free() */
    /* Pre-packed GPU segments (mmap'd, NULL on heap fallback) */
    const SegmentGpu* gpu_segs;
    /* Pre-computed GPU tile data (NULL/0 if not stored) */
    uint32_t     tile_size;
    uint32_t     tile_count;
    uint32_t     tile_total_refs;
    void*        tile_ranges;     /* TileRange[tile_count]         */
    uint32_t*    tile_segment_ids;/* uint32_t[tile_total_refs]     */
    /* Internal — mmap handle (NULL when heap-allocated) */
    void*        _mmap_base;
    size_t       _mmap_len;
} SpectralSegCacheLookupResult;

/* Compute cache key from audio input identity string and analysis/render parameters. */
uint64_t spectral_seg_cache_key(const char* input_id,
                                int n_fft, int hop,
                                float db_thresh,
                                float start_sec, float end_sec,
                                float stretch,
                                uint32_t tile_size);

/* O(log n) lookup.  Returns SPECTRAL_OK even on miss (result->hit == 0).
 * On hit, segment + tile data are mmap'd read-only (zero-copy).
 * Caller MUST call spectral_seg_cache_result_free() to release. */
SpectralError spectral_seg_cache_lookup(const char* cache_dir,
                                        uint64_t key,
                                        SpectralSegCacheLookupResult* result);

/* Release all resources held by a lookup result (mmap or heap).
 * Safe to call on miss results (no-op). */
void spectral_seg_cache_result_free(SpectralSegCacheLookupResult* result);

/* Transfer SegmentArray view from lookup result to caller.
 * Heap-backed hits transfer ownership (result_free will not free segments).
 * mmap-backed hits remain tied to the lookup result lifetime; caller must keep
 * the lookup result alive and call result_free after done with the segments. */
SegmentArray spectral_seg_cache_result_take_segments(
    SpectralSegCacheLookupResult* result);

/* Append segment + tile data then insert a sorted index entry.
 * output_length: pre-computed max(start+length) to avoid re-scanning.
 * tile_ranges / tile_segment_ids: pre-computed GPU tile data (NULL to skip). */
SpectralError spectral_seg_cache_store(const char* cache_dir,
                                       uint64_t key,
                                       const SegmentArray* sa,
                                       int sample_rate,
                                       float stretch, float pitch,
                                       size_t output_length,
                                       const void* tile_ranges,
                                       const uint32_t* tile_segment_ids,
                                       uint32_t tile_count,
                                       uint32_t tile_total_refs);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_SEG_CACHE_H */
