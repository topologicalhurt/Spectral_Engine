/* spectral_segment_pool.h - Block-allocated segment storage
 *
 * Avoids realloc during analysis by pre-allocating segment blocks.
 */
#ifndef SPECTRAL_SEGMENT_POOL_H
#define SPECTRAL_SEGMENT_POOL_H

#include "spectral_common.h"

/* Segment pool is desktop/analysis-only (uses malloc) */
#if !SPECTRAL_EMBEDDED

#define SEGMENT_POOL_BLOCK_SIZE 4096

typedef struct SegmentPool {
    Segment** blocks;
    uint32_t num_blocks;
    uint32_t max_blocks;
    uint32_t count;
    uint32_t block_size;
} SegmentPool;

SpectralError segment_pool_init(SegmentPool* pool, uint32_t expected_count);
SpectralError segment_pool_push(SegmentPool* pool, const Segment* seg);
SegmentArray segment_pool_to_array(SegmentPool* pool);
void segment_pool_destroy(SegmentPool* pool);
void segment_pool_reset(SegmentPool* pool);

#endif /* !SPECTRAL_EMBEDDED */

#endif
