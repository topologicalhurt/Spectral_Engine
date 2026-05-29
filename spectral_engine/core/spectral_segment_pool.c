/* spectral_segment_pool.c - Block-allocated segment storage */
#include "spectral_segment_pool.h"
#include "spectral_utils.h"

/* Segment pool is desktop/analysis-only (build-selected; see source-manifest). */
#include <stdlib.h>
#include <string.h>

SpectralError segment_pool_init(SegmentPool* pool, uint32_t expected_count) {
    if (!pool) return SPECTRAL_ERR_PARAM;
    
    pool->block_size = SPECTRAL_SEGMENT_POOL_BLOCK_SIZE;
    pool->num_blocks = 0;
    pool->count = 0;
    
    uint32_t est_blocks = (expected_count + pool->block_size - 1) / pool->block_size;
    if (est_blocks < 4) est_blocks = 4;
    
    pool->max_blocks = est_blocks;
    pool->blocks = (Segment**)spectral_calloc_array((size_t)est_blocks, sizeof(Segment*));
    return pool->blocks ? SPECTRAL_OK : SPECTRAL_ERR_MEMORY;
}

SpectralError segment_pool_push(SegmentPool* pool, const Segment* seg) {
    if (!pool || !seg) return SPECTRAL_ERR_PARAM;
    
    uint32_t block_idx = pool->count / pool->block_size;
    uint32_t slot_idx = pool->count % pool->block_size;
    
    if (block_idx >= pool->num_blocks) {
        if (block_idx >= pool->max_blocks) {
            uint32_t new_max = pool->max_blocks * 2;
            if (new_max < pool->max_blocks) return SPECTRAL_ERR_OVERFLOW;
            Segment** new_blocks = (Segment**)spectral_realloc_array(
                pool->blocks, (size_t)new_max, sizeof(Segment*));
            if (!new_blocks) return SPECTRAL_ERR_MEMORY;
            pool->blocks = new_blocks;
            pool->max_blocks = new_max;
        }

        Segment* new_block = (Segment*)spectral_malloc_array((size_t)pool->block_size, sizeof(Segment));
        if (!new_block) return SPECTRAL_ERR_MEMORY;
        pool->blocks[block_idx] = new_block;
        pool->num_blocks++;
    }
    
    pool->blocks[block_idx][slot_idx] = *seg;
    pool->count++;
    return SPECTRAL_OK;
}

SegmentArray segment_pool_to_array(SegmentPool* pool) {
    SegmentArray sa = SEGMENT_ARRAY_EMPTY;
    if (!pool || pool->count == 0) return sa;
    
    sa.segs = (Segment*)spectral_malloc_array((size_t)pool->count, sizeof(Segment));
    if (!sa.segs) return sa;
    
    uint32_t copied = 0;
    for (uint32_t b = 0; b < pool->num_blocks && copied < pool->count; b++) {
        uint32_t to_copy = pool->block_size;
        if (copied + to_copy > pool->count) {
            to_copy = pool->count - copied;
        }
        size_t copy_bytes = 0;
        if (!spectral_array_bytes((size_t)to_copy, sizeof(Segment), &copy_bytes)) {
            free(sa.segs);
            sa = (SegmentArray)SEGMENT_ARRAY_EMPTY;
            return sa;
        }
        memcpy(&sa.segs[copied], pool->blocks[b], copy_bytes);
        copied += to_copy;
    }
    
    sa.count = pool->count;
    sa.capacity = pool->count;
    return sa;
}

void segment_pool_destroy(SegmentPool* pool) {
    if (!pool) return;
    for (uint32_t b = 0; b < pool->num_blocks; b++) {
        free(pool->blocks[b]);
    }
    free(pool->blocks);
    pool->blocks = NULL;
    pool->num_blocks = 0;
    pool->max_blocks = 0;
    pool->count = 0;
}

void segment_pool_reset(SegmentPool* pool) {
    if (pool) pool->count = 0;
}
