/* spectral_peak_chain.c - Peak Track Segment Blockchain Allocator */
#include "spectral_peak_chain.h"

SegBlock* seg_block_new(void) {
    return (SegBlock*)calloc(1, sizeof(SegBlock));
}

void seg_chain_init(SegBlockChain* chain) {
    chain->head = NULL;
    chain->current = NULL;
    chain->total_count = 0;
}

void seg_chain_free(SegBlockChain* chain) {
    if (!chain) return;
    SegBlock* b = chain->head;
    while (b) {
        SegBlock* next = b->next;
        free(b);
        b = next;
    }
    chain->head = NULL;
    chain->current = NULL;
    chain->total_count = 0;
}

TrackSegment* seg_chain_push(SegBlockChain* chain) {
    SegBlock* curr = chain->current;
    if (SPECTRAL_UNLIKELY(!curr || curr->count >= SPECTRAL_TRACK_BLOCK_SEGS)) {
        SegBlock* next = seg_block_new();
        if (SPECTRAL_UNLIKELY(!next)) {
            return NULL;
        }
        if (curr) curr->next = next;
        else chain->head = next;
        chain->current = next;
        curr = next;
    }
    if (curr->count + SPECTRAL_TRACK_SEG_PREFETCH_DISTANCE < SPECTRAL_TRACK_BLOCK_SEGS) {
        SPECTRAL_PREFETCH_WRITE_LOCALITY(
            &curr->segs[curr->count + SPECTRAL_TRACK_SEG_PREFETCH_DISTANCE],
            SPECTRAL_TRACK_PREFETCH_WRITE_LOCALITY);
    }
    return &curr->segs[curr->count++];
}

void seg_chain_copy_to(const SegBlockChain* chain, Segment* out) {
    if (!chain || !out) return;
    SegBlock* blk = chain->head;
    size_t offset = 0;
    while (blk) {
        Segment* blk_out = out + offset;
        const TrackSegment* src = blk->segs;
        for (uint32_t i = 0; i < blk->count; i++) {
            memcpy(&blk_out[i], &src[i], sizeof(TrackSegment));
            if (sizeof(Segment) > sizeof(TrackSegment)) {
                memset((char*)&blk_out[i] + sizeof(TrackSegment), 0, sizeof(Segment) - sizeof(TrackSegment));
            }
        }
        offset += blk->count;
        blk = blk->next;
    }
}
