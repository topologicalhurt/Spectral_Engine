/* spectral_peak_chain.h - Peak Track Segment Blockchain Allocator */
#ifndef SPECTRAL_PEAK_CHAIN_H
#define SPECTRAL_PEAK_CHAIN_H

#include "spectral_peak_track_internal.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

SegBlock* seg_block_new(void);
void seg_chain_init(SegBlockChain* chain);
void seg_chain_free(SegBlockChain* chain);
TrackSegment* seg_chain_push(SegBlockChain* chain);
void seg_chain_copy_to(const SegBlockChain* chain, Segment* out);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PEAK_CHAIN_H */
