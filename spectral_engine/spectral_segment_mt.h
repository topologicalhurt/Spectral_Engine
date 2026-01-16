/* spectral_segment_mt.h - Thread-safe segment array
 *
 * Double-buffered segment storage for hot-swapping during synthesis.
 * Loader thread queues new segments, synth thread reads current set.
 */
#ifndef SPECTRAL_SEGMENT_MT_H
#define SPECTRAL_SEGMENT_MT_H

#include "spectral_common.h"

#if !SPECTRAL_EMBEDDED
#include <pthread.h>

typedef struct SegmentArrayMT {
    SegmentArray array;
    SegmentArray pending;
    pthread_mutex_t mutex;
    int pending_swap;
    int initialized;
} SegmentArrayMT;

int segment_array_mt_init(SegmentArrayMT* sa);
void segment_array_mt_destroy(SegmentArrayMT* sa);
int segment_array_mt_load(SegmentArrayMT* sa, Segment* segs, uint32_t count);
void segment_array_mt_get(SegmentArrayMT* sa, SegmentArray* out);
int segment_array_mt_copy(SegmentArrayMT* sa, SegmentArray* out);

#endif
#endif
