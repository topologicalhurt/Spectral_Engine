/* spectral_segment_mt.c - Thread-safe segment array */
#include "spectral_segment_mt.h"

#if !SPECTRAL_EMBEDDED

int segment_array_mt_init(SegmentArrayMT* sa) {
    if (!sa) return -1;
    memset(sa, 0, sizeof(*sa));
    if (pthread_mutex_init(&sa->mutex, NULL) != 0) return -1;
    sa->initialized = 1;
    return 0;
}

void segment_array_mt_destroy(SegmentArrayMT* sa) {
    if (!sa || !sa->initialized) return;
    pthread_mutex_lock(&sa->mutex);
    free(sa->array.segs);
    free(sa->pending.segs);
    sa->array.segs = NULL;
    sa->pending.segs = NULL;
    pthread_mutex_unlock(&sa->mutex);
    pthread_mutex_destroy(&sa->mutex);
    sa->initialized = 0;
}

int segment_array_mt_load(SegmentArrayMT* sa, Segment* segs, uint32_t count) {
    if (!sa || !sa->initialized) return -1;
    pthread_mutex_lock(&sa->mutex);
    free(sa->pending.segs);
    sa->pending.segs = segs;
    sa->pending.count = count;
    sa->pending.capacity = count;
    sa->pending_swap = 1;
    pthread_mutex_unlock(&sa->mutex);
    return 0;
}

void segment_array_mt_get(SegmentArrayMT* sa, SegmentArray* out) {
    if (!sa || !sa->initialized || !out) return;
    pthread_mutex_lock(&sa->mutex);
    if (sa->pending_swap) {
        free(sa->array.segs);
        sa->array = sa->pending;
        sa->pending.segs = NULL;
        sa->pending.count = 0;
        sa->pending_swap = 0;
    }
    *out = sa->array;
    pthread_mutex_unlock(&sa->mutex);
}

int segment_array_mt_copy(SegmentArrayMT* sa, SegmentArray* out) {
    if (!sa || !sa->initialized || !out) return -1;
    pthread_mutex_lock(&sa->mutex);
    if (sa->pending_swap) {
        free(sa->array.segs);
        sa->array = sa->pending;
        sa->pending.segs = NULL;
        sa->pending.count = 0;
        sa->pending_swap = 0;
    }
    out->count = sa->array.count;
    out->capacity = sa->array.count;
    out->segs = NULL;
    if (sa->array.count > 0 && sa->array.segs) {
        out->segs = (Segment*)malloc(sa->array.count * sizeof(Segment));
        if (!out->segs) {
            pthread_mutex_unlock(&sa->mutex);
            return -1;
        }
        memcpy(out->segs, sa->array.segs, sa->array.count * sizeof(Segment));
    }
    pthread_mutex_unlock(&sa->mutex);
    return 0;
}

#endif
