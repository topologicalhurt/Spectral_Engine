/* spectral_segment_mt.c - Thread-safe segment array */
#include "spectral_segment_mt.h"
#include "spectral_utils.h"
#include <stdlib.h>
#include <string.h>

SpectralError segment_array_mt_init(SegmentArrayMT* sa) {
    if (!sa) return SPECTRAL_ERR_PARAM;
    memset(sa, 0, sizeof(*sa));
    if (pthread_mutex_init(&sa->mutex, NULL) != 0) return SPECTRAL_ERR_BUSY;
    sa->initialized = 1;
    return SPECTRAL_OK;
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

static void segment_array_mt_apply_pending_locked(SegmentArrayMT* sa) {
    if (!sa || !sa->pending_swap) return;
    free(sa->array.segs);
    sa->array = sa->pending;
    sa->pending = (SegmentArray)SEGMENT_ARRAY_EMPTY;
    sa->pending_swap = 0;
}

SpectralError segment_array_mt_load(SegmentArrayMT* sa, Segment* segs, uint32_t count) {
    if (!sa || !sa->initialized) return SPECTRAL_ERR_PARAM;
    pthread_mutex_lock(&sa->mutex);
    free(sa->pending.segs);
    sa->pending.segs = segs;
    sa->pending.count = count;
    sa->pending.capacity = count;
    sa->pending_swap = 1;
    pthread_mutex_unlock(&sa->mutex);
    return SPECTRAL_OK;
}

void segment_array_mt_get(SegmentArrayMT* sa, SegmentArray* out) {
    if (!sa || !sa->initialized || !out) return;
    pthread_mutex_lock(&sa->mutex);
    segment_array_mt_apply_pending_locked(sa);
    *out = sa->array;
    pthread_mutex_unlock(&sa->mutex);
}

SpectralError segment_array_mt_copy(SegmentArrayMT* sa, SegmentArray* out) {
    if (!sa || !sa->initialized || !out) return SPECTRAL_ERR_PARAM;
    pthread_mutex_lock(&sa->mutex);
    segment_array_mt_apply_pending_locked(sa);
    out->count = sa->array.count;
    out->capacity = sa->array.count;
    out->segs = NULL;
    if (sa->array.count > 0 && sa->array.segs) {
        size_t copy_bytes = 0;
        if (!spectral_array_bytes((size_t)sa->array.count, sizeof(Segment), &copy_bytes)) {
            pthread_mutex_unlock(&sa->mutex);
            return SPECTRAL_ERR_OVERFLOW;
        }
        out->segs = (Segment*)spectral_malloc_array((size_t)sa->array.count, sizeof(Segment));
        if (!out->segs) {
            pthread_mutex_unlock(&sa->mutex);
            return SPECTRAL_ERR_MEMORY;
        }
        memcpy(out->segs, sa->array.segs, copy_bytes);
    }
    pthread_mutex_unlock(&sa->mutex);
    return SPECTRAL_OK;
}
