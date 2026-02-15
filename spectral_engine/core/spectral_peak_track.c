/* spectral_peak_track.c - Peak Tracking with Block-Chain Allocator
 *
 * Extracts spectral peaks from STFT magnitude/phase data and creates
 * Segment objects for resynthesis. Uses a block-chain allocator to
 * avoid realloc() in the hot tracking loop.
 *
 * Two modes:
 *   1. spectral_track_peaks()  — single-shot (processes entire STFT)
 *   2. SpectralTracker API     — incremental (processes chunks)
 */

#include "spectral_peak_track.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_omp.h"

/*
 * Block-chain segment allocator
 *
 * Each thread gets its own chain of 1MB blocks. When the current block is
 * full, a new block is malloc'd and linked. No realloc ever touches existing data.
 *
 * Merge phase copies whole blocks (cache-friendly 1MB chunks) into the
 * final contiguous array.
 */

typedef struct SegBlock {
    Segment segs[TRACK_BLOCK_SEGS];
    uint32_t count;
    struct SegBlock* next;
} SegBlock;

typedef struct {
    SegBlock* head;
    SegBlock* current;
    uint32_t total_count;
} SegBlockChain;

static void seg_chain_init(SegBlockChain* chain) {
    chain->head = NULL;
    chain->current = NULL;
    chain->total_count = 0;
}

static SegBlock* seg_block_new(void) {
    SegBlock* blk = (SegBlock*)malloc(sizeof(SegBlock));
    if (blk) {
        blk->count = 0;
        blk->next = NULL;
    }
    return blk;
}

/* Returns pointer to a new segment slot, or NULL on allocation failure */
static Segment* seg_chain_push(SegBlockChain* chain) {
    if (!chain->current || chain->current->count >= TRACK_BLOCK_SEGS) {
        SegBlock* blk = seg_block_new();
        if (!blk) return NULL;
        if (chain->current) {
            chain->current->next = blk;
        } else {
            chain->head = blk;
        }
        chain->current = blk;
    }
    Segment* seg = &chain->current->segs[chain->current->count++];
    chain->total_count++;
    return seg;
}

static void seg_chain_free(SegBlockChain* chain) {
    SegBlock* blk = chain->head;
    while (blk) {
        SegBlock* next = blk->next;
        free(blk);
        blk = next;
    }
    chain->head = NULL;
    chain->current = NULL;
    chain->total_count = 0;
}

/* Copy chain contents into a contiguous destination */
static void seg_chain_copy_to(const SegBlockChain* chain, Segment* dst) {
    size_t offset = 0;
    SegBlock* blk = chain->head;
    while (blk) {
        memcpy(dst + offset, blk->segs, blk->count * sizeof(Segment));
        offset += blk->count;
        blk = blk->next;
    }
}

/* SpectralTracker — incremental tracking state */

struct SpectralTracker {
    SegBlockChain* chains;
    int n_threads;
    int alloc_failed;

    /* Precomputed constants */
    size_t n_freqs;
    float threshsq;
    float thresh_linear_sq;  /* db-derived factor, independent of max_magsq */
    float freq_step_omega;
    float freq_step_df;
    float inv_hop;
    float hop_float;

    double start_time;
};

SpectralTracker* spectral_tracker_create(int n_threads, size_t n_freqs,
                                          int sr, int n_fft, int hop,
                                          float db_thresh, float max_magsq) {
    SpectralTracker* tracker = (SpectralTracker*)malloc(sizeof(SpectralTracker));
    if (!tracker) return NULL;

    tracker->n_threads = n_threads;
    tracker->n_freqs = n_freqs;
    tracker->alloc_failed = 0;
    tracker->start_time = omp_get_wtime();

    float thresh_linear = powf(10.0f, db_thresh / 20.0f);
    tracker->thresh_linear_sq = thresh_linear * thresh_linear;
    tracker->threshsq = tracker->thresh_linear_sq * max_magsq;

    float freq_step = (float)sr / n_fft;
    float two_pi_ts = TWO_PI / sr;
    tracker->inv_hop = 1.0f / hop;
    tracker->freq_step_omega = freq_step * two_pi_ts;
    tracker->freq_step_df = 0.5f * freq_step * tracker->inv_hop * two_pi_ts;
    tracker->hop_float = (float)hop;

    tracker->chains = (SegBlockChain*)malloc(n_threads * sizeof(SegBlockChain));
    if (!tracker->chains) {
        free(tracker);
        return NULL;
    }
    for (int t = 0; t < n_threads; t++) {
        seg_chain_init(&tracker->chains[t]);
    }

    return tracker;
}

void spectral_tracker_update_threshold(SpectralTracker* tracker, float new_max_magsq) {
    if (!tracker) return;
    tracker->threshsq = tracker->thresh_linear_sq * new_max_magsq;
}

void spectral_tracker_process(SpectralTracker* tracker,
                               const float* chunk_magsq, const float* chunk_phases,
                               size_t chunk_n_frames, size_t global_frame_offset,
                               const float* overlap_magsq_row) {
    if (!tracker || chunk_n_frames < 1) return;
    int af;
    #pragma omp atomic read
    af = tracker->alloc_failed;
    if (af) return;

    const size_t n_freqs = tracker->n_freqs;
    const float threshsq = tracker->threshsq;
    const float freq_step_omega = tracker->freq_step_omega;
    const float freq_step_df = tracker->freq_step_df;
    const float inv_hop = tracker->inv_hop;
    const float hop_float = tracker->hop_float;
    /* Number of frame-pairs we can process:
     * - All internal pairs within this chunk (chunk_n_frames - 1)
     * - Plus one extra pair if overlap_magsq_row is provided
     *   (last chunk frame paired with first frame of next chunk) */
    size_t n_pairs = chunk_n_frames - 1;
    if (overlap_magsq_row) n_pairs = chunk_n_frames;

    if (n_pairs == 0) return;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        SegBlockChain* chain = &tracker->chains[tid];

        #pragma omp for schedule(guided, 256) nowait
        for (size_t t = 0; t < n_pairs; t++) {
            const float* __restrict__ phase_row = chunk_phases + t * n_freqs;
            const float* __restrict__ row = chunk_magsq + t * n_freqs;
            const float* __restrict__ next_row;

            if (t + 1 < chunk_n_frames) {
                next_row = chunk_magsq + (t + 1) * n_freqs;
            } else {
                /* Last frame: use overlap row from next chunk */
                next_row = overlap_magsq_row;
            }

            const float t_hop = (global_frame_offset + t) * hop_float;

            for (size_t f = 1; f < n_freqs - 1; f++) {
                float curr = row[f];

                if (curr <= threshsq || curr <= row[f-1] || curr <= row[f+1]) {
                    continue;
                }

                float m0 = next_row[f-1];
                float m1 = next_row[f];
                float m2 = next_row[f+1];

                float max_vsq = (m0 > m1) ? m0 : m1;
                max_vsq = (max_vsq > m2) ? max_vsq : m2;

                if (max_vsq < threshsq) {
                    continue;
                }

                Segment* seg = seg_chain_push(chain);
                if (SPECTRAL_UNLIKELY(!seg)) {
                    #pragma omp atomic write
                    tracker->alloc_failed = 1;
                    break;
                }

                /* TODO: For higher-quality partial tracking, consider McAulay-Quatieri
                 * (1986) "Speech Analysis/Synthesis Based on a Sinusoidal Representation"
                 * — nearest-frequency matching with birth/death cost model. Current greedy
                 * frame-pair approach is adequate for additive resynthesis with short hops. */
                int best_idx = (m0 >= m1) ? 0 : 1;
                best_idx = (m2 > ((best_idx == 0) ? m0 : m1)) ? 2 : best_idx;
                int best_next = (int)f + best_idx - 1;

                float m = sqrtf(curr);
                float max_v = sqrtf(max_vsq);

                seg->start = t_hop;
                seg->length = hop_float;
                seg->phase = phase_row[f];

                /* Parabolic interpolation for sub-bin frequency accuracy (Smith & Serra, 1987) */
                float log_l = log10f(row[f-1] + 1e-30f);
                float log_c = log10f(curr + 1e-30f);
                float log_r = log10f(row[f+1] + 1e-30f);
                float denom = log_l - 2.0f * log_c + log_r;
                float p = (fabsf(denom) > 1e-10f) ? 0.5f * (log_l - log_r) / denom : 0.0f;
                seg->omega = ((float)f + p) * freq_step_omega;
                seg->df = (best_next - (int)f) * freq_step_df;
                seg->amp = m;
                seg->da = (max_v - m) * inv_hop;
                seg->width = TRACK_DEFAULT_WIDTH;
            }
        }
    }
}

SegmentArray spectral_tracker_finalize(SpectralTracker* tracker, double* t_track) {
    SegmentArray empty_result = SEGMENT_ARRAY_EMPTY;

    if (!tracker) {
        if (t_track) *t_track = 0;
        return empty_result;
    }

    int n_threads = tracker->n_threads;
    SegBlockChain* chains = tracker->chains;

    if (tracker->alloc_failed) {
        for (int t = 0; t < n_threads; t++) seg_chain_free(&chains[t]);
        free(chains);
        if (t_track) *t_track = 0;
        free(tracker);
        return empty_result;
    }

    /* Compute total and per-thread offsets */
    size_t total_segs = 0;
    size_t* offsets = (size_t*)malloc(n_threads * sizeof(size_t));
    if (!offsets) {
        for (int t = 0; t < n_threads; t++) seg_chain_free(&chains[t]);
        free(chains);
        if (t_track) *t_track = 0;
        free(tracker);
        return empty_result;
    }
    for (int t = 0; t < n_threads; t++) {
        offsets[t] = total_segs;
        total_segs += chains[t].total_count;
    }

    /* Merge into contiguous array */
    Segment* segs = (total_segs > 0) ? (Segment*)malloc(total_segs * sizeof(Segment)) : NULL;
    if (total_segs > 0 && !segs) {
        for (int t = 0; t < n_threads; t++) seg_chain_free(&chains[t]);
        free(chains);
        free(offsets);
        if (t_track) *t_track = 0;
        free(tracker);
        return empty_result;
    }

    /* Pre-fault pages in parallel for large merge allocations (>64MB) */
    size_t merge_bytes = total_segs * sizeof(Segment);
    if (segs && merge_bytes > PRETOUCH_THRESHOLD) {
        size_t page_size = PRETOUCH_PAGE_SIZE;
        size_t n_pages = (merge_bytes + page_size - 1) / page_size;
        #pragma omp parallel for schedule(static)
        for (size_t p = 0; p < n_pages; p++) {
            ((volatile char*)segs)[p * page_size] = 0;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_threads; t++) {
        seg_chain_copy_to(&chains[t], segs + offsets[t]);
        seg_chain_free(&chains[t]);
    }
    free(chains);
    free(offsets);

    if (t_track) *t_track = omp_get_wtime() - tracker->start_time;

    SegmentArray res;
    res.segs = segs;
    res.count = (uint32_t)total_segs;
    res.capacity = (uint32_t)total_segs;

    free(tracker);
    return res;
}

/* spectral_track_peaks — single-shot wrapper around SpectralTracker */

SegmentArray spectral_track_peaks(const float* magsq, const float* phases,
                                  float max_magsq,
                                  size_t n_frames, size_t n_freqs,
                                  int sr, int n_fft, int hop,
                                  float db_thresh, double* t_track) {
    SegmentArray empty_result = SEGMENT_ARRAY_EMPTY;

    if (!magsq || !phases || !t_track || n_frames < 2 || n_freqs < 3) {
        if (t_track) *t_track = 0;
        return empty_result;
    }

    int n_threads = omp_get_max_threads();

    SpectralTracker* tracker = spectral_tracker_create(
        n_threads, n_freqs, sr, n_fft, hop, db_thresh, max_magsq);
    if (!tracker) {
        *t_track = 0;
        return empty_result;
    }

    /* Process all frames in one shot — no overlap row (NULL for final chunk) */
    spectral_tracker_process(tracker, magsq, phases,
                              n_frames, 0, NULL);

    return spectral_tracker_finalize(tracker, t_track);
}
