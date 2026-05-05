/* spectral_peak_track.h - Peak Tracking from STFT Magnitude/Phase Data
 *
 * Extracts spectral peaks above a threshold from magnitude-squared and phase
 * matrices, creating Segment objects for resynthesis. Multi-threaded via OpenMP.
 *
 * Uses bounded per-thread segment buffers that grow on demand.
 *
 * Two modes of operation:
 *   1. spectral_track_peaks() — single-shot, processes entire STFT at once
 *   2. SpectralTracker API    — incremental, processes chunks for large datasets
 */
#ifndef SPECTRAL_PEAK_TRACK_H
#define SPECTRAL_PEAK_TRACK_H

#include "spectral_common.h"



#ifdef __cplusplus
extern "C" {
#endif

/* Single-shot API: processes entire STFT matrices at once */
SegmentArray spectral_track_peaks(const float* magsq, const float* phases,
                                  float max_magsq,
                                  size_t n_frames, size_t n_freqs,
                                  int sr, int n_fft, int hop,
                                  float db_thresh, double* t_track);

/* Incremental tracker API for chunked STFT processing */
typedef struct SpectralTracker SpectralTracker;

/* Per-frame context for the fused STFT+track path.
 * Groups the per-frame arguments that change on every call. */
typedef struct {
    const float* __restrict__ row;
    const float* __restrict__ next_row;
    const float* __restrict__ phase_row;
    float t_hop;
    float threshsq;
    int can_start_new;
} SpectralFrameContext;

/* Fully fused STFT track scan: given direct arrays, vector-scan and queue candidates */
int spectral_tracker_run_fused_frame(
    SpectralTracker* tracker,
    int tid,
    uint32_t* __restrict__ candidate_batch,
    size_t* candidate_batch_count,
    const SpectralFrameContext* ctx,
    uint64_t* local_candidates,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* pair_validate_time
    , double* pair_emit_time
    , double* pair_emit_alloc_time
    , double* pair_emit_interp_time
    , double* pair_emit_amp_time
#endif
);

SpectralTracker* spectral_tracker_create(int n_threads, size_t n_freqs,
                                          int sr, int n_fft, int hop,
                                          float db_thresh, float max_magsq);

/* Update threshold when a new global max is discovered across chunks.
 * Call before spectral_tracker_process() for each chunk with an updated max. */
void spectral_tracker_update_threshold(SpectralTracker* tracker, float new_max_magsq);

/* Process one chunk of STFT data.
 * overlap_magsq_row: pointer to the first magsq row of the NEXT chunk
 *                    (n_freqs floats) for look-ahead on the last frame pair.
 *                    Pass NULL for the final chunk (skips last frame pair). */
void spectral_tracker_process(SpectralTracker* tracker,
                               const float* chunk_magsq, const float* chunk_phases,
                               size_t chunk_n_frames, size_t global_frame_offset,
                               const float* overlap_magsq_row);

/* Finalize tracking: merges all per-thread segment arrays into a contiguous
 * SegmentArray. Frees internal arrays but does NOT free the tracker itself.
 * Call spectral_tracker_destroy() after this to release the tracker.
 * Writes elapsed tracking time to *t_track. */
SegmentArray spectral_tracker_finalize(SpectralTracker* tracker, double* t_track, double override_wall_time);

/* Free the tracker struct. Safe to call after spectral_tracker_finalize(),
 * or standalone to abandon tracking (will clean up any remaining internal state). */
void spectral_tracker_destroy(SpectralTracker* tracker);

/* Free the segment memory owned by a SegmentArray and zero the struct.
 * Correctly handles both malloc and posix_memalign allocations. */
void spectral_segment_array_free(SegmentArray* arr);

/* --- Encapsulated accessors (avoids exposing SpectralTracker internals) --- */

/* Returns non-zero if the tracker has entered an error state (e.g. allocation failure). */
int  spectral_tracker_has_failed(SpectralTracker* tracker);

/* Signal an allocation failure from outside the tracker module. */
void spectral_tracker_set_failed(SpectralTracker* tracker);

/* Read the precomputed power threshold (threshsq). */
float spectral_tracker_get_threshsq(const SpectralTracker* tracker);

/* Read accumulated wall-clock processing time. */
double spectral_tracker_get_process_time(const SpectralTracker* tracker);

/* Atomically accumulate per-thread statistics into the tracker.
 * Must be called from within an OpenMP parallel region. */
void spectral_tracker_accumulate_stats(
    SpectralTracker* tracker,
    uint64_t local_pairs, uint64_t local_candidates, uint64_t local_segments,
    double local_track_time
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double local_scan_time, double local_validate_time, double local_emit_time,
    double local_emit_alloc_time, double local_emit_interp_time,
    double local_emit_amp_time
#endif
);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PEAK_TRACK_H */
