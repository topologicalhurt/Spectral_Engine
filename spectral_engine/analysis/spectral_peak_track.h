/* spectral_peak_track.h - Peak Tracking from STFT Magnitude/Phase Data
 *
 * Extracts spectral peaks above a threshold from magnitude-squared and phase
 * matrices, creating Segment objects for resynthesis. Multi-threaded via OpenMP.
 *
 * Uses a block-chain allocator to avoid realloc in hot loops.
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

/* Finalize tracking: merges all per-thread block chains into a contiguous
 * SegmentArray. Frees the tracker. Writes elapsed tracking time to *t_track. */
SegmentArray spectral_tracker_finalize(SpectralTracker* tracker, double* t_track);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PEAK_TRACK_H */
