/* spectral_peak_track_internal.h
 *
 * Internal tracker-only declarations kept out of spectral_peak_track.c
 * to keep the implementation file focused on algorithm flow.
 */
#ifndef SPECTRAL_PEAK_TRACK_INTERNAL_H
#define SPECTRAL_PEAK_TRACK_INTERNAL_H

#include "spectral_peak_track.h"
#include "spectral_windows.h"

#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include "spectral_error.h"
#include <sys/mman.h>
#include <unistd.h>   /* sysconf(_SC_PAGESIZE) for the merge pre-touch path */

/* MADV_POPULATE_WRITE is Linux 5.14+. Keep a local fallback constant to
 * preserve compatibility with older libc headers. */
#if defined(__linux__)
  #if defined(MADV_POPULATE_WRITE)
    #define SPECTRAL_TRACK_MADV_POPULATE_WRITE MADV_POPULATE_WRITE
  #else
    /* Linux UAPI value from asm-generic/mman-common.h; used only when libc
     * headers predate the macro. */
    #define SPECTRAL_TRACK_MADV_POPULATE_WRITE 23
  #endif
#endif

typedef struct __attribute__((aligned(16))) {
    float start;
    float length;
    float phase;
    float omega;
    float df;
    float amp;
    float da;
    float width;
} TrackSegment;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(TrackSegment) == 32, "TrackSegment size");
_Static_assert(SPECTRAL_TRACK_CANDIDATE_BATCH > 0u, "SPECTRAL_TRACK_CANDIDATE_BATCH must be > 0");
_Static_assert(SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE > 0u, "SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE must be > 0");
_Static_assert((SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE &
                (SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE - 1u)) == 0u,
               "SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE must be power-of-two");
_Static_assert(SPECTRAL_TRACK_PAIR_OMP_CHUNK <= (unsigned)INT_MAX,
               "SPECTRAL_TRACK_PAIR_OMP_CHUNK must fit OpenMP int chunk range");
#endif

struct SpectralTracker {
    TrackSegment** seg_arrays;
    size_t* seg_counts;
    size_t* seg_capacities;
    /* Per-thread phase scratch (n_freqs floats each). Phase-at-peaks fills only
     * the tracked peak bins via atan2f(im,re) here, then hands these to
     * emit_segment exactly like the old dense phase rows — so the estimator and
     * its tests stay byte-identical while the all-bins producer atan2 is gone. */
    float** thread_phase_scratch;
    float** thread_next_phase_scratch;
    int n_threads;
    _Atomic SpectralError last_error;

    /* Precomputed constants */
    size_t n_freqs;
    float threshsq;
    float thresh_linear_sq;  /* db-derived factor, independent of max_magsq */
    float freq_step_omega;
    float freq_step_df;
    float inv_hop;
    float hop_float;
    float peak_candan_correction;
    SpectralWindowInterpMagsqFn interp_magsq;
    SpectralWindowPeakMagsqFn peak_magsq;
    SpectralPeakEstimatorType peak_estimator;
    SpectralPeakPhasePolicy phase_policy;
    SpectralPeakAmplitudePolicy amplitude_policy;
    SpectralResolvedPeakModel peak_model;

    double process_time_total;
    uint64_t total_pairs;
    uint64_t total_candidates;
    uint64_t total_segments;
#if SPECTRAL_TRACK_DEBUG_TIMING
    double debug_scan_time_total;
    double debug_validate_time_total;
    double debug_emit_time_total;
    double debug_emit_alloc_time_total;
    double debug_emit_interp_time_total;
    double debug_emit_amp_time_total;
#endif
};

typedef struct SpectralTrackerCandidateBatch {
    uint32_t ids[SPECTRAL_TRACK_CANDIDATE_BATCH];
    size_t count;
} SpectralTrackerCandidateBatch;

static inline void spectral_tracker_candidate_batch_reset(SpectralTrackerCandidateBatch* batch)
{
    if (batch) batch->count = 0u;
}

static inline SpectralError spectral_tracker_frame_time_from_index(size_t frame_index,
                                                                  float hop_float,
                                                                  float* out_t_hop)
{
    double t_hop_d = 0.0;

    if (!out_t_hop || !isfinite(hop_float) || hop_float <= 0.0f) {
        return SPECTRAL_ERR_PARAM;
    }

    t_hop_d = (double)frame_index * (double)hop_float;
    if (!isfinite(t_hop_d) || t_hop_d < 0.0 || t_hop_d > (double)FLT_MAX) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    *out_t_hop = (float)t_hop_d;
    return SPECTRAL_OK;
}

static inline SpectralError spectral_tracker_frame_context_init(
    SpectralFrameContext* ctx,
    const float* row,
    const float* next_row,
    const SpectralHalf* re_row,
    const SpectralHalf* im_row,
    const SpectralHalf* next_re_row,
    const SpectralHalf* next_im_row,
    size_t frame_index,
    float hop_float,
    float threshsq,
    int can_start_new)
{
    float frame_start_sample = 0.0f;
    SpectralError err = SPECTRAL_OK;

    if (!ctx || !row || !next_row || !re_row || !im_row) {
        return SPECTRAL_ERR_PARAM;
    }

    err = spectral_tracker_frame_time_from_index(frame_index, hop_float, &frame_start_sample);
    if (err != SPECTRAL_OK) return err;

    ctx->row = row;
    ctx->next_row = next_row;
    ctx->re_row = re_row;
    ctx->im_row = im_row;
    ctx->next_re_row = next_re_row;
    ctx->next_im_row = next_im_row;
    ctx->frame_start_sample = frame_start_sample;
    ctx->threshsq = threshsq;
    ctx->can_start_new = can_start_new;
    return SPECTRAL_OK;
}

void spectral_tracker_set_error(SpectralTracker* tracker, SpectralError error);


typedef struct SpectralTrackerWorkerStats {
    uint64_t pairs;
    uint64_t candidates;
    uint64_t segments;
    double track_time;
#if SPECTRAL_TRACK_DEBUG_TIMING
    double scan_time;
    double validate_time;
    double emit_time;
    double emit_alloc_time;
    double emit_interp_time;
    double emit_amp_time;
#endif
} SpectralTrackerWorkerStats;

static inline void spectral_tracker_worker_stats_commit(SpectralTracker* tracker,
                                                        const SpectralTrackerWorkerStats* stats)
{
    if (!tracker || !stats) return;
    spectral_tracker_accumulate_stats(tracker,
                                      stats->pairs,
                                      stats->candidates,
                                      stats->segments,
                                      stats->track_time
#if SPECTRAL_TRACK_DEBUG_TIMING
                                      , stats->scan_time,
                                      stats->validate_time,
                                      stats->emit_time,
                                      stats->emit_alloc_time,
                                      stats->emit_interp_time,
                                      stats->emit_amp_time
#endif
                                      );
}

#endif /* SPECTRAL_PEAK_TRACK_INTERNAL_H */
