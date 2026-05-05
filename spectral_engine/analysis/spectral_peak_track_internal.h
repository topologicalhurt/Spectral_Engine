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
#include <stdatomic.h>
#include "spectral_error.h"
#include <sys/mman.h>

/* Removed force-define */

/* MADV_POPULATE_WRITE is Linux 5.14+. Keep a local fallback constant to
 * preserve compatibility with older libc headers. */
#if defined(__linux__)
  #if defined(MADV_POPULATE_WRITE)
    #define SPECTRAL_TRACK_MADV_POPULATE_WRITE MADV_POPULATE_WRITE
  #else
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
    SpectralWindowInterpMagsqFn interp_magsq;

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

int spectral_tracker_flush_candidate_batch(
    SpectralTracker* tracker,
    int tid,
    uint32_t* __restrict__ candidate_batch,
    size_t* candidate_batch_count,
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    const float* __restrict__ phase_row,
    float t_hop,
    float threshsq,
    float freq_step_omega,
    float freq_step_df,
    float inv_hop,
    float hop_float,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* pair_validate_time
    , double* pair_emit_time
    , double* pair_emit_alloc_time
    , double* pair_emit_interp_time
    , double* pair_emit_amp_time
#endif
);

#endif /* SPECTRAL_PEAK_TRACK_INTERNAL_H */
