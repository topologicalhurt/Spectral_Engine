/* spectral_peak_interp.c - Sub-bin Frequency Interpolation */
#include "spectral_peak_interp.h"
#include "spectral_peak_estimator.h"
#include "spectral_log.h"
#include "spectral_utils.h"
#include "spectral_windows.h"
#include "spectral_omp.h"
#if !SPECTRAL_NO_PERF
#include "spectral_perf.h"  /* perf_track_realloc — count the no-realloc-contract violation */
#endif
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef __AVX2__
#include "simde/x86/avx2.h"
#endif

int spectral_tracker_validate_candidate(
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    size_t cf,
    float threshsq,
    float* out_curr,
    float* out_max_vsq,
    int* out_best_next)
{
    float left = 0.0f;
    float curr = 0.0f;
    float right = 0.0f;
    float m0 = 0.0f;
    float m1 = 0.0f;
    float m2 = 0.0f;
    float max_vsq = 0.0f;
    int best_idx = 0;

    if (!row || !next_row || !out_curr || !out_max_vsq || !out_best_next ||
        cf == 0u || cf > (size_t)INT_MAX ||
        !SPECTRAL_ISFINITE(threshsq) || threshsq < 0.0f) {
        return 0;
    }

    left = row[cf - 1u];
    curr = row[cf];
    right = row[cf + 1u];
    m0 = next_row[cf - 1u];
    m1 = next_row[cf];
    m2 = next_row[cf + 1u];
    max_vsq = (m0 > m1) ? m0 : m1;

    if (!SPECTRAL_ISFINITE(left) || !SPECTRAL_ISFINITE(curr) || !SPECTRAL_ISFINITE(right) ||
        !SPECTRAL_ISFINITE(m0) || !SPECTRAL_ISFINITE(m1) || !SPECTRAL_ISFINITE(m2) ||
        left < 0.0f || curr < 0.0f || right < 0.0f ||
        m0 < 0.0f || m1 < 0.0f || m2 < 0.0f) {
        return 0;
    }

    max_vsq = (max_vsq > m2) ? max_vsq : m2;
    if (!SPECTRAL_ISFINITE(max_vsq) || max_vsq < threshsq) return 0;

    best_idx = (m0 >= m1) ? 0 : 1;
    best_idx = (m2 > ((best_idx == 0) ? m0 : m1)) ? 2 : best_idx;
    *out_curr = curr;
    *out_max_vsq = max_vsq;
    *out_best_next = (int)cf + best_idx - 1;
    return 1;
}

static int spectral_tracker_emitted_segment_valid(const SpectralTracker* tracker,
                                                  int tid,
                                                  float frame_start_sample,
                                                  float hop_float,
                                                  float phase,
                                                  const SpectralPeakEstimate* estimate)
{
    if (!tracker || !estimate || tid < 0 || tid >= tracker->n_threads) {
        return 0;
    }
    if (!tracker->seg_arrays || !tracker->seg_counts || !tracker->seg_capacities) {
        return 0;
    }
    if (!SPECTRAL_ISFINITE(frame_start_sample) || frame_start_sample < 0.0f ||
        !SPECTRAL_ISFINITE(hop_float) || hop_float <= 0.0f ||
        !SPECTRAL_ISFINITE(phase) ||
        !SPECTRAL_ISFINITE(SPECTRAL_TRACK_DEFAULT_WIDTH) || SPECTRAL_TRACK_DEFAULT_WIDTH <= 0.0f) {
        return 0;
    }
    if (!SPECTRAL_ISFINITE(estimate->amp) || estimate->amp < 0.0f ||
        !SPECTRAL_ISFINITE(estimate->da) ||
        !SPECTRAL_ISFINITE(estimate->omega) || estimate->omega < 0.0f ||
        !SPECTRAL_ISFINITE(estimate->df)) {
        return 0;
    }
    return 1;
}


int spectral_tracker_emit_segment(
    SpectralTracker* tracker,
    int tid,
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    size_t cf,
    float frame_start_sample,
    float freq_step_omega,
    float freq_step_df,
    float inv_hop,
    float hop_float,
    float curr,
    float max_vsq,
    int best_next,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_emit_alloc_time
    , double* local_emit_interp_time
    , double* local_emit_amp_time
#endif
)
{
    SpectralPeakEstimateInput estimate_input = {0};
    SpectralPeakEstimate estimate = {0};
    size_t count = 0;
    TrackSegment* seg = NULL;

#if SPECTRAL_TRACK_DEBUG_TIMING
    double phase_start = omp_get_wtime();
#endif

    if (!tracker || !row || !phase_row || !local_segments || cf == 0u || cf + 1u >= tracker->n_freqs) {
        return 1;
    }

    estimate_input.magsq_row = row;
    estimate_input.phase_row = phase_row;
    estimate_input.next_magsq_row = next_row;
    estimate_input.next_phase_row = next_phase_row;
    estimate_input.n_freqs = tracker->n_freqs;
    estimate_input.bin = cf;
    estimate_input.curr_magsq = curr;
    estimate_input.next_max_magsq = max_vsq;
    estimate_input.best_next_bin = best_next;
    estimate_input.freq_step_omega = freq_step_omega;
    estimate_input.freq_step_df = freq_step_df;
    estimate_input.inv_hop = inv_hop;
    estimate_input.hop_float = hop_float;
    estimate_input.candan_correction = tracker->peak_candan_correction;
    estimate_input.interp_magsq = tracker->interp_magsq;
    estimate_input.peak_magsq = tracker->peak_magsq;
    estimate_input.type = tracker->peak_estimator;
    estimate_input.phase_policy = tracker->phase_policy;
    estimate_input.amplitude_policy = tracker->amplitude_policy;

    if (!spectral_peak_estimate_validated(&estimate_input, &estimate)) {
#if SPECTRAL_TRACK_DEBUG_TIMING
        *local_emit_interp_time += omp_get_wtime() - phase_start;
        *local_emit_amp_time += 0.0;
#endif
        return 1;
    }

#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_emit_interp_time += omp_get_wtime() - phase_start;
    phase_start = omp_get_wtime();
#endif

    if (!spectral_tracker_emitted_segment_valid(tracker, tid, frame_start_sample, hop_float, phase_row[cf], &estimate)) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);
        return 0;
    }

    count = tracker->seg_counts[tid * SPECTRAL_CACHE_LINE_STRIDE];
    if (SPECTRAL_UNLIKELY(count >= tracker->seg_capacities[tid * SPECTRAL_CACHE_LINE_STRIDE])) {
        size_t old_cap = tracker->seg_capacities[tid * SPECTRAL_CACHE_LINE_STRIDE];
        size_t new_cap = old_cap * 2u;
        size_t new_bytes = 0;
        size_t copy_bytes = 0;
        TrackSegment* new_arr = NULL;

        if (old_cap == 0u || new_cap < old_cap ||
            !spectral_size_mul(new_cap, sizeof(TrackSegment), &new_bytes) ||
            !spectral_size_mul(count, sizeof(TrackSegment), &copy_bytes)) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
            return 0;
        }

        SPECTRAL_LOG_WARN("Track segment realloc: tid=%d cap=%zu->%zu (unexpected)", tid, count, new_cap);
#if !SPECTRAL_NO_PERF
        perf_track_realloc();  /* surface the contract violation as a counted metric */
#endif
        new_arr = (TrackSegment*)spectral_aligned_alloc(new_bytes);
        if (!new_arr) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);
            return 0;
        }
        memcpy(new_arr, tracker->seg_arrays[tid], copy_bytes);
        free(tracker->seg_arrays[tid]);
        tracker->seg_arrays[tid] = new_arr;
        tracker->seg_capacities[tid * SPECTRAL_CACHE_LINE_STRIDE] = new_cap;
    }

    seg = &tracker->seg_arrays[tid][count];

    seg->start = frame_start_sample;
    seg->length = hop_float;
    seg->phase = phase_row[cf];
    seg->width = SPECTRAL_TRACK_DEFAULT_WIDTH;
    seg->amp = estimate.amp;
    seg->da = estimate.da;
    seg->omega = estimate.omega;
    seg->df = estimate.df;

#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_emit_alloc_time += omp_get_wtime() - phase_start;
    *local_emit_amp_time += 0.0;
#endif

    tracker->seg_counts[tid * SPECTRAL_CACHE_LINE_STRIDE] = count + 1;
    (*local_segments)++;
    return 1;
}
