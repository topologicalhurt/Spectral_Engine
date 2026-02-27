/* spectral_peak_interp.c - Sub-bin Frequency Interpolation */
#include "spectral_peak_interp.h"
#include "spectral_log.h"
#include "spectral_windows.h"
#include "spectral_omp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __AVX2__
#include "simde/x86/avx2.h"
#endif

#if SPECTRAL_CUSTOM_FAST_MATH_MODE
#include "spectral_fast_math.h"
#define SPECTRAL_LOCAL_SQRT(x) fast_sqrt(x)
#else
#define SPECTRAL_LOCAL_SQRT(x) sqrtf(x)
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
    const float m0 = next_row[cf - 1];
    const float m1 = next_row[cf];
    const float m2 = next_row[cf + 1];
    float max_vsq = (m0 > m1) ? m0 : m1;
    int best_idx = 0;
    const float curr = row[cf];

    max_vsq = (max_vsq > m2) ? max_vsq : m2;
    if (max_vsq < threshsq) return 0;

    best_idx = (m0 >= m1) ? 0 : 1;
    best_idx = (m2 > ((best_idx == 0) ? m0 : m1)) ? 2 : best_idx;
    *out_curr = curr;
    *out_max_vsq = max_vsq;
    *out_best_next = (int)cf + best_idx - 1;
    return 1;
}

int spectral_tracker_emit_segment(
    SpectralTracker* tracker,
    int tid,
    const float* __restrict__ row,
    const float* __restrict__ phase_row,
    size_t cf,
    float t_hop,
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
    float m = 0.0f;
    float da = 0.0f;
    float max_v = 0.0f;
    float left = 0.0f;
    float right = 0.0f;
    float p = 0.0f;

    size_t count = tracker->seg_counts[tid * SPECTRAL_CACHE_LINE_STRIDE];
    if (SPECTRAL_UNLIKELY(count >= tracker->seg_capacities[tid * SPECTRAL_CACHE_LINE_STRIDE])) {
        size_t new_cap = tracker->seg_capacities[tid * SPECTRAL_CACHE_LINE_STRIDE] * 2;
        SPECTRAL_LOG_WARN("Track segment realloc: tid=%d cap=%zu->%zu (unexpected)", tid, count, new_cap);
        TrackSegment* new_arr = (TrackSegment*)spectral_aligned_alloc(new_cap * sizeof(TrackSegment));
        if (!new_arr) {
            atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_MEMORY, memory_order_relaxed);
            return 0;
        }
        memcpy(new_arr, tracker->seg_arrays[tid], count * sizeof(TrackSegment));
        free(tracker->seg_arrays[tid]);
        tracker->seg_arrays[tid] = new_arr;
        tracker->seg_capacities[tid * SPECTRAL_CACHE_LINE_STRIDE] = new_cap;
    }

    TrackSegment* seg = &tracker->seg_arrays[tid][count];

#if SPECTRAL_TRACK_DEBUG_TIMING
    double phase_start = omp_get_wtime();
#endif

    seg->start = t_hop;
    seg->length = hop_float;
    seg->phase = phase_row[cf];
    seg->width = SPECTRAL_TRACK_DEFAULT_WIDTH;
#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_emit_alloc_time += omp_get_wtime() - phase_start;
    phase_start = omp_get_wtime();
#endif

    m = SPECTRAL_LOCAL_SQRT(curr);
    max_v = SPECTRAL_LOCAL_SQRT(max_vsq);
    da = (max_v - m) * inv_hop;
    seg->amp = m;
    seg->da = da;
#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_emit_amp_time += omp_get_wtime() - phase_start;
    phase_start = omp_get_wtime();
#endif

    /* Zero-cost rational approximation for exact sub-bin frequency on power spectrum */
    left = row[cf - 1];
    right = row[cf + 1];
    p = spectral_window_interp_magsq_parabolic(left, curr, right);
    
    seg->omega = ((float)cf + p) * freq_step_omega;
    seg->df = (best_next - (int)cf) * freq_step_df;
#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_emit_interp_time += omp_get_wtime() - phase_start;
#endif

    tracker->seg_counts[tid * SPECTRAL_CACHE_LINE_STRIDE] = count + 1;
    (*local_segments)++;
    return 1;
}
