/* spectral_peak_track.c - Peak Tracking 
 *
 * STRATEGY:
 * The Peak Tracking phase is the most mathematically dense stage of the engine.
 * To achieve multi-GiB/s bandwidth, it avoids scalar branching wherever possible.
 *
 * 1. Filtering: We use AVX2/SSE SIMD intrinsics to scan up to 8 bins per cycle.
 *    Instead of evaluating `sqrtf` on everything, we compare raw power (magnitude
 *    squared) against `threshsq`. The result is a hardware bitmask of valid peaks.
 * 2. Queueing: Rather than immediately allocating memory and calculating exact
 *    sub-bin interpolations per peak, we push raw candidate array indices into a
 *    small size-128 local execution batch (`candidate_batch`).
 * 3. Batch Emission: Once the batch fills, we loop over it. Because the STFT memory
 *    is accessed strictly sequentially by candidate index, this perfectly triggers
 *    the CPU's L1/L2 hardware prefetcher, completely hiding DRAM load latency.
 *    Only verified local-maximas are passed into the ALU-heavy `spectral_tracker_emit_segment`.
 *
 * Two modes:
 *   1. spectral_track_peaks()  — single-shot (processes entire STFT)
 *   2. SpectralTracker API     — incremental (processes chunks)
 */

#include "spectral_peak_track_internal.h"
#include "spectral_log.h"
#include "spectral_utils.h"
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_omp.h"
#include "spectral_peak_interp.h"
#include "spectral_windows.h"

/* SIMDe for branchless peak pre-scan */
#include "simde/x86/sse2.h"
#ifdef __AVX2__
#include "simde/x86/avx2.h"
#endif

static SegmentArray peak_track_return_empty(double* t_track) {
    if (t_track) *t_track = 0;
    return (SegmentArray)SEGMENT_ARRAY_EMPTY;
}

static SPECTRAL_FORCEINLINE int spectral_tracker_process_bitmask(
    SpectralTracker* tracker, int tid,
    uint32_t* __restrict__ candidate_batch, size_t* candidate_batch_count,
    const float* __restrict__ row, const float* __restrict__ next_row, const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    size_t f, int bits, float t_hop, float threshsq,
    float freq_step_omega, float freq_step_df, float inv_hop, float hop_float,
    uint64_t* local_candidates, uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_validate_time, double* local_emit_time, double* local_emit_alloc_time,
    double* local_emit_interp_time, double* local_emit_amp_time
#endif
);



/* ── Encapsulated accessors (items 1.2, 2.5) ──────────────────────── */

int spectral_tracker_has_failed(SpectralTracker* tracker) {
    if (!tracker) return 1;
    return atomic_load_explicit(&tracker->last_error, memory_order_relaxed) != SPECTRAL_OK;
}

void spectral_tracker_set_error(SpectralTracker* tracker, SpectralError error) {
    SpectralError expected = SPECTRAL_OK;
    if (!tracker || error == SPECTRAL_OK) return;

    (void)atomic_compare_exchange_strong_explicit(
        &tracker->last_error,
        &expected,
        error,
        memory_order_relaxed,
        memory_order_relaxed);
}

void spectral_tracker_set_failed(SpectralTracker* tracker) {
    spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);
}


static SpectralPeakModel spectral_tracker_current_peak_model(const SpectralTracker* tracker) {
    if (!tracker || !tracker->peak_model.window) {
        return spectral_peak_model_default();
    }
    return spectral_peak_model_from_resolved(&tracker->peak_model);
}

static void spectral_tracker_cache_resolved_peak_model(SpectralTracker* tracker,
                                                       const SpectralResolvedPeakModel* resolved) {
    if (!tracker || !resolved) return;

    tracker->peak_model = *resolved;
    tracker->interp_magsq = resolved->interp_magsq;
    tracker->peak_magsq = resolved->peak_magsq;
    tracker->peak_estimator = resolved->estimator;
    tracker->phase_policy = resolved->phase_policy;
    tracker->amplitude_policy = resolved->amplitude_policy;
}

static SpectralError spectral_tracker_apply_peak_model(SpectralTracker* tracker,
                                                       const SpectralPeakModel* model) {
    SpectralResolvedPeakModel resolved;
    SpectralError err = SPECTRAL_OK;

    if (!tracker || !model) return SPECTRAL_ERR_PARAM;
    err = spectral_peak_model_resolve(model, &resolved);
    if (err != SPECTRAL_OK) return err;

    spectral_tracker_cache_resolved_peak_model(tracker, &resolved);
    return SPECTRAL_OK;
}

SpectralError spectral_tracker_set_peak_model(SpectralTracker* tracker, const SpectralPeakModel* model) {
    if (!tracker || !model) return SPECTRAL_ERR_PARAM;
    /* Apply only after the full model resolves. A failed mutation must not
     * silently replace an existing Hamming/Blackman/rectangular/custom profile
     * with the Hann compatibility default. */
    return spectral_tracker_apply_peak_model(tracker, model);
}

void spectral_tracker_set_window_descriptor(SpectralTracker* tracker, const SpectralWindowDescriptor* desc) {
    SpectralPeakModel model;
    if (!tracker) return;

    model = spectral_tracker_current_peak_model(tracker);
    model.window = desc ? desc : spectral_window_descriptor(SPECTRAL_WINDOW_HANN);
    model.amplitude_policy = spectral_peak_model_for_window(model.window).amplitude_policy;
    (void)spectral_tracker_set_peak_model(tracker, &model);
}

void spectral_tracker_set_peak_estimator(SpectralTracker* tracker, SpectralPeakEstimatorType type) {
    SpectralPeakModel model;
    if (!tracker) return;

    model = spectral_tracker_current_peak_model(tracker);
    model.estimator = type;
    (void)spectral_tracker_set_peak_model(tracker, &model);
}

void spectral_tracker_set_phase_policy(SpectralTracker* tracker, SpectralPeakPhasePolicy policy) {
    SpectralPeakModel model;
    if (!tracker) return;

    model = spectral_tracker_current_peak_model(tracker);
    model.phase_policy = policy;
    (void)spectral_tracker_set_peak_model(tracker, &model);
}

void spectral_tracker_set_amplitude_policy(SpectralTracker* tracker, SpectralPeakAmplitudePolicy policy) {
    SpectralPeakModel model;
    if (!tracker) return;

    model = spectral_tracker_current_peak_model(tracker);
    model.amplitude_policy = policy;
    (void)spectral_tracker_set_peak_model(tracker, &model);
}

float spectral_tracker_get_threshsq(const SpectralTracker* tracker) {
    return tracker ? tracker->threshsq : 0.0f;
}

double spectral_tracker_get_process_time(const SpectralTracker* tracker) {
    return tracker ? tracker->process_time_total : 0.0;
}

static int spectral_tracker_u64_add_checked(uint64_t a, uint64_t b, uint64_t* out)
{
    if (!out || b > UINT64_MAX - a) return 0;
    *out = a + b;
    return 1;
}

void spectral_tracker_accumulate_stats(
    SpectralTracker* tracker,
    uint64_t local_pairs, uint64_t local_candidates, uint64_t local_segments,
    double local_track_time
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double local_scan_time, double local_validate_time, double local_emit_time,
    double local_emit_alloc_time, double local_emit_interp_time, double local_emit_amp_time
#endif
) {
    if (!tracker) return;
    if (!isfinite(local_track_time) || local_track_time < 0.0) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);
        return;
    }
#if SPECTRAL_TRACK_DEBUG_TIMING
    if (!isfinite(local_scan_time) || local_scan_time < 0.0 ||
        !isfinite(local_validate_time) || local_validate_time < 0.0 ||
        !isfinite(local_emit_time) || local_emit_time < 0.0 ||
        !isfinite(local_emit_alloc_time) || local_emit_alloc_time < 0.0 ||
        !isfinite(local_emit_interp_time) || local_emit_interp_time < 0.0 ||
        !isfinite(local_emit_amp_time) || local_emit_amp_time < 0.0) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);
        return;
    }
#endif

    /* Stats are accumulated once per worker, not per bin.  Use a critical
     * section so overflow can be checked before mutating shared counters. */
    #pragma omp critical(spectral_tracker_stats_accum)
    {
        uint64_t next_u64 = 0;
        double next_time = 0.0;

        if (!spectral_tracker_u64_add_checked(tracker->total_pairs, local_pairs, &next_u64)) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->total_pairs = next_u64;
        }

        if (!spectral_tracker_u64_add_checked(tracker->total_candidates, local_candidates, &next_u64)) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->total_candidates = next_u64;
        }

        if (!spectral_tracker_u64_add_checked(tracker->total_segments, local_segments, &next_u64)) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->total_segments = next_u64;
        }

        next_time = tracker->process_time_total + local_track_time;
        if (!isfinite(next_time) || next_time < tracker->process_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->process_time_total = next_time;
        }

#if SPECTRAL_TRACK_DEBUG_TIMING
        next_time = tracker->debug_scan_time_total + local_scan_time;
        if (!isfinite(next_time) || next_time < tracker->debug_scan_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->debug_scan_time_total = next_time;
        }
        next_time = tracker->debug_validate_time_total + local_validate_time;
        if (!isfinite(next_time) || next_time < tracker->debug_validate_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->debug_validate_time_total = next_time;
        }
        next_time = tracker->debug_emit_time_total + local_emit_time;
        if (!isfinite(next_time) || next_time < tracker->debug_emit_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->debug_emit_time_total = next_time;
        }
        next_time = tracker->debug_emit_alloc_time_total + local_emit_alloc_time;
        if (!isfinite(next_time) || next_time < tracker->debug_emit_alloc_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->debug_emit_alloc_time_total = next_time;
        }
        next_time = tracker->debug_emit_interp_time_total + local_emit_interp_time;
        if (!isfinite(next_time) || next_time < tracker->debug_emit_interp_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->debug_emit_interp_time_total = next_time;
        }
        next_time = tracker->debug_emit_amp_time_total + local_emit_amp_time;
        if (!isfinite(next_time) || next_time < tracker->debug_emit_amp_time_total) {
            spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        } else {
            tracker->debug_emit_amp_time_total = next_time;
        }
#endif
    }
}


void spectral_segment_array_free(SegmentArray* arr) {
    if (!arr) return;
    free(arr->segs);
    arr->segs = NULL;
    arr->count = 0;
    arr->capacity = 0;
}

static void spectral_tracker_free_segment_storage(SpectralTracker* tracker) {
    if (!tracker) return;
    if (tracker->seg_arrays) {
        for (int t = 0; t < tracker->n_threads; t++) {
            free(tracker->seg_arrays[t]);
        }
        free(tracker->seg_arrays);
    }
    free(tracker->seg_counts);
    free(tracker->seg_capacities);
    tracker->seg_arrays = NULL;
    tracker->seg_counts = NULL;
    tracker->seg_capacities = NULL;
}

/* ── End accessors ─────────────────────────────────────────────────── */


static SPECTRAL_FORCEINLINE int spectral_tracker_handle_candidate(
    SpectralTracker* tracker,
    int tid,
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    size_t cf,
    float t_hop,
    float threshsq,
    float freq_step_omega,
    float freq_step_df,
    float inv_hop,
    float hop_float,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_validate_time
    , double* local_emit_time
    , double* local_emit_alloc_time
    , double* local_emit_interp_time
    , double* local_emit_amp_time
#endif
)
{
    float curr = 0.0f;
    float max_vsq = 0.0f;
    int best_next = 0;
    int validated = 0;
#if SPECTRAL_TRACK_DEBUG_TIMING
    double phase_start = omp_get_wtime();
#endif

    validated = spectral_tracker_validate_candidate(row, next_row, cf, threshsq,
                                                    &curr, &max_vsq, &best_next);
#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_validate_time += omp_get_wtime() - phase_start;
#endif
    if (!validated) return 1;

#if SPECTRAL_TRACK_DEBUG_TIMING
    phase_start = omp_get_wtime();
#endif
    if (!spectral_tracker_emit_segment(tracker, tid, row, next_row, phase_row, next_phase_row, cf, t_hop,
                                       freq_step_omega, freq_step_df,
                                       inv_hop, hop_float,
                                       curr, max_vsq, best_next,
                                       local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                                       , local_emit_alloc_time
                                       , local_emit_interp_time
                                       , local_emit_amp_time
#endif
                                       )) {
#if SPECTRAL_TRACK_DEBUG_TIMING
        *local_emit_time += omp_get_wtime() - phase_start;
#endif
        return 0;
    }
#if SPECTRAL_TRACK_DEBUG_TIMING
    *local_emit_time += omp_get_wtime() - phase_start;
#endif
    return 1;
}

static SPECTRAL_FORCEINLINE int spectral_tracker_process_candidate_batch(
    SpectralTracker* tracker,
    int tid,
    const uint32_t* __restrict__ candidates,
    size_t candidate_count,
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    float t_hop,
    float threshsq,
    float freq_step_omega,
    float freq_step_df,
    float inv_hop,
    float hop_float,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_validate_time
    , double* local_emit_time
    , double* local_emit_alloc_time
    , double* local_emit_interp_time
    , double* local_emit_amp_time
#endif
)
{
    for (size_t i = 0; i < candidate_count; i++) {
        /* Candidate order is monotonic in frequency, so lookahead prefetch
         * tends to pull contiguous cache lines for the validate/emit step. */
        if (i + SPECTRAL_TRACK_PREFETCH_LOOKAHEAD < candidate_count) {
            size_t pf = (size_t)candidates[i + SPECTRAL_TRACK_PREFETCH_LOOKAHEAD];
            SPECTRAL_PREFETCH_READ_LOCALITY(next_row + pf - 1u, SPECTRAL_TRACK_PREFETCH_READ_LOCALITY);
            SPECTRAL_PREFETCH_READ_LOCALITY(row + pf - 1u, SPECTRAL_TRACK_PREFETCH_READ_LOCALITY);
#if SPECTRAL_TRACK_PREFETCH_PHASE
            SPECTRAL_PREFETCH_READ_LOCALITY(phase_row + pf, SPECTRAL_TRACK_PREFETCH_READ_LOCALITY);
#endif
        }

        if (!spectral_tracker_handle_candidate(tracker, tid, row, next_row, phase_row, next_phase_row, (size_t)candidates[i],
                                               t_hop, threshsq,
                                               freq_step_omega, freq_step_df,
                                               inv_hop, hop_float,
                                               local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                                               , local_validate_time
                                               , local_emit_time
                                               , local_emit_alloc_time
                                               , local_emit_interp_time
                                               , local_emit_amp_time
#endif
                                               )) {
            return 0;
        }
    }
    return 1;
}

int spectral_tracker_flush_candidate_batch(
    SpectralTracker* tracker,
    int tid,
    uint32_t* __restrict__ candidate_batch,
    size_t* candidate_batch_count,
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    float t_hop,
    float threshsq,
    float freq_step_omega,
    float freq_step_df,
    float inv_hop,
    float hop_float,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_validate_time
    , double* local_emit_time
    , double* local_emit_alloc_time
    , double* local_emit_interp_time
    , double* local_emit_amp_time
#endif
)
{
    if (!tracker || tid < 0 || tid >= tracker->n_threads ||
        !candidate_batch_count || !local_segments) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);
        return 0;
    }
    if (*candidate_batch_count == 0) return 1;
    if (!candidate_batch || !row || !next_row || !phase_row) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);
        return 0;
    }
    if (*candidate_batch_count > SPECTRAL_TRACK_CANDIDATE_BATCH) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        *candidate_batch_count = 0;
        return 0;
    }

    if (!spectral_tracker_process_candidate_batch(
            tracker,
            tid,
            candidate_batch,
            *candidate_batch_count,
            row,
            next_row,
            phase_row,
            next_phase_row,
            t_hop,
            threshsq,
            freq_step_omega,
            freq_step_df,
            inv_hop,
            hop_float,
            local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
            , local_validate_time
            , local_emit_time
            , local_emit_alloc_time
            , local_emit_interp_time
            , local_emit_amp_time
#endif
            )) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_MEMORY);
        return 0;
    }
    *candidate_batch_count = 0;
    return 1;
}


static SPECTRAL_FORCEINLINE int spectral_tracker_queue_candidate(
    SpectralTracker* tracker,
    int tid,
    uint32_t* __restrict__ candidate_batch,
    size_t* candidate_batch_count,
    uint32_t candidate,
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    float t_hop,
    float threshsq,
    float freq_step_omega,
    float freq_step_df,
    float inv_hop,
    float hop_float,
    uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_validate_time
    , double* local_emit_time
    , double* local_emit_alloc_time
    , double* local_emit_interp_time
    , double* local_emit_amp_time
#endif
)
{
    if (!tracker || !candidate_batch || !candidate_batch_count) return 0;
    if (*candidate_batch_count >= SPECTRAL_TRACK_CANDIDATE_BATCH) {
        if (!spectral_tracker_flush_candidate_batch(
            tracker,
            tid,
            candidate_batch,
            candidate_batch_count,
            row,
            next_row,
            phase_row,
            next_phase_row,
            t_hop,
            threshsq,
            freq_step_omega,
            freq_step_df,
            inv_hop,
            hop_float,
            local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
            , local_validate_time
            , local_emit_time
            , local_emit_alloc_time
            , local_emit_interp_time
            , local_emit_amp_time
#endif
        )) {
            return 0;
        }
    }
    if (*candidate_batch_count >= SPECTRAL_TRACK_CANDIDATE_BATCH) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        return 0;
    }

    candidate_batch[(*candidate_batch_count)++] = candidate;
    if (*candidate_batch_count < SPECTRAL_TRACK_CANDIDATE_BATCH) return 1;

    return spectral_tracker_flush_candidate_batch(
        tracker,
        tid,
        candidate_batch,
        candidate_batch_count,
        row,
        next_row,
        phase_row,
        next_phase_row,
        t_hop,
        threshsq,
        freq_step_omega,
        freq_step_df,
        inv_hop,
        hop_float,
        local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
        , local_validate_time
        , local_emit_time
        , local_emit_alloc_time
        , local_emit_interp_time
        , local_emit_amp_time
#endif
    );
}

static int spectral_tracker_estimate_bytes(const SpectralTracker* tracker, size_t* out_bytes)
{
    size_t pairs = 0;
    size_t candidates = 0;
    size_t emitted = 0;
    size_t scan_bins = 0;
    size_t scan_triplets = 0;
    size_t validate_triplets = 0;
    size_t interp_neighbors = 0;
    size_t total_read_floats = 0;
    size_t read_floats = 0;
    size_t float_read_bytes = 0;
    size_t segment_emit_bytes = 0;
    size_t segment_merge_read_bytes = 0;
    size_t segment_merge_write_bytes = 0;
    size_t segment_bytes = 0;
    size_t total_bytes = 0;

    if (!tracker || !out_bytes) return 0;
    *out_bytes = 0;

    if (tracker->n_freqs < 3) return 1;
    if (tracker->total_pairs > (uint64_t)SIZE_MAX ||
        tracker->total_candidates > (uint64_t)SIZE_MAX ||
        tracker->total_segments > (uint64_t)SIZE_MAX) {
        return 0;
    }

    pairs = (size_t)tracker->total_pairs;
    candidates = (size_t)tracker->total_candidates;
    emitted = (size_t)tracker->total_segments;
    scan_bins = tracker->n_freqs - 2u;

    if (!spectral_size_mul(pairs, scan_bins, &scan_triplets) ||
        !spectral_size_mul(scan_triplets, 3u, &scan_triplets) ||
        !spectral_size_mul(candidates, 3u, &validate_triplets) ||
        !spectral_size_mul(emitted, 2u, &interp_neighbors) ||
        !spectral_size_add(scan_triplets, validate_triplets, &read_floats) ||
        !spectral_size_add(read_floats, interp_neighbors, &read_floats) ||
        !spectral_size_add(read_floats, emitted, &total_read_floats)) {
        return 0;
    }

    /* Fusion architecture memory access: FFT STFT data is also processed in this loop */
    size_t stft_writes = 0;
    {
        size_t fft_mags_phases = 0;
        if (!spectral_size_mul(pairs, tracker->n_freqs, &stft_writes) ||
            !spectral_size_mul(stft_writes, 2u, &fft_mags_phases) ||
            !spectral_size_add(total_read_floats, fft_mags_phases, &total_read_floats)) {
            return 0;
        }
    }

    if (!spectral_size_mul(total_read_floats, sizeof(float), &float_read_bytes) ||
        !spectral_size_mul(emitted, sizeof(TrackSegment), &segment_emit_bytes) ||
        !spectral_size_mul(emitted, sizeof(TrackSegment), &segment_merge_read_bytes) ||
        !spectral_size_mul(emitted, sizeof(TrackSegment), &segment_merge_write_bytes) ||
        !spectral_size_add(segment_emit_bytes, segment_merge_read_bytes, &segment_bytes) ||
        !spectral_size_add(segment_bytes, segment_merge_write_bytes, &segment_bytes) ||
        !spectral_size_add(float_read_bytes, segment_bytes, &total_bytes)) {
        return 0;
    }

    *out_bytes = total_bytes;
    return 1;
}

static int spectral_tracker_derive_create_scalars(size_t n_freqs,
                                                  int sr,
                                                  int n_fft,
                                                  int hop,
                                                  float db_thresh,
                                                  float max_magsq,
                                                  float* out_thresh_linear_sq,
                                                  float* out_threshsq,
                                                  float* out_inv_hop,
                                                  float* out_freq_step_omega,
                                                  float* out_freq_step_df,
                                                  float* out_hop_float)
{
    double thresh_linear_sq_d = 0.0;
    double threshsq_d = 0.0;
    double freq_step_d = 0.0;
    double two_pi_ts_d = 0.0;
    double inv_hop_d = 0.0;
    double freq_step_omega_d = 0.0;
    double freq_step_df_d = 0.0;

    if (!out_thresh_linear_sq || !out_threshsq || !out_inv_hop ||
        !out_freq_step_omega || !out_freq_step_df || !out_hop_float) {
        return 0;
    }

    if (n_freqs < 3u ||
        n_freqs > (size_t)INT_MAX ||
        sr < SPECTRAL_MIN_SAMPLE_RATE || sr > SPECTRAL_MAX_SAMPLE_RATE ||
        n_fft < SPECTRAL_MIN_FFT_SIZE ||
        ((size_t)n_fft & ((size_t)n_fft - 1u)) != 0u ||
        n_freqs != ((size_t)n_fft / 2u + 1u) ||
        hop <= 0 ||
        !isfinite(db_thresh) ||
        !isfinite(max_magsq) || max_magsq < 0.0f) {
        return 0;
    }

    thresh_linear_sq_d = pow(10.0, (double)db_thresh / 10.0);
    if (!isfinite(thresh_linear_sq_d) ||
        thresh_linear_sq_d < 0.0 ||
        thresh_linear_sq_d > (double)FLT_MAX) {
        return 0;
    }

    threshsq_d = thresh_linear_sq_d * (double)max_magsq;
    if (!isfinite(threshsq_d) ||
        threshsq_d < 0.0 ||
        threshsq_d > (double)FLT_MAX) {
        return 0;
    }

    freq_step_d = (double)sr / (double)n_fft;
    two_pi_ts_d = (double)SPECTRAL_TWO_PI / (double)sr;
    inv_hop_d = 1.0 / (double)hop;
    freq_step_omega_d = freq_step_d * two_pi_ts_d;
    freq_step_df_d = 0.5 * freq_step_d * inv_hop_d * two_pi_ts_d;

    if (!isfinite(freq_step_omega_d) || freq_step_omega_d <= 0.0 ||
        freq_step_omega_d > (double)FLT_MAX ||
        !isfinite(freq_step_df_d) || freq_step_df_d <= 0.0 ||
        freq_step_df_d > (double)FLT_MAX ||
        !isfinite(inv_hop_d) || inv_hop_d <= 0.0 ||
        inv_hop_d > (double)FLT_MAX ||
        (double)hop > (double)FLT_MAX) {
        return 0;
    }

    *out_thresh_linear_sq = (float)thresh_linear_sq_d;
    *out_threshsq = (float)threshsq_d;
    *out_inv_hop = (float)inv_hop_d;
    *out_freq_step_omega = (float)freq_step_omega_d;
    *out_freq_step_df = (float)freq_step_df_d;
    *out_hop_float = (float)hop;
    return 1;
}

SpectralTracker* spectral_tracker_create(int n_threads, size_t n_freqs,
                                          int sr, int n_fft, int hop,
                                          float db_thresh, float max_magsq) {
    int t = 0;
    float thresh_linear_sq = 0.0f;
    float threshsq = 0.0f;
    float inv_hop = 0.0f;
    float freq_step_omega = 0.0f;
    float freq_step_df = 0.0f;
    float hop_float = 0.0f;

    if (!spectral_tracker_derive_create_scalars(n_freqs, sr, n_fft, hop, db_thresh, max_magsq,
                                                &thresh_linear_sq, &threshsq, &inv_hop,
                                                &freq_step_omega, &freq_step_df, &hop_float)) {
        return NULL;
    }

    SpectralTracker* tracker = (SpectralTracker*)malloc(sizeof(SpectralTracker));
    if (!tracker) return NULL;

    if (n_threads < 1) n_threads = 1;
    if (n_threads > SPECTRAL_MAX_THREADS) return NULL;
    tracker->n_threads = n_threads;
    tracker->n_freqs = n_freqs;
    tracker->seg_arrays = NULL;
    tracker->seg_counts = NULL;
    tracker->seg_capacities = NULL;
    tracker->peak_model = (SpectralResolvedPeakModel){0};
    atomic_init(&tracker->last_error, SPECTRAL_OK);
    tracker->process_time_total = 0.0;
    tracker->total_pairs = 0;
    tracker->total_candidates = 0;
    tracker->total_segments = 0;
#if SPECTRAL_TRACK_DEBUG_TIMING
    tracker->debug_scan_time_total = 0.0;
    tracker->debug_validate_time_total = 0.0;
    tracker->debug_emit_time_total = 0.0;
    tracker->debug_emit_alloc_time_total = 0.0;
    tracker->debug_emit_interp_time_total = 0.0;
    tracker->debug_emit_amp_time_total = 0.0;
#endif

    tracker->thresh_linear_sq = thresh_linear_sq;
    tracker->threshsq = threshsq;
    tracker->inv_hop = inv_hop;
    tracker->freq_step_omega = freq_step_omega;
    tracker->freq_step_df = freq_step_df;
    tracker->hop_float = hop_float;
    tracker->peak_candan_correction = spectral_peak_candan_correction_for_n_freqs(n_freqs);
    tracker->interp_magsq = spectral_window_interp_magsq_parabolic;
    tracker->peak_magsq = spectral_window_peak_magsq_center;
    tracker->peak_estimator = SPECTRAL_PEAK_ESTIMATOR_DEFAULT;
    {
        SpectralPeakModel default_peak_model = spectral_peak_model_default();
        if (spectral_tracker_set_peak_model(tracker, &default_peak_model) != SPECTRAL_OK) {
            goto fail;
        }
    }

    /* Start with a bounded per-thread segment capacity and grow on demand.
     * The previous 16M-per-thread virtual allocation depended on Linux
     * overcommit behavior and is not a portable real-time contract. */
    size_t init_cap = (size_t)SPECTRAL_TRACK_INITIAL_SEG_CAP;
    size_t thread_slots = 0;
    size_t thread_slots_bytes = 0;
    size_t init_seg_bytes = 0;

    if (init_cap == 0u ||
        !spectral_size_mul((size_t)n_threads, (size_t)SPECTRAL_CACHE_LINE_STRIDE, &thread_slots) ||
        !spectral_size_mul(thread_slots, sizeof(size_t), &thread_slots_bytes) ||
        !spectral_size_mul(init_cap, sizeof(TrackSegment), &init_seg_bytes)) {
        goto fail;
    }

    tracker->seg_arrays = (TrackSegment**)spectral_calloc_array((size_t)n_threads, sizeof(TrackSegment*));
    /* Padded per-thread counters: only slot [tid * SPECTRAL_CACHE_LINE_STRIDE]
     * is live, but the full padded allocation must be overflow-checked. */
    tracker->seg_counts = (size_t*)spectral_aligned_alloc(thread_slots_bytes);
    tracker->seg_capacities = (size_t*)spectral_aligned_alloc(thread_slots_bytes);
    if (!tracker->seg_arrays || !tracker->seg_counts || !tracker->seg_capacities) {
        goto fail;
    }
    memset(tracker->seg_counts, 0, thread_slots_bytes);
    memset(tracker->seg_capacities, 0, thread_slots_bytes);
    for (t = 0; t < n_threads; t++) {
        tracker->seg_capacities[t * SPECTRAL_CACHE_LINE_STRIDE] = init_cap;
        tracker->seg_arrays[t] = (TrackSegment*)spectral_aligned_alloc(init_seg_bytes);
        if (!tracker->seg_arrays[t]) {
            goto fail;
        }
    }

    return tracker;

fail:
    spectral_tracker_free_segment_storage(tracker);
    free(tracker);
    return NULL;
}

void spectral_tracker_update_threshold(SpectralTracker* tracker, float new_max_magsq) {
    double threshsq_d = 0.0;

    if (!tracker) return;
    if (!isfinite(new_max_magsq) || new_max_magsq < 0.0f ||
        !isfinite(tracker->thresh_linear_sq) || tracker->thresh_linear_sq < 0.0f) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_PARAM);
        return;
    }

    threshsq_d = (double)tracker->thresh_linear_sq * (double)new_max_magsq;
    if (!isfinite(threshsq_d) || threshsq_d < 0.0 || threshsq_d > (double)FLT_MAX) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        return;
    }

    tracker->threshsq = (float)threshsq_d;
}

void spectral_tracker_process(SpectralTracker* tracker,
                               const float* chunk_magsq, const float* chunk_phases,
                               size_t chunk_n_frames, size_t global_frame_offset,
                               const float* overlap_magsq_row) {
    if (!tracker || chunk_n_frames < 1 || !chunk_magsq || !chunk_phases) return;
    if (atomic_load_explicit(&tracker->last_error, memory_order_relaxed) != SPECTRAL_OK) return;

    const size_t n_freqs = tracker->n_freqs;
    size_t chunk_bins = 0;
    const float threshsq = tracker->threshsq;
    const float freq_step_omega = tracker->freq_step_omega;
    const float freq_step_df = tracker->freq_step_df;
    const float inv_hop = tracker->inv_hop;
    const float hop_float = tracker->hop_float;

    if (!isfinite(threshsq) || threshsq < 0.0f ||
        !isfinite(freq_step_omega) || freq_step_omega <= 0.0f ||
        !isfinite(freq_step_df) || freq_step_df <= 0.0f ||
        !isfinite(inv_hop) || inv_hop <= 0.0f ||
        !isfinite(hop_float) || hop_float <= 0.0f ||
        !spectral_size_mul(chunk_n_frames, n_freqs, &chunk_bins)) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        return;
    }
    (void)chunk_bins;
    /* Number of frame-pairs we can process:
     * - All internal pairs within this chunk (chunk_n_frames - 1)
     * - Plus one extra pair if overlap_magsq_row is provided
     *   (last chunk frame paired with first frame of next chunk) */
    size_t n_pairs = chunk_n_frames - 1;
    if (overlap_magsq_row) n_pairs = chunk_n_frames;

    if (n_pairs == 0) return;
    if (global_frame_offset > SIZE_MAX - (n_pairs - 1u)) {
        spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
        return;
    }
    double process_start = omp_get_wtime();

    #pragma omp parallel num_threads(tracker->n_threads)
    {
        int tid = omp_get_thread_num();
        uint64_t local_pairs = 0;
        uint64_t local_candidates = 0;
        uint64_t local_segments = 0;
        int local_failed = 0;
#if SPECTRAL_TRACK_DEBUG_TIMING
        double local_scan_time = 0.0;
        double local_validate_time = 0.0;
        double local_emit_time = 0.0;
        double local_emit_alloc_time = 0.0;
        double local_emit_interp_time = 0.0;
        double local_emit_amp_time = 0.0;
#endif

        /* Keep contiguous static scheduling by default; optional chunking can
         * be enabled via SPECTRAL_TRACK_PAIR_OMP_CHUNK for imbalance tuning. */
#if SPECTRAL_TRACK_PAIR_OMP_CHUNK > 0u
        #pragma omp for schedule(static, SPECTRAL_TRACK_PAIR_OMP_CHUNK)
#else
        #pragma omp for schedule(static)
#endif
        for (size_t t = 0; t < n_pairs; t++) {
            /* Two-stage hot loop:
             * 1) SIMD peak scan over current frame
             * 2) batched candidate validation/emit against next frame */
            uint32_t candidate_batch[SPECTRAL_TRACK_CANDIDATE_BATCH];
            size_t candidate_batch_count = 0;
            if (local_failed) continue;
            /* Poll the global failure flag at a fixed stride to reduce
             * atomic traffic in the steady-state hot path. */
            if ((t & (SPECTRAL_TRACK_ALLOC_FAILED_POLL_STRIDE - 1u)) == 0u) {
                if (atomic_load_explicit(&tracker->last_error, memory_order_relaxed) != SPECTRAL_OK) {
                    local_failed = 1;
                    continue;
                }
            }
            local_pairs++;

            const float* __restrict__ phase_row = chunk_phases + t * n_freqs;
            const float* __restrict__ row = chunk_magsq + t * n_freqs;
            const float* __restrict__ next_row;
            const float* __restrict__ next_phase_row = NULL;

            if (t + 1 < chunk_n_frames) {
                next_row = chunk_magsq + (t + 1) * n_freqs;
                next_phase_row = chunk_phases + (t + 1) * n_freqs;
            } else {
                /* Last frame: use overlap row from next chunk */
                next_row = overlap_magsq_row;
            }

            double t_hop_d = (double)(global_frame_offset + t) * (double)hop_float;
            if (!isfinite(t_hop_d) || t_hop_d < 0.0 || t_hop_d > (double)FLT_MAX) {
                spectral_tracker_set_error(tracker, SPECTRAL_ERR_OVERFLOW);
                local_failed = 1;
                continue;
            }
            const float t_hop = (float)t_hop_d;
#if SPECTRAL_TRACK_DEBUG_TIMING
            double pair_start = omp_get_wtime();
            double pair_validate_time = 0.0;
            double pair_emit_time = 0.0;
            double pair_emit_alloc_time = 0.0;
            double pair_emit_interp_time = 0.0;
            double pair_emit_amp_time = 0.0;
#endif

            size_t f = 1;
            /* 
             * =========================================================================
             *  SIMD Peak Detection Strategy
             * =========================================================================
             * To avoid evaluating `sqrtf` or utilizing expensive branch prediction on
             * every frequency bin (which stalls the CPU pipeline), we use AVX2/SSE 
             * intrinsics to scan up to 8 bins in parallel.
             * 
             * A bin is considered a valid "peak candidate" if it satisfies 3 conditions:
             *   1. center > threshsq   (It exceeds the absolute minimum noise floor power)
             *   2. center > left       (It is strictly greater than its lower frequency neighbor)
             *   3. center > right      (It is strictly greater than its upper frequency neighbor)
             * 
             * We calculate these 3 boolean masks using float comparisons (`cmp_gt`).
             * We then mathematically AND them together: mask = (1) & (2) & (3).
             * Finally, `movemask_ps` collapses the 32-bit SIMD lane outputs into a
             * single integer bitmask. Each bit tells us if the bin at that lane index 
             * is a peak. If the 8-bit integer is 0, we found no peaks and instantly 
             * skip to the next 8 bins!
             * =========================================================================
             */
#ifdef __AVX2__
            {
                const simde__m256 vthresh = simde_mm256_set1_ps(threshsq);
                for (; f + 7 < n_freqs - 1; f += 8) {
                    simde__m256 window_left = simde_mm256_loadu_ps(row + f - 1);
                    simde__m256 window_center = simde_mm256_loadu_ps(row + f);
                    size_t pf = f + 8 + SPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE;
                    if (pf < n_freqs - 1) {
                        SPECTRAL_PREFETCH_READ_LOCALITY(row + pf, SPECTRAL_TRACK_PREFETCH_READ_LOCALITY);
                    }
                    simde__m256 window_right = simde_mm256_loadu_ps(row + f + 1);
                    int bits = simde_mm256_movemask_ps(simde_mm256_and_ps(
                        simde_mm256_cmp_ps(window_center, vthresh, SIMDE_CMP_GT_OQ),
                        simde_mm256_and_ps(
                            simde_mm256_cmp_ps(window_center, window_left,  SIMDE_CMP_GT_OQ),
                            simde_mm256_cmp_ps(window_center, window_right, SIMDE_CMP_GT_OQ))));
                    
                    if (!spectral_tracker_process_bitmask(
                            tracker, tid, candidate_batch, &candidate_batch_count,
                            row, next_row, phase_row, next_phase_row, f, bits, t_hop, threshsq,
                            freq_step_omega, freq_step_df, inv_hop, hop_float,
                            &local_candidates, &local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                            , &pair_validate_time, &pair_emit_time, &pair_emit_alloc_time,
                            &pair_emit_interp_time, &pair_emit_amp_time
#endif
                            )) {
                        local_failed = 1;
                        break;
                    }
                    if (local_failed) break;
                }
            }
#else
            {
                const simde__m128 vthresh = simde_mm_set1_ps(threshsq);
                for (; f + 3 < n_freqs - 1; f += 4) {
                    simde__m128 window_left = simde_mm_loadu_ps(row + f - 1);
                    simde__m128 window_center = simde_mm_loadu_ps(row + f);
                    size_t pf = f + 4 + SPECTRAL_TRACK_SCAN_PREFETCH_DISTANCE;
                    if (pf < n_freqs - 1) {
                        SPECTRAL_PREFETCH_READ_LOCALITY(row + pf, SPECTRAL_TRACK_PREFETCH_READ_LOCALITY);
                    }
                    simde__m128 window_right = simde_mm_loadu_ps(row + f + 1);
                    int bits = simde_mm_movemask_ps(simde_mm_and_ps(
                        simde_mm_cmpgt_ps(window_center, vthresh),
                        simde_mm_and_ps(
                            simde_mm_cmpgt_ps(window_center, window_left),
                            simde_mm_cmpgt_ps(window_center, window_right))));
                    
                    if (!spectral_tracker_process_bitmask(
                            tracker, tid, candidate_batch, &candidate_batch_count,
                            row, next_row, phase_row, next_phase_row, f, bits, t_hop, threshsq,
                            freq_step_omega, freq_step_df, inv_hop, hop_float,
                            &local_candidates, &local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                            , &pair_validate_time, &pair_emit_time, &pair_emit_alloc_time,
                            &pair_emit_interp_time, &pair_emit_amp_time
#endif
                            )) {
                        local_failed = 1;
                        break;
                    }
                    if (local_failed) break;
                }
            }
#endif
            /* Scalar tail for remaining bins */
            for (; !local_failed && f < n_freqs - 1; f++) {
                float curr = row[f];
                if (curr > threshsq && curr > row[f-1] && curr > row[f+1]) {
                    local_candidates++;
                    if (!spectral_tracker_queue_candidate(
                            tracker,
                            tid,
                            candidate_batch,
                            &candidate_batch_count,
                            (uint32_t)f,
                            row,
                            next_row,
                            phase_row,
                            next_phase_row,
                            t_hop,
                            threshsq,
                            freq_step_omega,
                            freq_step_df,
                            inv_hop,
                            hop_float,
                            &local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                            , &pair_validate_time
                            , &pair_emit_time
                            , &pair_emit_alloc_time
                            , &pair_emit_interp_time
                            , &pair_emit_amp_time
#endif
                            )) {
                        local_failed = 1;
                        break;
                    }
                }
            }

            if (!local_failed &&
                !spectral_tracker_flush_candidate_batch(
                    tracker,
                    tid,
                    candidate_batch,
                    &candidate_batch_count,
                    row,
                    next_row,
                    phase_row,
                    next_phase_row,
                    t_hop,
                    threshsq,
                    freq_step_omega,
                    freq_step_df,
                    inv_hop,
                    hop_float,
                    &local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                    , &pair_validate_time
                    , &pair_emit_time
                    , &pair_emit_alloc_time
                    , &pair_emit_interp_time
                    , &pair_emit_amp_time
#endif
                    )) {
                local_failed = 1;
            }
#if SPECTRAL_TRACK_DEBUG_TIMING
            {
                double pair_total = omp_get_wtime() - pair_start;
                double pair_scan = pair_total - pair_validate_time - pair_emit_time;
                if (pair_scan < 0.0) pair_scan = 0.0;
                local_scan_time += pair_scan;
                local_validate_time += pair_validate_time;
                local_emit_time += pair_emit_time;
                local_emit_alloc_time += pair_emit_alloc_time;
                local_emit_interp_time += pair_emit_interp_time;
                local_emit_amp_time += pair_emit_amp_time;
            }
#endif
        }

        #pragma omp atomic update
        tracker->total_pairs += local_pairs;
        #pragma omp atomic update
        tracker->total_candidates += local_candidates;
        #pragma omp atomic update
        tracker->total_segments += local_segments;
        if (tracker->seg_counts[tid * SPECTRAL_CACHE_LINE_STRIDE] > (size_t)UINT32_MAX) {
            atomic_store_explicit(&tracker->last_error, SPECTRAL_ERR_OVERFLOW, memory_order_relaxed);
        }
#if SPECTRAL_TRACK_DEBUG_TIMING
        #pragma omp atomic update
        tracker->debug_scan_time_total += local_scan_time;
        #pragma omp atomic update
        tracker->debug_validate_time_total += local_validate_time;
        #pragma omp atomic update
        tracker->debug_emit_time_total += local_emit_time;
        #pragma omp atomic update
        tracker->debug_emit_alloc_time_total += local_emit_alloc_time;
        #pragma omp atomic update
        tracker->debug_emit_interp_time_total += local_emit_interp_time;
        #pragma omp atomic update
        tracker->debug_emit_amp_time_total += local_emit_amp_time;
#endif
    }

    tracker->process_time_total += omp_get_wtime() - process_start;
}

SegmentArray spectral_tracker_finalize(SpectralTracker* tracker, double* t_track, double override_wall_time) {
    size_t total_segs = 0;
    size_t* offsets = NULL;
    Segment* segs = NULL;
    int n_threads = 0;
    size_t merge_bytes = 0;
    double finalize_start = 0.0;
    double track_total_time = 0.0;

    if (!tracker) return peak_track_return_empty(t_track);
    finalize_start = omp_get_wtime();

    n_threads = tracker->n_threads;

    if (atomic_load_explicit(&tracker->last_error, memory_order_relaxed) != SPECTRAL_OK) {
        goto fail;
    }

    /* Compute total and per-thread offsets */
    offsets = (size_t*)spectral_malloc_array((size_t)n_threads, sizeof(size_t));
    if (!offsets) {
        goto fail;
    }
    for (int t = 0; t < n_threads; t++) {
        size_t next_total = 0;
        offsets[t] = total_segs;
        if (!spectral_size_add(total_segs, tracker->seg_counts[t * SPECTRAL_CACHE_LINE_STRIDE], &next_total)) {
            goto fail;
        }
        total_segs = next_total;
    }
    if (total_segs > (size_t)UINT32_MAX) {
        goto fail;
    }

    /* Compute merge allocation size */
    if (!spectral_array_bytes(total_segs, sizeof(Segment), &merge_bytes)) {
        goto fail;
    }

    /* Merge into contiguous array.  For large allocations, use page-aligned
     * memory with kernel-assisted pre-fault (MADV_POPULATE_WRITE) to avoid
     * 63K+ individual minor page faults during the parallel copy. */
    if (total_segs > 0) {
        if (merge_bytes > SPECTRAL_PRETOUCH_THRESHOLD) {
            if (posix_memalign((void**)&segs, SPECTRAL_PRETOUCH_PAGE_SIZE, merge_bytes) != 0) {
                segs = NULL;
                goto fail;
            }
#if defined(__linux__)
            if (madvise(segs, merge_bytes, SPECTRAL_TRACK_MADV_POPULATE_WRITE) != 0)
#endif
            {
                size_t n_pages = (merge_bytes + SPECTRAL_PRETOUCH_PAGE_SIZE - 1) / SPECTRAL_PRETOUCH_PAGE_SIZE;
                #pragma omp parallel for schedule(static)
                for (size_t pg = 0; pg < n_pages; pg++) {
                    ((volatile char*)segs)[pg * SPECTRAL_PRETOUCH_PAGE_SIZE] = 0;
                }
            }
        } else {
            segs = (Segment*)spectral_malloc_array(total_segs, sizeof(Segment));
            if (!segs) goto fail;
        }
    }

    /* _Static_assert to enforce layout compatibility between TrackSegment and Segment prefix */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    _Static_assert(offsetof(Segment, start) == offsetof(TrackSegment, start), "Segment/TrackSegment layout: start");
    _Static_assert(offsetof(Segment, width) == offsetof(TrackSegment, width), "Segment/TrackSegment layout: width");
    _Static_assert(sizeof(TrackSegment) <= sizeof(Segment), "TrackSegment must fit within Segment");
#endif

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < n_threads; t++) {
        if (tracker->seg_counts[t * SPECTRAL_CACHE_LINE_STRIDE] > 0) {
            /* We copy the TrackSegment array into the final Segment array */
            Segment* dst = segs + offsets[t];
            TrackSegment* src = tracker->seg_arrays[t];
            
            for (size_t i = 0; i < tracker->seg_counts[t * SPECTRAL_CACHE_LINE_STRIDE]; i++) {
                /* Use memcpy instead of pointer cast to avoid strict aliasing violation.
                 * Modern compilers optimize 32-byte memcpy into the same AVX move. */
                memcpy(&dst[i], &src[i], sizeof(TrackSegment));
                /* Clear the 32-byte cache-line padding to prevent uninitialized heap 
                 * memory from leaking when the segments are serialized to disk cache. */
                memset((void*)&dst[i]._pad_w, 0, sizeof(dst[i]._pad_w));
            }
        }
    }
    
    /* Perform OS-level free sequentially to avoid massive mmap_sem lock
     * contention across 20 threads simultaneously unmapping 512MB allocations. */
    spectral_tracker_free_segment_storage(tracker);
    free(offsets);

    track_total_time = (override_wall_time > 0.0) 
        ? override_wall_time + (omp_get_wtime() - finalize_start)
        : tracker->process_time_total + (omp_get_wtime() - finalize_start);
    if (t_track) *t_track = track_total_time;
    {
        size_t tracker_bytes = 0;
        size_t tracker_pairs = (tracker->total_pairs > (uint64_t)SIZE_MAX)
            ? SIZE_MAX
            : (size_t)tracker->total_pairs;
        size_t tracker_candidates = (tracker->total_candidates > (uint64_t)SIZE_MAX)
            ? SIZE_MAX
            : (size_t)tracker->total_candidates;
        size_t tracker_segments = (tracker->total_segments > (uint64_t)SIZE_MAX)
            ? SIZE_MAX
            : (size_t)tracker->total_segments;
        if (spectral_tracker_estimate_bytes(tracker, &tracker_bytes)) {
            double track_bw_gibps = spectral_bandwidth_gibps(tracker_bytes, track_total_time);
            double candidates_per_pair = (tracker_pairs > 0)
                ? ((double)tracker_candidates / (double)tracker_pairs)
                : 0.0;
            double acceptance_pct = (tracker_candidates > 0)
                ? (100.0 * (double)tracker_segments / (double)tracker_candidates)
                : 0.0;
            SPECTRAL_LOG_INFO("Analysis track: pairs=%zu candidates=%zu segments=%zu cand_per_pair=%.2f accept=%.1f%% bytes=%.1fMiB time=%.3fms bw=%.2fGiB/s",
                              tracker_pairs,
                              tracker_candidates,
                              tracker_segments,
                              candidates_per_pair,
                              acceptance_pct,
                              BYTES_TO_MB(tracker_bytes),
                              track_total_time * SPECTRAL_MILLIS_PER_SECOND_D,
                              track_bw_gibps);
        }
#if SPECTRAL_TRACK_DEBUG_TIMING
        {
            double process_total = tracker->process_time_total;
            double scan_ms = tracker->debug_scan_time_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double validate_ms = tracker->debug_validate_time_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double emit_ms = tracker->debug_emit_time_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double emit_alloc_ms = tracker->debug_emit_alloc_time_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double emit_interp_ms = tracker->debug_emit_interp_time_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double emit_amp_ms = tracker->debug_emit_amp_time_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double thread_total = tracker->debug_scan_time_total +
                                  tracker->debug_validate_time_total +
                                  tracker->debug_emit_time_total;
            double thread_total_ms = thread_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double process_ms = process_total * SPECTRAL_MILLIS_PER_SECOND_D;
            double scan_pct = (thread_total > 0.0)
                ? (100.0 * tracker->debug_scan_time_total / thread_total)
                : 0.0;
            double validate_pct = (thread_total > 0.0)
                ? (100.0 * tracker->debug_validate_time_total / thread_total)
                : 0.0;
            double emit_pct = (thread_total > 0.0)
                ? (100.0 * tracker->debug_emit_time_total / thread_total)
                : 0.0;
            double avg_parallel = (process_total > 0.0)
                ? (thread_total / process_total)
                : 0.0;
            double emit_alloc_pct = (tracker->debug_emit_time_total > 0.0)
                ? (100.0 * tracker->debug_emit_alloc_time_total / tracker->debug_emit_time_total)
                : 0.0;
            double emit_interp_pct = (tracker->debug_emit_time_total > 0.0)
                ? (100.0 * tracker->debug_emit_interp_time_total / tracker->debug_emit_time_total)
                : 0.0;
            double emit_amp_pct = (tracker->debug_emit_time_total > 0.0)
                ? (100.0 * tracker->debug_emit_amp_time_total / tracker->debug_emit_time_total)
                : 0.0;
            double emit_unattributed = tracker->debug_emit_time_total -
                (tracker->debug_emit_alloc_time_total +
                 tracker->debug_emit_interp_time_total +
                 tracker->debug_emit_amp_time_total);
            double emit_unattributed_ms = 0.0;
            double emit_unattributed_pct = 0.0;
            if (emit_unattributed < 0.0) emit_unattributed = 0.0;
            emit_unattributed_ms = emit_unattributed * SPECTRAL_MILLIS_PER_SECOND_D;
            emit_unattributed_pct = (tracker->debug_emit_time_total > 0.0)
                ? (100.0 * emit_unattributed / tracker->debug_emit_time_total)
                : 0.0;
            SPECTRAL_LOG_INFO("Track phase breakdown (thread-time): scan=%.3fms (%.1f%%) validate=%.3fms (%.1f%%) emit_total=%.3fms (%.1f%%) thread_total=%.3fms wall_process=%.3fms avg_parallel=%.2fx",
                         scan_ms, scan_pct,
                         validate_ms, validate_pct,
                         emit_ms, emit_pct,
                         thread_total_ms, process_ms, avg_parallel);
            SPECTRAL_LOG_INFO("Track emit breakdown (thread-time): alloc_store=%.3fms (%.1f%% emit) sqrt_amp=%.3fms (%.1f%% emit) interp_fit=%.3fms (%.1f%% emit) unattributed=%.3fms (%.1f%% emit)",
                         emit_alloc_ms, emit_alloc_pct,
                         emit_amp_ms, emit_amp_pct,
                         emit_interp_ms, emit_interp_pct,
                         emit_unattributed_ms, emit_unattributed_pct);
        }
#endif
    }

    SegmentArray res;
    res.segs = segs;
    res.count = (uint32_t)total_segs;
    res.capacity = (uint32_t)total_segs;

    return res;

fail:
    free(segs);
    free(offsets);
    spectral_tracker_free_segment_storage(tracker);
    return peak_track_return_empty(t_track);
}

void spectral_tracker_destroy(SpectralTracker* tracker) {
    if (!tracker) return;
    spectral_tracker_free_segment_storage(tracker);
    free(tracker);
}

/* spectral_track_peaks_with_window_descriptor — single-shot wrapper around
 * SpectralTracker with an explicit analysis-window/estimator contract. */

SegmentArray spectral_track_peaks_with_window_descriptor(
    const float* magsq, const float* phases,
    float max_magsq,
    size_t n_frames, size_t n_freqs,
    int sr, int n_fft, int hop,
    float db_thresh,
    const SpectralWindowDescriptor* window_desc,
    SpectralPeakEstimatorType estimator,
    double* t_track) {
    if (!magsq || !phases || !t_track || n_frames < 2 || n_freqs < 3) {
        return peak_track_return_empty(t_track);
    }
    /* Guard against NaN/Inf in float parameters at the public API boundary */
    if (isnan(max_magsq) || isinf(max_magsq) || max_magsq < 0.0f) {
        return peak_track_return_empty(t_track);
    }
    if (isnan(db_thresh) || isinf(db_thresh)) {
        return peak_track_return_empty(t_track);
    }

    int n_threads = omp_get_max_threads();
    SpectralPeakModel peak_model = spectral_peak_model_for_window(
        window_desc ? window_desc : spectral_window_descriptor(SPECTRAL_WINDOW_HANN));

    SpectralTracker* tracker = spectral_tracker_create(
        n_threads, n_freqs, sr, n_fft, hop, db_thresh, max_magsq);
    if (!tracker) {
        return peak_track_return_empty(t_track);
    }

    peak_model.estimator = estimator;
    if (spectral_tracker_set_peak_model(tracker, &peak_model) != SPECTRAL_OK) {
        spectral_tracker_destroy(tracker);
        return peak_track_return_empty(t_track);
    }

    /* Process all frames in one shot — no overlap row (NULL for final chunk) */
    spectral_tracker_process(tracker, magsq, phases,
                              n_frames, 0, NULL);

    SegmentArray result = spectral_tracker_finalize(tracker, t_track, 0.0);
    spectral_tracker_destroy(tracker);
    return result;
}

/* spectral_track_peaks — compatibility wrapper for the engine default
 * Hann-windowed analysis path and AUTO estimator policy. */

SegmentArray spectral_track_peaks(const float* magsq, const float* phases,
                                  float max_magsq,
                                  size_t n_frames, size_t n_freqs,
                                  int sr, int n_fft, int hop,
                                  float db_thresh, double* t_track) {
    return spectral_track_peaks_with_window_descriptor(
        magsq, phases, max_magsq,
        n_frames, n_freqs,
        sr, n_fft, hop,
        db_thresh,
        spectral_window_descriptor(SPECTRAL_WINDOW_HANN),
        SPECTRAL_PEAK_ESTIMATOR_AUTO,
        t_track);
}

static SPECTRAL_FORCEINLINE int spectral_tracker_process_bitmask(
    SpectralTracker* tracker, int tid,
    uint32_t* __restrict__ candidate_batch, size_t* candidate_batch_count,
    const float* __restrict__ row, const float* __restrict__ next_row, const float* __restrict__ phase_row,
    const float* __restrict__ next_phase_row,
    size_t f, int bits, float t_hop, float threshsq,
    float freq_step_omega, float freq_step_df, float inv_hop, float hop_float,
    uint64_t* local_candidates, uint64_t* local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
    , double* local_validate_time, double* local_emit_time, double* local_emit_alloc_time,
    double* local_emit_interp_time, double* local_emit_amp_time
#endif
) {
    /* 
     * We have a non-zero SIMD bitmask representing one or more peak hits.
     * `__builtin_ctz` (Count Trailing Zeros) gives us the exact lane index 
     * offset of the lowest set bit in a single hardware instruction. 
     * We then clear that bit (`bits &= bits - 1`) and repeat until no bits 
     * remain. This algorithm instantly hops from peak to peak, entirely 
     * bypassing the intermediate empty space.
     */
    while (bits) {
        size_t cf = f + (size_t)__builtin_ctz((unsigned)bits);
        bits &= bits - 1;
        (*local_candidates)++;
        if (!spectral_tracker_queue_candidate(
                tracker, tid, candidate_batch, candidate_batch_count,
                (uint32_t)cf, row, next_row, phase_row, next_phase_row, t_hop, threshsq,
                freq_step_omega, freq_step_df, inv_hop, hop_float,
                local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                , local_validate_time, local_emit_time, local_emit_alloc_time,
                local_emit_interp_time, local_emit_amp_time
#endif
                )) {
            return 0;
        }
    }
    return 1;
}

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
) {
    int local_failed = 0;
    const float* __restrict__ row = ctx->row;
    const float* __restrict__ next_row = ctx->next_row;
    const float* __restrict__ phase_row = ctx->phase_row;
    const float* __restrict__ next_phase_row = ctx->next_phase_row;
    float t_hop = ctx->t_hop;
    float threshsq = ctx->threshsq;
    int can_start_new = ctx->can_start_new;
    size_t n_freqs = tracker->n_freqs;
    float freq_step_omega = tracker->freq_step_omega;
    float freq_step_df = tracker->freq_step_df;
    float inv_hop = tracker->inv_hop;
    float hop_float = tracker->hop_float;
    size_t f = 1;

#ifdef __AVX2__
    if (can_start_new) {
        const simde__m256 vthresh = simde_mm256_set1_ps(threshsq);
        for (; f + 7 < n_freqs - 1; f += 8) {
            simde__m256 window_left = simde_mm256_loadu_ps(row + f - 1);
            simde__m256 window_center = simde_mm256_loadu_ps(row + f);
            simde__m256 thresh_mask = simde_mm256_cmp_ps(window_center, vthresh, SIMDE_CMP_GT_OQ);
            int thresh_bits = simde_mm256_movemask_ps(thresh_mask);

            if (SPECTRAL_UNLIKELY(thresh_bits)) {
                simde__m256 window_right = simde_mm256_loadu_ps(row + f + 1);
                int bits = simde_mm256_movemask_ps(simde_mm256_and_ps(
                    thresh_mask,
                    simde_mm256_and_ps(
                        simde_mm256_cmp_ps(window_center, window_left,  SIMDE_CMP_GT_OQ),
                        simde_mm256_cmp_ps(window_center, window_right, SIMDE_CMP_GT_OQ))));
                if (!spectral_tracker_process_bitmask(
                        tracker, tid, candidate_batch, candidate_batch_count,
                        row, next_row, phase_row, next_phase_row, f, bits, t_hop, threshsq,
                        freq_step_omega, freq_step_df, inv_hop, hop_float,
                        local_candidates, local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                        , pair_validate_time, pair_emit_time, pair_emit_alloc_time,
                        pair_emit_interp_time, pair_emit_amp_time
#endif
                        )) {
                    local_failed = 1;
                    break;
                }
            }
            if (local_failed) break;
        }
    }
#else
    if (can_start_new) {
        const simde__m128 vthresh = simde_mm_set1_ps(threshsq);
        for (; f + 3 < n_freqs - 1; f += 4) {
            simde__m128 window_left = simde_mm_loadu_ps(row + f - 1);
            simde__m128 window_center = simde_mm_loadu_ps(row + f);
            simde__m128 thresh_mask = simde_mm_cmpgt_ps(window_center, vthresh);
            int thresh_bits = simde_mm_movemask_ps(thresh_mask);

            if (SPECTRAL_UNLIKELY(thresh_bits)) {
                simde__m128 window_right = simde_mm_loadu_ps(row + f + 1);
                int bits = simde_mm_movemask_ps(simde_mm_and_ps(
                    thresh_mask,
                    simde_mm_and_ps(
                        simde_mm_cmpgt_ps(window_center, window_left),
                        simde_mm_cmpgt_ps(window_center, window_right))));
                
                if (!spectral_tracker_process_bitmask(
                        tracker, tid, candidate_batch, candidate_batch_count,
                        row, next_row, phase_row, next_phase_row, f, bits, t_hop, threshsq,
                        freq_step_omega, freq_step_df, inv_hop, hop_float,
                        local_candidates, local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                        , pair_validate_time, pair_emit_time, pair_emit_alloc_time,
                        pair_emit_interp_time, pair_emit_amp_time
#endif
                        )) {
                    local_failed = 1;
                    break;
                }
            }
            if (local_failed) break;
        }
    }
#endif
    if (can_start_new) {
        for (; !local_failed && f < n_freqs - 1; f++) {
            float curr = row[f];
            if (curr > threshsq && curr > row[f-1] && curr > row[f+1]) {
                (*local_candidates)++;
                if (!spectral_tracker_queue_candidate(
                        tracker, tid, candidate_batch, candidate_batch_count,
                        (uint32_t)f, row, next_row, phase_row, next_phase_row, t_hop, threshsq,
                        freq_step_omega, freq_step_df, inv_hop, hop_float,
                        local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                        , pair_validate_time, pair_emit_time, pair_emit_alloc_time,
                        pair_emit_interp_time, pair_emit_amp_time
#endif
                        )) {
                    local_failed = 1;
                    break;
                }
            }
        }
    }
    if (!local_failed && *candidate_batch_count > 0) {
        if (!spectral_tracker_flush_candidate_batch(
                tracker, tid, candidate_batch, candidate_batch_count,
                row, next_row, phase_row, next_phase_row, t_hop, threshsq,
                freq_step_omega, freq_step_df, inv_hop, hop_float,
                local_segments
#if SPECTRAL_TRACK_DEBUG_TIMING
                , pair_validate_time, pair_emit_time, pair_emit_alloc_time,
                pair_emit_interp_time, pair_emit_amp_time
#endif
                )) {
            local_failed = 1;
        }
    }
    
    return local_failed;
}
