/* spectral_peak_interp.h - Sub-bin Frequency Interpolation
 *
 * INTERNAL: This header is part of the analysis module implementation.
 * Do NOT include from outside spectral_engine/analysis/. */
#ifndef SPECTRAL_PEAK_INTERP_H
#define SPECTRAL_PEAK_INTERP_H

#include "spectral_config.h"
#include "spectral_peak_track_internal.h"
#include "spectral_omp.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * STRATEGY: Sub-Bin Peak Interpolation & Validation
 *
 * This module handles the precise sub-bin frequency and amplitude estimation
 * for detected spectral peaks. Because STFT bins are discrete, a peak rarely
 * falls exactly on a bin center.
 * 
 * 1. Validation: Ensures a candidate peak is a true local maximum comparing
 *    its magnitude against immediately adjacent frequency bins.
 * 2. Emission: Estimates sub-bin frequency (omega), delta frequency (df),
 *    and amplitude slope (amp/da). The sub-bin estimator is window-aware:
 *    the active SpectralWindowDescriptor supplies the interpolation rule.
 */

int spectral_tracker_validate_candidate(
    const float* __restrict__ row,
    const float* __restrict__ next_row,
    size_t cf,
    float threshsq,
    float* out_curr,
    float* out_max_vsq,
    int* out_best_next);

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
);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PEAK_INTERP_H */
