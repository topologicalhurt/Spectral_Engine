/* spectral_peak_estimator.h - Peak frequency/amplitude estimation policies
 *
 * Internal analysis module API.
 *
 * A detected FFT-bin maximum is only a coarse peak location. This module
 * estimates the sub-bin offset, amplitude, amplitude slope and segment
 * frequency fields used by the sinusoidal tracker.
 */
#ifndef SPECTRAL_PEAK_ESTIMATOR_H
#define SPECTRAL_PEAK_ESTIMATOR_H

#include "spectral_config.h"
#include "spectral_windows.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SpectralPeakEstimatorType {
    SPECTRAL_PEAK_ESTIMATOR_AUTO = 0,
    SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC = 1,
    SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC = 2,
    SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX = 3,
    SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX = 4,
    SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND = 5,
    SPECTRAL_PEAK_ESTIMATOR_COUNT
} SpectralPeakEstimatorType;

#ifndef SPECTRAL_PEAK_ESTIMATOR_DEFAULT
#define SPECTRAL_PEAK_ESTIMATOR_DEFAULT SPECTRAL_PEAK_ESTIMATOR_AUTO
#endif

typedef enum SpectralPeakEstimateFlags {
    SPECTRAL_PEAK_ESTIMATE_BIN_OFFSET_VALID = 1u << 0,
    SPECTRAL_PEAK_ESTIMATE_AMP_VALID        = 1u << 1,
    SPECTRAL_PEAK_ESTIMATE_OMEGA_VALID      = 1u << 2,
    SPECTRAL_PEAK_ESTIMATE_DF_VALID         = 1u << 3,
    SPECTRAL_PEAK_ESTIMATE_USED_FALLBACK    = 1u << 4,
    SPECTRAL_PEAK_ESTIMATE_COMPLEX_USED     = 1u << 5
} SpectralPeakEstimateFlags;

typedef struct SpectralPeakEstimateInput {
    const float* magsq_row;
    const float* phase_row;
    size_t n_freqs;
    size_t bin;

    float curr_magsq;
    float next_max_magsq;
    /* Winner from next_frame[bin-1..bin+1]. Public-safe estimation rejects
     * values outside that neighborhood before df arithmetic. */
    int   best_next_bin;

    float freq_step_omega;
    float freq_step_df;
    float inv_hop;
    float hop_float;
    /* Precomputed tan(pi/n_fft)/(pi/n_fft) for Candan. Zero means compute
     * locally, which keeps standalone public calls safe. */
    float candan_correction;

    SpectralWindowInterpMagsqFn interp_magsq;
    SpectralPeakEstimatorType type;
} SpectralPeakEstimateInput;

typedef struct SpectralPeakEstimate {
    float bin_offset;
    float amp;
    float next_amp;
    float da;
    float omega;
    float df;
    unsigned flags;
    SpectralPeakEstimatorType requested_type;
    SpectralPeakEstimatorType used_type;
} SpectralPeakEstimate;

const char* spectral_peak_estimator_name(SpectralPeakEstimatorType type);
int spectral_peak_estimator_type_supported(SpectralPeakEstimatorType type);
SpectralPeakEstimatorType spectral_peak_estimator_resolve_default(SpectralPeakEstimatorType type);
float spectral_peak_candan_correction_for_n_freqs(size_t n_freqs);

int spectral_peak_estimate(const SpectralPeakEstimateInput* input,
                           SpectralPeakEstimate* out);
/* Tracker-only hot path. Call only after spectral_tracker_validate_candidate()
 * has accepted row[bin-1..bin+1] and the next-frame triplet. */
int spectral_peak_estimate_validated(const SpectralPeakEstimateInput* input,
                                     SpectralPeakEstimate* out);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PEAK_ESTIMATOR_H */
