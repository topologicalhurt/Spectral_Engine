/* spectral_peak_model.h - Validated peak-model profiles
 *
 * A peak model binds the coupled DSP choices used by the tracker:
 * window shape, frequency estimator, amplitude peak-height model,
 * phase policy, and temporal requirements.
 */
#ifndef SPECTRAL_PEAK_MODEL_H
#define SPECTRAL_PEAK_MODEL_H

#include "spectral_peak_estimator.h"
#include "spectral_windows.h"
#include "spectral_error.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SpectralPeakModelCapability {
    SPECTRAL_PEAK_MODEL_CAP_MAG_TRIPLET          = 1u << 0,
    SPECTRAL_PEAK_MODEL_CAP_COMPLEX_TRIPLET      = 1u << 1,
    SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW            = 1u << 2,
    SPECTRAL_PEAK_MODEL_CAP_NEXT_MAG_ROW         = 1u << 3,
    SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW       = 1u << 4,
    SPECTRAL_PEAK_MODEL_CAP_PEAK_HEIGHT          = 1u << 5,
    SPECTRAL_PEAK_MODEL_CAP_PHASE_DIAGNOSTIC     = 1u << 6
} SpectralPeakModelCapability;

typedef enum SpectralPeakModelAssumption {
    SPECTRAL_PEAK_MODEL_ASSUME_WINDOW_LOG_MAINLOBE     = 1u << 0,
    SPECTRAL_PEAK_MODEL_ASSUME_WINDOW_RECT_MAINLOBE    = 1u << 1,
    SPECTRAL_PEAK_MODEL_ASSUME_ESTIMATOR_COMPLEX_DFT   = 1u << 2,
    SPECTRAL_PEAK_MODEL_ASSUME_ESTIMATOR_MAG_POWER     = 1u << 3,
    SPECTRAL_PEAK_MODEL_ASSUME_PHASE_VOCODER_ADVANCE   = 1u << 4
} SpectralPeakModelAssumption;

typedef struct SpectralPeakModel {
    const SpectralWindowDescriptor* window;
    SpectralPeakEstimatorType estimator;
    SpectralPeakPhasePolicy phase_policy;
    SpectralPeakAmplitudePolicy amplitude_policy;
    unsigned capabilities;
    unsigned assumptions;
} SpectralPeakModel;

typedef struct SpectralResolvedPeakModel {
    const SpectralWindowDescriptor* window;
    SpectralWindowInterpMagsqFn interp_magsq;
    SpectralWindowPeakMagsqFn peak_magsq;
    SpectralPeakEstimatorType estimator;
    SpectralPeakPhasePolicy phase_policy;
    SpectralPeakAmplitudePolicy amplitude_policy;
    unsigned capabilities;
    unsigned assumptions;
} SpectralResolvedPeakModel;

const char* spectral_peak_phase_policy_name(SpectralPeakPhasePolicy policy);
const char* spectral_peak_amplitude_policy_name(SpectralPeakAmplitudePolicy policy);

SpectralPeakModel spectral_peak_model_default(void);
SpectralPeakModel spectral_peak_model_for_window(const SpectralWindowDescriptor* window);
SpectralPeakModel spectral_peak_model_for_window_type(SpectralWindowType type);
SpectralPeakModel spectral_peak_model_from_resolved(const SpectralResolvedPeakModel* resolved);

SpectralError spectral_peak_model_validate(const SpectralPeakModel* model);
SpectralError spectral_peak_model_resolve(const SpectralPeakModel* model,
                                          SpectralResolvedPeakModel* out);

int spectral_peak_model_has_capability(const SpectralPeakModel* model, unsigned capability);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PEAK_MODEL_H */
