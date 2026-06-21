/* spectral_peak_model.c - Validated peak-model profiles */
#include "spectral_peak_model.h"
#include <stddef.h>

static int spectral_peak_phase_policy_supported(SpectralPeakPhasePolicy policy) {
    return policy == SPECTRAL_PEAK_PHASE_POLICY_IGNORE ||
           policy == SPECTRAL_PEAK_PHASE_POLICY_OBSERVE ||
           policy == SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT;
}

static int spectral_peak_amplitude_policy_supported(SpectralPeakAmplitudePolicy policy) {
    return policy == SPECTRAL_PEAK_AMP_POLICY_CENTER ||
           policy == SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED;
}

const char* spectral_peak_phase_policy_name(SpectralPeakPhasePolicy policy) {
    switch (policy) {
        case SPECTRAL_PEAK_PHASE_POLICY_IGNORE: return "ignore";
        case SPECTRAL_PEAK_PHASE_POLICY_OBSERVE: return "observe";
        case SPECTRAL_PEAK_PHASE_POLICY_REJECT_INCONSISTENT: return "reject-inconsistent";
        default: return "unknown";
    }
}

const char* spectral_peak_amplitude_policy_name(SpectralPeakAmplitudePolicy policy) {
    switch (policy) {
        case SPECTRAL_PEAK_AMP_POLICY_CENTER: return "center";
        case SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED: return "interp-bounded";
        default: return "unknown";
    }
}

static unsigned spectral_peak_model_estimator_capabilities(SpectralPeakEstimatorType estimator) {
    switch (spectral_peak_estimator_resolve_default(estimator)) {
        case SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX:
        case SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX:
        case SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND:
            return SPECTRAL_PEAK_MODEL_CAP_MAG_TRIPLET |
                   SPECTRAL_PEAK_MODEL_CAP_COMPLEX_TRIPLET |
                   SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW;
        case SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC:
        case SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC:
        default:
            return SPECTRAL_PEAK_MODEL_CAP_MAG_TRIPLET;
    }
}

static unsigned spectral_peak_model_estimator_assumptions(SpectralPeakEstimatorType estimator) {
    switch (spectral_peak_estimator_resolve_default(estimator)) {
        case SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX:
        case SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX:
        case SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND:
            return SPECTRAL_PEAK_MODEL_ASSUME_ESTIMATOR_COMPLEX_DFT;
        case SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC:
            return SPECTRAL_PEAK_MODEL_ASSUME_ESTIMATOR_MAG_POWER;
        case SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC:
        default:
            return SPECTRAL_PEAK_MODEL_ASSUME_ESTIMATOR_MAG_POWER |
                   SPECTRAL_PEAK_MODEL_ASSUME_WINDOW_LOG_MAINLOBE;
    }
}

static unsigned spectral_peak_model_window_assumptions(const SpectralWindowDescriptor* window) {
    if (!window) return 0u;
    if (window->type == SPECTRAL_WINDOW_RECTANGULAR) {
        return SPECTRAL_PEAK_MODEL_ASSUME_WINDOW_RECT_MAINLOBE;
    }
    return SPECTRAL_PEAK_MODEL_ASSUME_WINDOW_LOG_MAINLOBE;
}

static SpectralPeakAmplitudePolicy spectral_peak_model_default_amplitude_for_window(
    const SpectralWindowDescriptor* window)
{
    if (!window || !window->peak_magsq ||
        window->type == SPECTRAL_WINDOW_RECTANGULAR) {
        return SPECTRAL_PEAK_AMP_POLICY_CENTER;
    }
    return SPECTRAL_PEAK_AMP_POLICY_DEFAULT;
}

SpectralPeakModel spectral_peak_model_default(void) {
    return spectral_peak_model_for_window_type(SPECTRAL_WINDOW_HANN);
}

SpectralPeakModel spectral_peak_model_for_window(const SpectralWindowDescriptor* window) {
    SpectralPeakModel model;
    if (!window) {
        window = spectral_window_descriptor(SPECTRAL_WINDOW_HANN);
    }

    model.window = window;
    model.estimator = SPECTRAL_PEAK_ESTIMATOR_DEFAULT;
    model.phase_policy = SPECTRAL_PEAK_PHASE_POLICY_DEFAULT;
    model.amplitude_policy = spectral_peak_model_default_amplitude_for_window(window);
    model.capabilities = 0u;
    model.assumptions = 0u;
    return model;
}

SpectralPeakModel spectral_peak_model_for_window_type(SpectralWindowType type) {
    const SpectralWindowDescriptor* window = spectral_window_descriptor(type);
    return spectral_peak_model_for_window(window);
}

SpectralPeakModel spectral_peak_model_from_resolved(const SpectralResolvedPeakModel* resolved) {
    SpectralPeakModel model = spectral_peak_model_default();
    if (!resolved || !resolved->window) return model;

    model.window = resolved->window;
    model.estimator = resolved->estimator;
    model.phase_policy = resolved->phase_policy;
    model.amplitude_policy = resolved->amplitude_policy;
    model.capabilities = resolved->capabilities;
    model.assumptions = resolved->assumptions;
    return model;
}

SpectralError spectral_peak_model_validate(const SpectralPeakModel* model) {
    SpectralPeakEstimatorType estimator = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;

    if (!model || !model->window) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!model->window->interp_magsq) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!spectral_peak_estimator_type_supported(model->estimator)) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!spectral_peak_phase_policy_supported(model->phase_policy)) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!spectral_peak_amplitude_policy_supported(model->amplitude_policy)) {
        return SPECTRAL_ERR_PARAM;
    }

    estimator = spectral_peak_estimator_resolve_default(model->estimator);
    if (!spectral_peak_estimator_type_supported(estimator) ||
        estimator == SPECTRAL_PEAK_ESTIMATOR_COUNT) {
        return SPECTRAL_ERR_PARAM;
    }

    /* INTERP_BOUNDED is a promise to evaluate a window-owned peak-height model.
     * Custom descriptors that only provide frequency interpolation remain valid
     * by defaulting to center amplitude, but an explicit bounded-interp request
     * without a callback is a contract error. */
    if (model->amplitude_policy == SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED &&
        !model->window->peak_magsq) {
        return SPECTRAL_ERR_PARAM;
    }

    /* Rectangular's Dirichlet/sinc main lobe should not silently inherit the
     * log-parabolic peak-height model. Use center-bin height by default unless
     * a future rectangular-specific callback is deliberately installed. */
    if (model->window->type == SPECTRAL_WINDOW_RECTANGULAR &&
        model->window->peak_magsq == spectral_window_peak_magsq_log_parabolic) {
        return SPECTRAL_ERR_PARAM;
    }

    return SPECTRAL_OK;
}

SpectralError spectral_peak_model_resolve(const SpectralPeakModel* model,
                                          SpectralResolvedPeakModel* out)
{
    SpectralError err = SPECTRAL_OK;
    SpectralPeakEstimatorType estimator = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    unsigned capabilities = 0u;
    unsigned assumptions = 0u;

    if (!out) return SPECTRAL_ERR_PARAM;
    *out = (SpectralResolvedPeakModel){0};

    err = spectral_peak_model_validate(model);
    if (err != SPECTRAL_OK) return err;

    estimator = spectral_peak_estimator_resolve_default(model->estimator);
    capabilities = spectral_peak_model_estimator_capabilities(estimator);
    assumptions = spectral_peak_model_estimator_assumptions(estimator) |
                  spectral_peak_model_window_assumptions(model->window);

    if (model->amplitude_policy == SPECTRAL_PEAK_AMP_POLICY_INTERP_BOUNDED) {
        capabilities |= SPECTRAL_PEAK_MODEL_CAP_PEAK_HEIGHT;
    }
    if (model->phase_policy != SPECTRAL_PEAK_PHASE_POLICY_IGNORE) {
        capabilities |= SPECTRAL_PEAK_MODEL_CAP_PHASE_ROW |
                        SPECTRAL_PEAK_MODEL_CAP_NEXT_PHASE_ROW |
                        SPECTRAL_PEAK_MODEL_CAP_PHASE_DIAGNOSTIC;
        assumptions |= SPECTRAL_PEAK_MODEL_ASSUME_PHASE_VOCODER_ADVANCE;
    }

    capabilities |= SPECTRAL_PEAK_MODEL_CAP_NEXT_MAG_ROW;

    out->window = model->window;
    out->interp_magsq = model->window->interp_magsq
        ? model->window->interp_magsq
        : spectral_window_interp_magsq_parabolic;
    out->peak_magsq = model->window->peak_magsq
        ? model->window->peak_magsq
        : spectral_window_peak_magsq_center;
    out->estimator = estimator;
    out->phase_policy = model->phase_policy;
    out->amplitude_policy = model->amplitude_policy;
    out->capabilities = capabilities;
    out->assumptions = assumptions;
    return SPECTRAL_OK;
}

int spectral_peak_model_has_capability(const SpectralPeakModel* model, unsigned capability) {
    SpectralResolvedPeakModel resolved;
    if (capability == 0u) return 0;
    if (spectral_peak_model_resolve(model, &resolved) != SPECTRAL_OK) return 0;
    return (resolved.capabilities & capability) != 0u;
}
