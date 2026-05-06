/* spectral_windows.c - Window Functions Implementation
 * 
 * Implements common window functions for FFT-based spectral analysis.
 * Uses vDSP on macOS for hardware-accelerated window generation when available.
 */
#include "spectral_windows.h"
#include "spectral_config.h"
#include "spectral_fast_math.h"
#include <math.h>
#include <string.h>

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

void spectral_window_hann(float* window, size_t length) {
    if (!window || length == 0) return;
    if (length == 1) { window[0] = 1.0f; return; }
    
#if SPECTRAL_USE_VDSP
    vDSP_hann_window(window, (vDSP_Length)length, vDSP_HANN_DENORM);
#else
    /* Portable implementation: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1))) */
    const float scale = (float)(2.0 * SPECTRAL_PI / (length - 1));
    for (size_t i = 0; i < length; i++) {
        window[i] = 0.5f * (1.0f - cosf((float)i * scale));
    }
#endif
}

void spectral_window_hamming(float* window, size_t length) {
    if (!window || length == 0) return;
    if (length == 1) { window[0] = 1.0f; return; }
    
#if SPECTRAL_USE_VDSP
    vDSP_hamm_window(window, (vDSP_Length)length, 0);
#else
    /* Hamming: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1)) */
    const float scale = (float)(2.0 * SPECTRAL_PI / (length - 1));
    for (size_t i = 0; i < length; i++) {
        window[i] = 0.54f - 0.46f * cosf((float)i * scale);
    }
#endif
}

void spectral_window_blackman(float* window, size_t length) {
    if (!window || length == 0) return;
    if (length == 1) { window[0] = 1.0f; return; }
    
#if SPECTRAL_USE_VDSP
    vDSP_blkman_window(window, (vDSP_Length)length, 0);
#else
    /* Blackman: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1)) */
    const float scale = (float)(2.0 * SPECTRAL_PI / (length - 1));
    for (size_t i = 0; i < length; i++) {
        float angle = (float)i * scale;
        window[i] = 0.42f - 0.5f * cosf(angle) + 0.08f * cosf(2.0f * angle);
    }
#endif
}

void spectral_window_rectangular(float* window, size_t length) {
    if (!window || length == 0) return;
    
    for (size_t i = 0; i < length; i++) {
        window[i] = 1.0f;
    }
}


int spectral_window_peak_magsq_center(float left_sq, float center_sq, float right_sq,
                                      float bin_offset, float* out_peak_magsq) {
    (void)left_sq;
    (void)right_sq;
    (void)bin_offset;
    if (!out_peak_magsq || !isfinite(center_sq) || center_sq <= 0.0f) return 0;
    *out_peak_magsq = center_sq;
    return 1;
}

int spectral_window_peak_magsq_log_parabolic(float left_sq, float center_sq, float right_sq,
                                             float bin_offset, float* out_peak_magsq) {
    float left = 0.0f;
    float center = 0.0f;
    float right = 0.0f;
    float log_l = 0.0f;
    float log_c = 0.0f;
    float log_r = 0.0f;
    float log_peak = 0.0f;
    float peak = 0.0f;

    if (!out_peak_magsq || !isfinite(bin_offset) ||
        !isfinite(left_sq) || !isfinite(center_sq) || !isfinite(right_sq) ||
        left_sq < 0.0f || center_sq <= 0.0f || right_sq < 0.0f) {
        return 0;
    }

    /* Once a log-power parabola is chosen, the fitted peak height is:
     *   y_peak = y[0] - 0.25 * (y[-1] - y[1]) * p
     * where y = log(power). Whether this model is appropriate is selected by
     * the window descriptor, not by the tracker. */
    left = fmaxf(left_sq, SPECTRAL_TRACK_LOG_FLOOR);
    center = fmaxf(center_sq, SPECTRAL_TRACK_LOG_FLOOR);
    right = fmaxf(right_sq, SPECTRAL_TRACK_LOG_FLOOR);
    log_l = fast_peak_log(left);
    log_c = fast_peak_log(center);
    log_r = fast_peak_log(right);
    log_peak = log_c - 0.25f * (log_l - log_r) * bin_offset;
    if (!isfinite(log_peak)) return 0;

    peak = expf(log_peak);
    if (!isfinite(peak) || peak <= 0.0f) return 0;
    if (peak < center_sq) peak = center_sq;

    *out_peak_magsq = peak;
    return 1;
}


static const SpectralWindowDescriptor spectral_window_descriptors[] = {
    { SPECTRAL_WINDOW_HANN, "hann", "Hann", spectral_window_hann, spectral_window_interp_magsq_parabolic, spectral_window_peak_magsq_log_parabolic },
    { SPECTRAL_WINDOW_HAMMING, "hamming", "Hamming", spectral_window_hamming, spectral_window_interp_magsq_parabolic, spectral_window_peak_magsq_log_parabolic },
    { SPECTRAL_WINDOW_BLACKMAN, "blackman", "Blackman", spectral_window_blackman, spectral_window_interp_magsq_parabolic, spectral_window_peak_magsq_log_parabolic },
    { SPECTRAL_WINDOW_RECTANGULAR, "rectangular", "Rectangular", spectral_window_rectangular, spectral_window_interp_magsq_parabolic, spectral_window_peak_magsq_center }
};

const SpectralWindowDescriptor* spectral_window_descriptor(SpectralWindowType type) {
    size_t count = spectral_window_descriptor_count();
    for (size_t i = 0; i < count; i++) {
        if (spectral_window_descriptors[i].type == type) {
            return &spectral_window_descriptors[i];
        }
    }
    return NULL;
}

const SpectralWindowDescriptor* spectral_window_descriptor_at(size_t index) {
    if (index >= spectral_window_descriptor_count()) return NULL;
    return &spectral_window_descriptors[index];
}

size_t spectral_window_descriptor_count(void) {
    return sizeof(spectral_window_descriptors) / sizeof(spectral_window_descriptors[0]);
}

const SpectralWindowDescriptor* spectral_window_find_by_id(const char* id) {
    size_t count = spectral_window_descriptor_count();
    if (!id) return NULL;
    for (size_t i = 0; i < count; i++) {
        if (spectral_window_descriptors[i].id &&
            strcmp(spectral_window_descriptors[i].id, id) == 0) {
            return &spectral_window_descriptors[i];
        }
    }
    return NULL;
}

void spectral_window_generate(float* window, size_t length, SpectralWindowType type) {
    const SpectralWindowDescriptor* desc = spectral_window_descriptor(type);
    if (!desc || !desc->generate) {
        desc = spectral_window_descriptor(SPECTRAL_WINDOW_RECTANGULAR);
    }
    if (desc && desc->generate) {
        desc->generate(window, length);
    }
}

const char* spectral_window_name(SpectralWindowType type) {
    const SpectralWindowDescriptor* desc = spectral_window_descriptor(type);
    return desc ? desc->display_name : "Unknown";
}

float spectral_window_sum(const float* window, size_t length) {
    if (!window || length == 0) return 0.0f;

    double sum = 0.0;
    for (size_t i = 0; i < length; i++) {
        sum += (double)window[i];
    }
    return (float)sum;
}

float spectral_window_energy(const float* window, size_t length) {
    if (!window || length == 0) return 0.0f;

    double energy = 0.0;
    for (size_t i = 0; i < length; i++) {
        double v = (double)window[i];
        energy += v * v;
    }
    return (float)energy;
}

SpectralWindowMetrics spectral_window_metrics(const float* window, size_t length) {
    SpectralWindowMetrics metrics = {0};

    metrics.positive_bin_amp_scale = 1.0f;
    metrics.positive_bin_magsq_scale = 1.0f;
    metrics.endpoint_bin_amp_scale = 1.0f;
    metrics.endpoint_bin_magsq_scale = 1.0f;

    if (!window || length == 0u) {
        return metrics;
    }

    metrics.sum = spectral_window_sum(window, length);
    metrics.energy = spectral_window_energy(window, length);

    if (isfinite(metrics.sum)) {
        float coherent_gain = metrics.sum / (float)length;
        if (isfinite(coherent_gain)) {
            metrics.coherent_gain = coherent_gain;
        }
    }
    if (isfinite(metrics.energy) && metrics.energy >= 0.0f) {
        float rms_gain = sqrtf(metrics.energy / (float)length);
        if (isfinite(rms_gain)) {
            metrics.rms_gain = rms_gain;
        }
    }

    if (isfinite(metrics.sum) && metrics.sum > 0.0f) {
        double positive_amp_scale_d = 2.0 / (double)metrics.sum;
        double positive_magsq_scale_d = positive_amp_scale_d * positive_amp_scale_d;
        float positive_amp_scale = (float)positive_amp_scale_d;
        float positive_magsq_scale = (float)positive_magsq_scale_d;

        if (isfinite(positive_amp_scale) && positive_amp_scale > 0.0f &&
            isfinite(positive_magsq_scale) && positive_magsq_scale > 0.0f) {
            metrics.positive_bin_amp_scale = positive_amp_scale;
            metrics.positive_bin_magsq_scale = positive_magsq_scale;
            metrics.flags |= SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID;
        }

        {
            double endpoint_amp_scale_d = 1.0 / (double)metrics.sum;
            double endpoint_magsq_scale_d = endpoint_amp_scale_d * endpoint_amp_scale_d;
            float endpoint_amp_scale = (float)endpoint_amp_scale_d;
            float endpoint_magsq_scale = (float)endpoint_magsq_scale_d;

            if (isfinite(endpoint_amp_scale) && endpoint_amp_scale > 0.0f &&
                isfinite(endpoint_magsq_scale) && endpoint_magsq_scale > 0.0f) {
                metrics.endpoint_bin_amp_scale = endpoint_amp_scale;
                metrics.endpoint_bin_magsq_scale = endpoint_magsq_scale;
                metrics.flags |= SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID;
            }
        }
    }

    if (isfinite(metrics.sum) && isfinite(metrics.energy) &&
        metrics.sum > 0.0f && metrics.energy > 0.0f) {
        metrics.enbw_bins = ((float)length * metrics.energy) / (metrics.sum * metrics.sum);
        if (isfinite(metrics.enbw_bins) && metrics.enbw_bins > 0.0f) {
            metrics.flags |= SPECTRAL_WINDOW_METRIC_ENBW_VALID;
        } else {
            metrics.enbw_bins = 0.0f;
        }
    }

    return metrics;
}

float spectral_window_coherent_gain(const float* window, size_t length) {
    return spectral_window_metrics(window, length).coherent_gain;
}

float spectral_window_rms_gain(const float* window, size_t length) {
    return spectral_window_metrics(window, length).rms_gain;
}

float spectral_window_positive_bin_amp_scale(const float* window, size_t length) {
    return spectral_window_metrics(window, length).positive_bin_amp_scale;
}

float spectral_window_positive_bin_magsq_scale(const float* window, size_t length) {
    return spectral_window_metrics(window, length).positive_bin_magsq_scale;
}

float spectral_window_endpoint_bin_amp_scale(const float* window, size_t length) {
    return spectral_window_metrics(window, length).endpoint_bin_amp_scale;
}

float spectral_window_endpoint_bin_magsq_scale(const float* window, size_t length) {
    return spectral_window_metrics(window, length).endpoint_bin_magsq_scale;
}

float spectral_window_interp_magsq_parabolic(float left_sq, float center_sq, float right_sq) {
    /* Correctness-first estimator: log-power quadratic interpolation over
     * adjacent magnitude-squared bins. This is a local spectral-peak model,
     * not an exact MLE; see Smith's spectral-peak interpolation derivation and
     * Harris's window-analysis treatment:
     *   https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html
     *   https://doi.org/10.1109/PROC.1978.10837
     */
#if defined(SPECTRAL_TRACK_INTERP_POWER_RATIONAL) && SPECTRAL_TRACK_INTERP_POWER_RATIONAL
    float denom = left_sq + right_sq + 2.0f * center_sq;
    if (denom < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) return 0.0f;
    float p = (right_sq - left_sq) / denom;
    p *= 1.5f;
#else
    float left = fmaxf(left_sq, SPECTRAL_TRACK_LOG_FLOOR);
    float center = fmaxf(center_sq, SPECTRAL_TRACK_LOG_FLOOR);
    float right = fmaxf(right_sq, SPECTRAL_TRACK_LOG_FLOOR);
    float log_lc = fast_peak_log(left / center);
    float log_rc = fast_peak_log(right / center);
    float denom = log_lc + log_rc;
    float p = 0.0f;

    /* Exact algebraic reduction of Smith's three-log quadratic formula:
     * a = log(left/center), b = log(right/center),
     * p = 0.5 * (a - b) / (a + b). The fallback handles extreme finite
     * power ratios where left/center or right/center overflows to Inf. */
    if (!isfinite(denom) || fabsf(denom) < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) {
        float log_l = fast_peak_log(left);
        float log_c = fast_peak_log(center);
        float log_r = fast_peak_log(right);
        denom = log_l - 2.0f * log_c + log_r;
        if (!isfinite(denom) || fabsf(denom) < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) return 0.0f;
        p = 0.5f * (log_l - log_r) / denom;
    } else {
        p = 0.5f * (log_lc - log_rc) / denom;
    }
#endif
    if (!isfinite(p)) return 0.0f;
    if (p > 0.5f) return 0.5f;
    if (p < -0.5f) return -0.5f;
    return p;
}
