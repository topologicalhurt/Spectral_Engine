/* spectral_windows.h - Window Functions for Spectral Analysis
 * 
 * Provides common window functions for FFT analysis and synthesis.
 * These functions generate conventional, unnormalized sample-domain
 * window shapes. Apply coherent-gain or RMS normalization explicitly at
 * the call site when amplitude calibration requires it.
 * 
 * Supported Windows:
 *   - Hann (raised cosine): Good frequency resolution, moderate leakage
 *   - Hamming: Better sidelobe suppression than Hann
 *   - Blackman: Excellent sidelobe suppression, wider main lobe
 *   - Rectangular: No windowing (equivalent to no window)
 * 
 * Usage:
 *   float* window = malloc(n_fft * sizeof(float));
 *   spectral_window_hann(window, n_fft);
 *   // Apply: for (i=0; i<n_fft; i++) windowed[i] = signal[i] * window[i];
 */
#ifndef SPECTRAL_WINDOWS_H
#define SPECTRAL_WINDOWS_H

#include <stddef.h>
#include "spectral_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Window type enumeration for generic window generation */
typedef enum {
    SPECTRAL_WINDOW_HANN = 0,
    SPECTRAL_WINDOW_HAMMING,
    SPECTRAL_WINDOW_BLACKMAN,
    SPECTRAL_WINDOW_RECTANGULAR,
    SPECTRAL_WINDOW_COUNT
} SpectralWindowType;

typedef enum {
    SPECTRAL_WINDOW_METRIC_POSITIVE_BIN_SCALE_VALID = 1u << 0,
    SPECTRAL_WINDOW_METRIC_ENDPOINT_BIN_SCALE_VALID = 1u << 1,
    SPECTRAL_WINDOW_METRIC_ENBW_VALID = 1u << 2
} SpectralWindowMetricFlags;

typedef void (*SpectralWindowGenerateFn)(float* window, size_t length);
typedef float (*SpectralWindowInterpMagsqFn)(float left_sq, float center_sq, float right_sq);
typedef int (*SpectralWindowPeakMagsqFn)(float left_sq, float center_sq, float right_sq,
                                           float bin_offset, float* out_peak_magsq);

typedef struct {
    SpectralWindowType type;
    const char* id;
    const char* display_name;
    SpectralWindowGenerateFn generate;
    SpectralWindowInterpMagsqFn interp_magsq;
    SpectralWindowPeakMagsqFn peak_magsq;
} SpectralWindowDescriptor;

typedef struct {
    float sum;
    float energy;
    float coherent_gain;
    float rms_gain;
    float enbw_bins;
    float positive_bin_amp_scale;
    float positive_bin_magsq_scale;
    float endpoint_bin_amp_scale;
    float endpoint_bin_magsq_scale;
    unsigned flags;
} SpectralWindowMetrics;

/*
 * spectral_window_hann: Generate a Hann (raised cosine) window
 * 
 * Formula: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
 * 
 * Parameters:
 *   window - Output buffer (must be at least 'length' floats)
 *   length - Window length in samples
 * 
 * The Hann window provides good frequency resolution with moderate
 * sidelobe suppression (-31.5 dB first sidelobe).
 */
void spectral_window_hann(float* window, size_t length);

/*
 * spectral_window_hamming: Generate a Hamming window
 * 
 * Formula: w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
 * 
 * The Hamming window has better sidelobe suppression than Hann (-42 dB)
 * but discontinuous endpoints (doesn't go to zero at edges).
 */
void spectral_window_hamming(float* window, size_t length);

/*
 * spectral_window_blackman: Generate a Blackman window
 * 
 * Formula: w[n] = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
 * 
 * The Blackman window has excellent sidelobe suppression (-58 dB)
 * at the cost of a wider main lobe.
 */
void spectral_window_blackman(float* window, size_t length);

/*
 * spectral_window_rectangular: Generate a rectangular (no) window
 * 
 * All values are 1.0. Equivalent to no windowing.
 * Maximum frequency resolution but severe spectral leakage.
 */
void spectral_window_rectangular(float* window, size_t length);

const SpectralWindowDescriptor* spectral_window_descriptor(SpectralWindowType type);
const SpectralWindowDescriptor* spectral_window_descriptor_at(size_t index);
size_t spectral_window_descriptor_count(void);
const SpectralWindowDescriptor* spectral_window_find_by_id(const char* id);

/*
 * spectral_window_generate: Generate window by type
 * 
 * Convenience function for selecting window type at runtime.
 */
SpectralError spectral_window_generate(float* window, size_t length, SpectralWindowType type);

/*
 * spectral_window_name: Get human-readable window name
 */
const char* spectral_window_name(SpectralWindowType type);

/*
 * Window calibration helpers.
 *
 * coherent_gain = sum(window) / length
 * rms_gain      = sqrt(sum(window^2) / length)
 *
 * For real-valued STFT analysis, a bin-centered sinusoid of peak amplitude A
 * appears in an interior positive-frequency DFT bin with magnitude
 * approximately A * sum(window) / 2. DC and Nyquist are unpaired one-sided
 * endpoints and use A * sum(window) instead. This follows the one-sided
 * spectrum convention used by periodogram implementations and real-DFT
 * endpoint packing:
 *   https://scipy.github.io/devdocs/reference/generated/scipy.signal.periodogram.html
 *   https://www.fftw.org/doc/The-1d-Real_002ddata-DFT.html
 *
 * Window coherent-gain and ENBW definitions follow the conventional window
 * metric treatment summarized by Harris 1978:
 *   https://www.site2241.net/sdr/Use-of-Windows-for-Harmonic-Analysis-Harris-1978.pdf
 */
float spectral_window_sum(const float* window, size_t length);
float spectral_window_energy(const float* window, size_t length);
SpectralWindowMetrics spectral_window_metrics(const float* window, size_t length);
float spectral_window_coherent_gain(const float* window, size_t length);
float spectral_window_rms_gain(const float* window, size_t length);
float spectral_window_positive_bin_amp_scale(const float* window, size_t length);
float spectral_window_positive_bin_magsq_scale(const float* window, size_t length);
float spectral_window_endpoint_bin_amp_scale(const float* window, size_t length);
float spectral_window_endpoint_bin_magsq_scale(const float* window, size_t length);

/*
 * Computes the sub-bin frequency offset for a detected peak based on the
 * power spectrum (magnitude squared), adapted for the window function.
 *
 * Default implementation uses parabolic interpolation on log-power bins:
 *   p = 0.5 * (log(left) - log(right)) /
 *       (log(left) - 2*log(center) + log(right))
 * The output is a dimensionless bin offset clamped to [-0.5, 0.5].
 * This follows the local quadratic spectral-peak model documented by
 * J. O. Smith, Spectral Audio Signal Processing:
 *   https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html
 * Window-shape assumptions follow Harris 1978:
 *   https://doi.org/10.1109/PROC.1978.10837
 * Non-log rational estimators must be separately derived and validated for the
 * exact window and spectrum representation used here.
 */
float spectral_window_interp_magsq_parabolic(float left_sq, float center_sq, float right_sq);
int spectral_window_peak_magsq_center(float left_sq, float center_sq, float right_sq,
                                      float bin_offset, float* out_peak_magsq);
int spectral_window_peak_magsq_log_parabolic(float left_sq, float center_sq, float right_sq,
                                             float bin_offset, float* out_peak_magsq);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_WINDOWS_H */
