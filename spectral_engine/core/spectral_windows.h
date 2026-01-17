/* spectral_windows.h - Window Functions for Spectral Analysis
 * 
 * Provides common window functions for FFT analysis and synthesis.
 * All windows are normalized such that their sum equals 1.0 (for analysis)
 * or their RMS equals 1.0 (for synthesis), depending on the function used.
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

/*
 * spectral_window_generate: Generate window by type
 * 
 * Convenience function for selecting window type at runtime.
 */
void spectral_window_generate(float* window, size_t length, SpectralWindowType type);

/*
 * spectral_window_name: Get human-readable window name
 */
const char* spectral_window_name(SpectralWindowType type);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_WINDOWS_H */
