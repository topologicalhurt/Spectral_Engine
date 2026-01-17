/* spectral_windows.c - Window Functions Implementation
 * 
 * Implements common window functions for FFT-based spectral analysis.
 * Uses vDSP on macOS for hardware-accelerated window generation when available.
 */
#include "spectral_windows.h"
#include "spectral_config.h"
#include <math.h>

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

void spectral_window_hann(float* window, size_t length) {
    if (!window || length == 0) return;
    
#if SPECTRAL_USE_VDSP
    vDSP_hann_window(window, (vDSP_Length)length, vDSP_HANN_NORM);
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

void spectral_window_generate(float* window, size_t length, SpectralWindowType type) {
    switch (type) {
        case SPECTRAL_WINDOW_HANN:
            spectral_window_hann(window, length);
            break;
        case SPECTRAL_WINDOW_HAMMING:
            spectral_window_hamming(window, length);
            break;
        case SPECTRAL_WINDOW_BLACKMAN:
            spectral_window_blackman(window, length);
            break;
        case SPECTRAL_WINDOW_RECTANGULAR:
        default:
            spectral_window_rectangular(window, length);
            break;
    }
}

const char* spectral_window_name(SpectralWindowType type) {
    static const char* names[] = {
        "Hann",
        "Hamming", 
        "Blackman",
        "Rectangular"
    };
    if (type < SPECTRAL_WINDOW_COUNT) {
        return names[type];
    }
    return "Unknown";
}
