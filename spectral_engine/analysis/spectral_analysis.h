/* spectral_analysis.h - FFT-Based Spectral Analysis
 * 
 * Performs Short-Time Fourier Transform (STFT) analysis on audio input,
 * extracts spectral peaks above a threshold, and tracks them across frames
 * to create "segments" representing individual sinusoidal components.
 * 
 * The output SegmentArray can be used directly for resynthesis or saved
 */

/* Internal chunks for large file STFT analysis to preserve memory */
#ifndef SPECTRAL_ANALYSIS_H
#define SPECTRAL_ANALYSIS_H

#include "spectral_common.h"

#ifdef __cplusplus
extern "C" {
#endif

SegmentArray analyze_audio(const float* audio, size_t n_samples, int sr, 
                           int n_fft, int hop, float db_thresh,
                           double* t_fft, double* t_track);

#ifdef __cplusplus
}
#endif

#endif
