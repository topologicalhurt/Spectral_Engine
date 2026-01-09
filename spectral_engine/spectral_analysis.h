/* spectral_analysis.h - FFT and peak tracking */

#ifndef SPECTRAL_ANALYSIS_H
#define SPECTRAL_ANALYSIS_H

#include "spectral_common.h"

SegmentArray analyze_audio(const float* audio, size_t n_samples, int sr, 
                           int n_fft, int hop, float db_thresh,
                           double* t_fft, double* t_track);

#endif
