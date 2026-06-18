/* spectral_analysis_internal.h
 *
 * Internal shared interfaces for analysis path implementations.
 * Not part of the public API.
 */
#ifndef SPECTRAL_ANALYSIS_INTERNAL_H
#define SPECTRAL_ANALYSIS_INTERNAL_H

#include "spectral_analysis.h"
#include "spectral_log.h"
#include "spectral_peak_track.h"
#include "spectral_utils.h"
#include "spectral_vector_ops.h"
#include "spectral_windows.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "spectral_omp.h"

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#else
#include <fftw3.h>
#endif

typedef struct {
    int n_threads;
    size_t n_fft;
    size_t n_freqs;
    float endpoint_bin_magsq_scale;
    float positive_bin_magsq_scale;
#if SPECTRAL_USE_VDSP
    vDSP_Length log2n;
    FFTSetup* fft_setups;
    float** thread_real;
    float** thread_imag;
    float** thread_windowed;
    float** thread_imag_sq;
#else
    fftwf_plan* fft_plans;
    float** thread_in;
    fftwf_complex** thread_out;
#endif
} SpectralFftResources;

typedef struct SpectralAnalysisWindowContext {
    float* samples;
    size_t bytes;
    SpectralWindowMetrics metrics;
    const SpectralWindowDescriptor* descriptor;
} SpectralAnalysisWindowContext;

SpectralError spectral_analysis_window_context_init(SpectralAnalysisWindowContext* ctx,
                                                    size_t n_fft,
                                                    SpectralWindowType type);
void spectral_analysis_window_context_free(SpectralAnalysisWindowContext* ctx);
void spectral_analysis_window_context_apply_magsq_scales(const SpectralAnalysisWindowContext* ctx,
                                                         SpectralFftResources* res);

SegmentArray spectral_analysis_return_empty(double* t_fft, double* t_track);

typedef enum SpectralAnalysisPathMode {
    SPECTRAL_ANALYSIS_PATH_AUTO = 0,
    SPECTRAL_ANALYSIS_PATH_FULL = 1,
    SPECTRAL_ANALYSIS_PATH_FUSED = 2
} SpectralAnalysisPathMode;

typedef struct SpectralAnalysisPathDecision {
    size_t total_bins;
    size_t threshold_bins;
    SpectralAnalysisPathMode mode;
    int use_fused_path;
    const char* name;
} SpectralAnalysisPathDecision;

SpectralAnalysisPathDecision spectral_analysis_path_decide(size_t total_bins,
                                                           SpectralAnalysisPathMode mode);

typedef struct SpectralAnalysisShape {
    size_t n_fft_size;
    size_t n_frames;
    size_t n_freqs;
    size_t total_bins;
    SpectralAnalysisPathDecision path;
} SpectralAnalysisShape;

SpectralError spectral_analysis_shape_init(SpectralAnalysisShape* shape,
                                           size_t n_samples,
                                           int sr,
                                           int n_fft,
                                           int hop,
                                           float db_thresh);
const char* spectral_analysis_path_name(int use_fused_path);

typedef struct SpectralAnalysisStftMatrix {
    float* magsq;
    /* re/im hold the (0.5-scaled, on vDSP) complex spectrum. Phase is computed
     * lazily via atan2f at the tracked peak bins only (in the tracker), not
     * densely here — retiring the all-bins producer atan2. magsq stays the
     * double-scaled peak-scan key (parity), so this is +50% vs the old
     * magsq+phase layout on the full-matrix path. See
     * docs/core_audit/ANALYSIS_PHASE_AT_PEAKS_PLAN.md. */
    SpectralHalf* re;
    SpectralHalf* im;
    size_t total_bins;
    size_t total_bytes;
} SpectralAnalysisStftMatrix;

SpectralError spectral_analysis_stft_matrix_alloc(SpectralAnalysisStftMatrix* matrix,
                                                  size_t n_frames,
                                                  size_t n_freqs);
void spectral_analysis_stft_matrix_free(SpectralAnalysisStftMatrix* matrix);

int spectral_analysis_estimate_fft_bytes(size_t frame_count,
                                         size_t n_fft,
                                         size_t n_freqs,
                                         size_t* out_bytes);

int spectral_fft_resources_alloc(SpectralFftResources* res, int n_threads,
                                 size_t n_fft, size_t n_freqs);
void spectral_fft_resources_free(SpectralFftResources* res);
void spectral_fft_resources_set_magsq_scales(SpectralFftResources* res,
                                             float endpoint_bin_magsq_scale,
                                             float positive_bin_magsq_scale);
float spectral_fft_frames(const SpectralFftResources* res,
                          const float* audio, int hop,
                          const float* window_func,
                          size_t frame_start, size_t frame_end,
                          float* out_magsq, SpectralHalf* out_re, SpectralHalf* out_im,
                          int magsq_only);

void spectral_fft_single_frame(const SpectralFftResources* res,
                               int tid,
                               const float* audio, int hop,
                               const float* window_func,
                               size_t t,
                               float* out_magsq, SpectralHalf* out_re, SpectralHalf* out_im,
                               float* out_frame_max);

SegmentArray analyze_audio_with_path_mode(const float* audio, size_t n_samples, int sr,
                                        int n_fft, int hop, float db_thresh,
                                        SpectralAnalysisPathMode path_mode,
                                        double* t_fft, double* t_track);

SegmentArray spectral_analysis_run_full(const float* audio, size_t n_samples,
                                        int sr, int n_fft, int hop, float db_thresh,
                                        size_t n_frames, size_t n_freqs,
                                        double* t_fft, double* t_track);

SegmentArray spectral_analysis_run_fused(const float* audio, size_t n_samples,
                                         int sr, int n_fft, int hop, float db_thresh,
                                         size_t n_frames, size_t n_freqs,
                                         double* t_fft, double* t_track);

#endif /* SPECTRAL_ANALYSIS_INTERNAL_H */
