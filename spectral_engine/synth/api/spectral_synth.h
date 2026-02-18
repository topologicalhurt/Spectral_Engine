/* spectral_synth.h - Synthesis backend interface */
#ifndef SPECTRAL_SYNTH_H
#define SPECTRAL_SYNTH_H

#include "spectral_common.h"
#include "spectral_error.h"

/* Forward declaration keeps this public API header lightweight.
 * Include spectral_wavetable.h in implementation code that needs full layout. */
typedef struct SpectralWavetableBank SpectralWavetableBank;

#ifdef __cplusplus
extern "C" {
#endif

#if defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#include "spectral_synth_arm32.h"
#endif

/* Metal backend (macOS) */
#if defined(__APPLE__) && !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
#define HAS_METAL 1
void metal_init(void);
int  metal_available(void);
SpectralError synth_metal(SegmentArray sa, float* out_buffer, size_t out_len,
                          float stretch, float pitch, SpectralTimbre timbre, double* t_synth);
void metal_cleanup(void);
#else
#define HAS_METAL 0
#endif

/* CUDA backend (Linux/NVIDIA) */
#if defined(USE_CUDA) && !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
#define HAS_CUDA 1
void cuda_init(void);
int  cuda_available(void);
SpectralError synth_cuda(SegmentArray sa, float* out_buffer, size_t out_len,
                         float stretch, float pitch, SpectralTimbre timbre, double* t_synth);
void cuda_cleanup(void);
#else
#define HAS_CUDA 0
#endif

/* CPU synthesis - returns SPECTRAL_OK on success, error code on failure */
SpectralError synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len,
                        float stretch, float pitch, SpectralTimbre timbre, int n_threads, double* t_synth);

SpectralError synth_cpu_wavetable(SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch,
                                  const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                  int n_threads, double* t_synth);

/* Native sample type synthesis */
SpectralError synth_cpu_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                               float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                               double* t_synth);

SpectralError synth_cpu_wavetable_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                                         float stretch, float pitch,
                                         const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                         int n_threads, double* t_synth);

#ifdef __cplusplus
}
#endif

#endif
