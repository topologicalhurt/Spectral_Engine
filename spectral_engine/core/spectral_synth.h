/* spectral_synth.h - Synthesis backend interface */
#ifndef SPECTRAL_SYNTH_H
#define SPECTRAL_SYNTH_H

/* Public desktop synthesis API. SemVer; FROZEN as part of the spectral_kernel.h 1.0 surface
 * (synth_cpu / synth_cpu_wavetable are re-exported there). A breaking change requires a MAJOR
 * bump here + a kernel_api_freeze update. */
#define SPECTRAL_SYNTH_API_VERSION_MAJOR 1
#define SPECTRAL_SYNTH_API_VERSION_MINOR 0
#define SPECTRAL_SYNTH_API_VERSION_PATCH 0

#include "spectral_common.h"
#include "spectral_error.h"

/* Forward declaration keeps this public API header lightweight.
 * Include spectral_wavetable.h in implementation code that needs full layout. */
typedef struct SpectralWavetableBank SpectralWavetableBank;

#ifdef __cplusplus
extern "C" {
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
