/* spectral_synth.h - Synthesis backend interface */
#ifndef SPECTRAL_SYNTH_H
#define SPECTRAL_SYNTH_H

/* Public desktop synthesis API. SemVer; PRE-RELEASE (0.0.1) — part of the spectral_kernel.h 0.0.1
 * surface (synth_cpu / synth_cpu_wavetable are re-exported there). The surface is guarded
 * (kernel_api_freeze) so changes are deliberate, but it is NOT frozen until the kernel reaches 1.0. */
#define SPECTRAL_SYNTH_API_VERSION_MAJOR 0
#define SPECTRAL_SYNTH_API_VERSION_MINOR 0
#define SPECTRAL_SYNTH_API_VERSION_PATCH 1

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

#ifdef __cplusplus
}
#endif

#endif
