/* spectral_synth.h - Synthesis backend interface */
#ifndef SPECTRAL_SYNTH_H
#define SPECTRAL_SYNTH_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_wavetable.h"

#if defined(SPECTRAL_USE_EMBEDDED_SYNTH)
#include "spectral_synth_embedded.h"
#endif

typedef enum {
    BACKEND_AUTO   = 0,
    BACKEND_CPU    = 1,
    BACKEND_METAL  = 2,
    BACKEND_CUDA   = 3,
    BACKEND_EXPORT = 4
} SynthBackend;

/* Backend timbre limits */
#define BACKEND_CPU_TIMBRE_MAX          TIMBRE_MAX
#define BACKEND_METAL_TIMBRE_MAX        TIMBRE_PARABOLA
#define BACKEND_CUDA_TIMBRE_MAX         TIMBRE_PARABOLA
#define BACKEND_CPU_WAVETABLE_SUPPORT   1
#define BACKEND_METAL_WAVETABLE_SUPPORT 0
#define BACKEND_CUDA_WAVETABLE_SUPPORT  0

/* Backend capability query */
typedef struct {
    SynthBackend id;
    const char*  name;
    int          available;
    int          max_timbre;
    int          has_wavetable;
    int          is_gpu;
    int          is_parallel;
    int          max_segments;
    size_t       max_output_len;
} SpectralBackendCaps;

/* Backend queries */
int spectral_backend_supports_timbre(SynthBackend backend, int timbre_id);
int spectral_backend_max_timbre(SynthBackend backend);
const char* spectral_backend_name(SynthBackend backend);
int spectral_backend_supports_wavetable(SynthBackend backend);
SpectralBackendCaps spectral_backend_get_caps(SynthBackend backend);
SynthBackend spectral_backend_select_for_timbre(int timbre_id, int prefer_gpu);
int spectral_backend_available(SynthBackend backend);

/* CPU synthesis - returns SPECTRAL_OK on success, error code on failure */
SpectralError synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len,
                        float stretch, float pitch, SpectralTimbre timbre, int n_threads, double* t_synth);

SpectralError synth_cpu_wavetable(SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch,
                                  const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                  int n_threads, double* t_synth);

/* Metal backend (macOS) */
#if defined(__APPLE__) && !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
#define HAS_METAL 1
void metal_init(void);
int  metal_available(void);
void synth_metal(SegmentArray sa, float* out_buffer, size_t out_len,
                 float stretch, float pitch, SpectralTimbre timbre, double* t_synth);
#else
#define HAS_METAL 0
#endif

/* CUDA backend (Linux/NVIDIA) */
#if defined(USE_CUDA) && !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
#define HAS_CUDA 1
void cuda_init(void);
int  cuda_available(void);
void synth_cuda(SegmentArray sa, float* out_buffer, size_t out_len,
                float stretch, float pitch, SpectralTimbre timbre, double* t_synth);
#else
#define HAS_CUDA 0
#endif

/* Native sample type synthesis */
void synth_cpu_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                      float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                      double* t_synth);

void synth_cpu_wavetable_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                                float stretch, float pitch,
                                const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                int n_threads, double* t_synth);

#endif
