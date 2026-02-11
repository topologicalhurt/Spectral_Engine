/* spectral_backend.h - Synthesis Backend Query and Dispatch API
 *
 * Backend capability queries, timbre support checking, and unified
 * synthesis dispatch across CPU, Metal, and CUDA backends.
 */
#ifndef SPECTRAL_BACKEND_H
#define SPECTRAL_BACKEND_H

#include "spectral_common.h"
#include "spectral_config.h"
#include "spectral_wavetable.h"

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

/* Unified synthesis dispatch - handles backend selection, timbre checks, fallback */
SpectralError spectral_synth_dispatch(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    SynthBackend backend, const SpectralWavetableBank* bank,
    int n_threads, double* t_synth);

#endif /* SPECTRAL_BACKEND_H */
