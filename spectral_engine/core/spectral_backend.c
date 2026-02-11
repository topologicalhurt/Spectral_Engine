/* spectral_backend.c - Backend Query Functions and Synthesis Dispatch
 *
 * Provides backend capability queries and unified dispatch across
 * CPU, Metal, and CUDA synthesis backends.
 */

/* Include spectral_synth.h for HAS_METAL/HAS_CUDA definitions and
 * forward declarations of synth_cpu, synth_metal, synth_cuda, etc. */
#include "spectral_synth.h"
#include "spectral_synth_internal.h"

/* Backend queries */

int spectral_backend_supports_timbre(SynthBackend backend, int timbre_id) {
    switch (backend) {
        case BACKEND_CPU:   return (timbre_id >= TIMBRE_MIN && timbre_id <= BACKEND_CPU_TIMBRE_MAX);
        case BACKEND_METAL: return (timbre_id >= TIMBRE_MIN && timbre_id <= BACKEND_METAL_TIMBRE_MAX);
        case BACKEND_CUDA:  return (timbre_id >= TIMBRE_MIN && timbre_id <= BACKEND_CUDA_TIMBRE_MAX);
        default:            return 1;
    }
}

int spectral_backend_max_timbre(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:   return BACKEND_CPU_TIMBRE_MAX;
        case BACKEND_METAL: return BACKEND_METAL_TIMBRE_MAX;
        case BACKEND_CUDA:  return BACKEND_CUDA_TIMBRE_MAX;
        default:            return TIMBRE_MAX;
    }
}

const char* spectral_backend_name(SynthBackend backend) {
    static const char* names[] = {"Auto", "CPU", "Metal", "CUDA", "Export"};
    return (backend <= BACKEND_EXPORT) ? names[backend] : "Unknown";
}

int spectral_backend_supports_wavetable(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:   return BACKEND_CPU_WAVETABLE_SUPPORT;
        case BACKEND_METAL: return BACKEND_METAL_WAVETABLE_SUPPORT;
        case BACKEND_CUDA:  return BACKEND_CUDA_WAVETABLE_SUPPORT;
        default:            return 0;
    }
}

int spectral_backend_available(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:
            return 1;
        case BACKEND_METAL:
            #if HAS_METAL
            return metal_available();
            #else
            return 0;
            #endif
        case BACKEND_CUDA:
            #if HAS_CUDA
            return cuda_available();
            #else
            return 0;
            #endif
        case BACKEND_EXPORT:
            return 1;
        case BACKEND_AUTO:
            return 1;
        default:
            return 0;
    }
}

SpectralBackendCaps spectral_backend_get_caps(SynthBackend backend) {
    SpectralBackendCaps caps = {0};
    caps.id = backend;
    caps.name = spectral_backend_name(backend);
    caps.available = spectral_backend_available(backend);
    caps.max_timbre = spectral_backend_max_timbre(backend);
    caps.has_wavetable = spectral_backend_supports_wavetable(backend);

    switch (backend) {
        case BACKEND_CPU:
            caps.is_parallel = 1;
            break;
        case BACKEND_METAL:
            caps.is_gpu = 1;
            caps.is_parallel = 1;
            break;
        case BACKEND_CUDA:
            caps.is_gpu = 1;
            caps.is_parallel = 1;
            break;
        default:
            break;
    }
    return caps;
}

SynthBackend spectral_backend_select_for_timbre(int timbre_id, int prefer_gpu) {
    (void)timbre_id; /* used conditionally by HAS_METAL/HAS_CUDA */
    if (prefer_gpu) {
        /* Try Metal first (more timbre support than CUDA) */
        #if HAS_METAL
        if (metal_available() && spectral_backend_supports_timbre(BACKEND_METAL, timbre_id)) {
            return BACKEND_METAL;
        }
        #endif
        #if HAS_CUDA
        if (cuda_available() && spectral_backend_supports_timbre(BACKEND_CUDA, timbre_id)) {
            return BACKEND_CUDA;
        }
        #endif
    }
    /* Fall back to CPU which supports all timbres */
    return BACKEND_CPU;
}

/* Unified synthesis dispatch - handles backend selection, timbre checks, GPU fallback */
SpectralError spectral_synth_dispatch(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    SynthBackend backend, const SpectralWavetableBank* bank,
    int n_threads, double* t_synth)
{
    /* CPU wavetable helper */
    #define CPU_SYNTH_WITH_WT() \
        (bank ? synth_cpu_wavetable(sa, out_buffer, out_len, stretch, pitch, bank, timbre, n_threads, t_synth) \
              : synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth))

#if HAS_METAL
    if (backend != BACKEND_CPU && backend != BACKEND_CUDA) metal_init();
    int metal_supports = spectral_backend_supports_timbre(BACKEND_METAL, (int)timbre);
    int use_metal = (backend == BACKEND_METAL) ||
                    (backend == BACKEND_AUTO && metal_available() && metal_supports);

    if (use_metal && metal_available()) {
        SpectralError err = synth_metal(sa, out_buffer, out_len, stretch, pitch, timbre, t_synth);
        if (err != SPECTRAL_OK) {
            return CPU_SYNTH_WITH_WT();
        }
        return SPECTRAL_OK;
    }
    return CPU_SYNTH_WITH_WT();

#elif HAS_CUDA
    if (backend == BACKEND_CUDA || backend == BACKEND_AUTO) {
        cuda_init();
        if (cuda_available() && spectral_backend_supports_timbre(BACKEND_CUDA, (int)timbre) && !bank) {
            return synth_cuda(sa, out_buffer, out_len, stretch, pitch, timbre, t_synth);
        }
    }
    return CPU_SYNTH_WITH_WT();

#else
    (void)backend;
    return CPU_SYNTH_WITH_WT();
#endif

    #undef CPU_SYNTH_WITH_WT
}
