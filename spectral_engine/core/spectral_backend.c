/* spectral_backend.c - Backend Query Functions and Synthesis Dispatch
 *
 * Provides backend capability queries and unified dispatch across
 * CPU, Metal, and CUDA synthesis backends via a vtable.
 */

/* Include spectral_synth.h for HAS_METAL/HAS_CUDA definitions and
 * forward declarations of synth_cpu, synth_metal, synth_cuda, etc. */
#include "spectral_synth.h"
#include "spectral_synth_internal.h"

static SpectralError cpu_synth_vtable(SegmentArray sa, float* buf, size_t len,
                                       float stretch, float pitch,
                                       SpectralTimbre timbre, double* t_synth) {
    return synth_cpu(sa, buf, len, stretch, pitch, timbre, 1, t_synth);
}

static void noop_init(void) {}
static int  always_available(void) { return 1; }
static void noop_cleanup(void) {}

/* Static vtable entries */

static const SpectralBackendVTable vtable_cpu = {
    BACKEND_CPU, "CPU", BACKEND_CPU_TIMBRE_MAX, 1, 0,
    noop_init, always_available, cpu_synth_vtable, noop_cleanup
};

#if HAS_METAL
static const SpectralBackendVTable vtable_metal = {
    BACKEND_METAL, "Metal", BACKEND_METAL_TIMBRE_MAX, 0, 1,
    metal_init, metal_available, synth_metal, metal_cleanup
};
#endif

#if HAS_CUDA
static const SpectralBackendVTable vtable_cuda = {
    BACKEND_CUDA, "CUDA", BACKEND_CUDA_TIMBRE_MAX, 0, 1,
    cuda_init, cuda_available, synth_cuda, cuda_cleanup
};
#endif

/* Fallback vtable for unknown/virtual backends (Auto, Export) */
static const SpectralBackendVTable vtable_fallback = {
    BACKEND_AUTO, "Auto", TIMBRE_MAX, 0, 0,
    noop_init, always_available, cpu_synth_vtable, noop_cleanup
};

const SpectralBackendVTable* spectral_backend_vtable(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:   return &vtable_cpu;
#if HAS_METAL
        case BACKEND_METAL: return &vtable_metal;
#endif
#if HAS_CUDA
        case BACKEND_CUDA:  return &vtable_cuda;
#endif
        default:            return &vtable_fallback;
    }
}

/* Backend queries — collapsed to vtable lookups */

int spectral_backend_supports_timbre(SynthBackend backend, int timbre_id) {
    return timbre_id >= TIMBRE_MIN && timbre_id <= spectral_backend_vtable(backend)->max_timbre;
}

int spectral_backend_max_timbre(SynthBackend backend) {
    return spectral_backend_vtable(backend)->max_timbre;
}

const char* spectral_backend_name(SynthBackend backend) {
    static const char* names[] = {"Auto", "CPU", "Metal", "CUDA", "Export"};
    unsigned idx = (unsigned)backend;
    return (idx <= BACKEND_EXPORT) ? names[idx] : "Unknown";
}

int spectral_backend_supports_wavetable(SynthBackend backend) {
    return spectral_backend_vtable(backend)->has_wavetable;
}

int spectral_backend_available(SynthBackend backend) {
    if (backend == BACKEND_AUTO || backend == BACKEND_EXPORT) return 1;
    return spectral_backend_vtable(backend)->available();
}

SpectralBackendCaps spectral_backend_get_caps(SynthBackend backend) {
    const SpectralBackendVTable* vt = spectral_backend_vtable(backend);
    SpectralBackendCaps caps = {0};
    caps.id = backend;
    caps.name = spectral_backend_name(backend);
    caps.available = spectral_backend_available(backend);
    caps.max_timbre = vt->max_timbre;
    caps.has_wavetable = vt->has_wavetable;
    caps.is_gpu = vt->is_gpu;
    caps.is_parallel = 1;
    return caps;
}

SynthBackend spectral_backend_select_for_timbre(int timbre_id, int prefer_gpu) {
    (void)timbre_id; /* used conditionally by HAS_METAL/HAS_CUDA */
    if (prefer_gpu) {
#if HAS_METAL
        vtable_metal.init();
        if (vtable_metal.available() && spectral_backend_supports_timbre(BACKEND_METAL, timbre_id))
            return BACKEND_METAL;
#endif
#if HAS_CUDA
        vtable_cuda.init();
        if (vtable_cuda.available() && spectral_backend_supports_timbre(BACKEND_CUDA, timbre_id))
            return BACKEND_CUDA;
#endif
    }
    return BACKEND_CPU;
}

/* Unified synthesis dispatch — vtable-driven with CPU fallback */
SpectralError spectral_synth_dispatch(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    SynthBackend backend, const SpectralWavetableBank* bank,
    int n_threads, double* t_synth)
{
    if (n_threads < 1) n_threads = 1;

    if (backend == BACKEND_AUTO || backend == BACKEND_EXPORT)
        backend = spectral_backend_select_for_timbre((int)timbre, 1);

    if (backend == BACKEND_CPU) {
        return bank
            ? synth_cpu_wavetable(sa, out_buffer, out_len, stretch, pitch, bank, timbre, n_threads, t_synth)
            : synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }

    const SpectralBackendVTable* vt = spectral_backend_vtable(backend);
    vt->init();

    if (vt->available() && spectral_backend_supports_timbre(backend, (int)timbre)) {
        /* GPU backends don't support wavetable — skip if bank provided */
        if (!vt->is_gpu || !bank) {
            SpectralError err = vt->synth(sa, out_buffer, out_len, stretch, pitch, timbre, t_synth);
            if (err == SPECTRAL_OK) return SPECTRAL_OK;
        }
    }
    /* Fallback to CPU */
    return bank
        ? synth_cpu_wavetable(sa, out_buffer, out_len, stretch, pitch, bank, timbre, n_threads, t_synth)
        : synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
}
