/* spectral_backend.c - Backend Query Functions and Synthesis Dispatch
 *
 * Provides backend capability queries and unified dispatch across
 * CPU, Metal, and CUDA synthesis backends via a vtable.
 */

/* spectral_synth.h declares the CPU entry points; the driver headers carry
 * each backend's HAS_* capability and externs. This registry TU is the ONE
 * sanctioned core->driver include edge (the static vtable table must name
 * the driver symbols; everything else reaches them through the vtable). */
#include "spectral_synth.h"
#include "spectral_backend.h"
#include "spectral_synth_internal.h"
#include "spectral_metal.h"
#include "spectral_cuda.h"
#include "spectral_utils.h"
#include "spectral_log.h"
#include <stdio.h>

static SpectralError cpu_synth_vtable(SegmentArray sa, float* buf, size_t len,
                                       float stretch, float pitch,
                                       SpectralTimbre timbre, double* t_synth) {
    return synth_cpu(sa, buf, len, stretch, pitch, timbre, 1, t_synth);
}

static void noop_init(void) {}
static int  always_available(void) { return 1; }
static void noop_cleanup(void) {}

static void spectral_log_backend_resolution(
    const char* event,
    SynthBackend requested_backend,
    SynthBackend effective_backend,
    const char* reason)
{
    SpectralResolutionContext context = {0};
    char resolution[384] = {0};

    context.event = event;
    context.backend = SPECTRAL_RESOLUTION_BACKEND_DISPATCH;
    context.scope = SPECTRAL_RESOLUTION_SCOPE_BACKEND;
    context.requested_label = spectral_backend_name(requested_backend);
    context.requested_id = (int)requested_backend;
    context.effective_label = spectral_backend_name(effective_backend);
    context.effective_id = (int)effective_backend;
    context.max_supported_id = -1;
    context.mode = spectral_exec_mode_name();
    context.reason = reason;
    spectral_format_resolution_context(resolution, sizeof(resolution), &context);

    if (requested_backend == BACKEND_AUTO || requested_backend == BACKEND_EXPORT) {
        SPECTRAL_LOG_INFO("%s", resolution);
    } else {
        SPECTRAL_LOG_WARN("%s", resolution);
    }
}

/* Static vtable entries */

static const SpectralBackendVTable vtable_cpu = {
    BACKEND_CPU, "CPU", TIMBRE_PWM, BACKEND_CPU_WAVETABLE_SUPPORT, 0,
    noop_init, always_available, cpu_synth_vtable, noop_cleanup
};

#if HAS_METAL
static const SpectralBackendVTable vtable_metal = {
    BACKEND_METAL, "Metal", SPECTRAL_GPU_MAX_TIMBRE, BACKEND_METAL_WAVETABLE_SUPPORT, 1,
    metal_init, metal_available, synth_metal, metal_cleanup
};
#endif

#if HAS_CUDA
static const SpectralBackendVTable vtable_cuda = {
    BACKEND_CUDA, "CUDA", SPECTRAL_GPU_MAX_TIMBRE, BACKEND_CUDA_WAVETABLE_SUPPORT, 1,
    cuda_init, cuda_available, synth_cuda, cuda_cleanup
};
#endif

/* Fallback vtable for unknown/virtual backends (Auto, Export) */
static const SpectralBackendVTable vtable_fallback = {
    BACKEND_AUTO, "Auto", TIMBRE_PWM, 0, 0,
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

void spectral_backend_cleanup_all(void) {
    /* Real backends only: AUTO/EXPORT are virtual entries sharing the
     * fallback vtable, and a backend missing from this build resolves to
     * the fallback's no-op cleanup. */
    static const SynthBackend real[] = { BACKEND_CPU, BACKEND_METAL, BACKEND_CUDA };
    for (size_t i = 0; i < sizeof(real) / sizeof(real[0]); i++) {
        spectral_backend_vtable(real[i])->cleanup();
    }
}

/* Backend queries — collapsed to vtable lookups */

int spectral_backend_supports_timbre(SynthBackend backend, int timbre_id) {
    if (timbre_id < TIMBRE_SINE) return 0;
    const SpectralBackendVTable* vt = spectral_backend_vtable(backend);
    /* BACKEND_AUTO/EXPORT use virtual fallback entries by design. */
    if (backend != BACKEND_AUTO && backend != BACKEND_EXPORT && vt->id != backend) return 0;
    return timbre_id <= vt->max_timbre;
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
    const SpectralBackendVTable* vt = spectral_backend_vtable(backend);
    if (vt->id != backend) return 0;
    return vt->available();
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
static SpectralError dispatch_cpu_fallback(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    const SpectralWavetableBank* bank,
    int n_threads, double* t_synth,
    SynthBackend* out_effective_backend,
    SpectralTimbre* out_effective_timbre)
{
    SpectralError err;
    if (out_effective_backend) *out_effective_backend = BACKEND_CPU;
    err = bank
        ? synth_cpu_wavetable(sa, out_buffer, out_len, stretch, pitch, bank, timbre, n_threads, t_synth)
        : synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    if (out_effective_timbre) *out_effective_timbre = synth_effective_timbre_get();
    return err;
}

SpectralError spectral_synth_dispatch_ex(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    SynthBackend backend, const SpectralWavetableBank* bank,
    int n_threads, double* t_synth,
    SynthBackend* out_effective_backend,
    SpectralTimbre* out_effective_timbre)
{
    SynthBackend requested_backend = backend;
    char reason[192] = {0};

    if (n_threads < 1) n_threads = 1;
    synth_effective_timbre_reset(timbre);
    if (out_effective_timbre) *out_effective_timbre = timbre;

    if (backend == BACKEND_AUTO || backend == BACKEND_EXPORT)
        backend = spectral_backend_select_for_timbre((int)timbre, 1);

    if (backend != requested_backend) {
        spectral_log_backend_resolution(
            SPECTRAL_RESOLUTION_EVENT_RESOLUTION,
            requested_backend,
            backend,
            SPECTRAL_RESOLUTION_REASON_AUTO_BACKEND);
    }

    if (backend == BACKEND_CPU) {
        return dispatch_cpu_fallback(sa, out_buffer, out_len, stretch, pitch, timbre,
                                     bank, n_threads, t_synth, out_effective_backend, out_effective_timbre);
    }

    const SpectralBackendVTable* vt = spectral_backend_vtable(backend);
    /* Unsupported compile-time backend (e.g., CUDA requested in non-CUDA build). */
    if (vt->id != backend) {
        spectral_log_backend_resolution(
            SPECTRAL_RESOLUTION_EVENT_FALLBACK,
            requested_backend,
            BACKEND_CPU,
            SPECTRAL_RESOLUTION_REASON_BACKEND_NOT_COMPILED);
        return dispatch_cpu_fallback(sa, out_buffer, out_len, stretch, pitch, timbre,
                                     bank, n_threads, t_synth, out_effective_backend, out_effective_timbre);
    }
    vt->init();

    if (!vt->available()) {
        spectral_log_backend_resolution(
            SPECTRAL_RESOLUTION_EVENT_FALLBACK,
            requested_backend,
            BACKEND_CPU,
            SPECTRAL_RESOLUTION_REASON_BACKEND_UNAVAILABLE);
        return dispatch_cpu_fallback(sa, out_buffer, out_len, stretch, pitch, timbre,
                                     bank, n_threads, t_synth, out_effective_backend, out_effective_timbre);
    }

    if (!spectral_backend_supports_timbre(backend, (int)timbre)) {
        snprintf(reason, sizeof(reason),
                 "requested timbre unsupported by backend (timbre=%s/%d)",
                 timbre_name(timbre), (int)timbre);
        spectral_log_backend_resolution(SPECTRAL_RESOLUTION_EVENT_FALLBACK, requested_backend, BACKEND_CPU, reason);
        return dispatch_cpu_fallback(sa, out_buffer, out_len, stretch, pitch, timbre,
                                     bank, n_threads, t_synth, out_effective_backend, out_effective_timbre);
    }

    /* GPU backends don't support wavetable — force CPU path when bank is provided. */
    if (vt->is_gpu && bank) {
        spectral_log_backend_resolution(
            SPECTRAL_RESOLUTION_EVENT_FALLBACK,
            requested_backend,
            BACKEND_CPU,
            SPECTRAL_RESOLUTION_REASON_WAVETABLE_CPU_ONLY);
        return dispatch_cpu_fallback(sa, out_buffer, out_len, stretch, pitch, timbre,
                                     bank, n_threads, t_synth, out_effective_backend, out_effective_timbre);
    }

    {
        SpectralError err = vt->synth(sa, out_buffer, out_len, stretch, pitch, timbre, t_synth);
        if (err == SPECTRAL_OK) {
            if (out_effective_backend) *out_effective_backend = backend;
            if (out_effective_timbre) *out_effective_timbre = synth_effective_timbre_get();
            return SPECTRAL_OK;
        }
        snprintf(reason, sizeof(reason),
                 "backend synth call failed (code=%d, message=%s)",
                 (int)err,
                 spectral_error_message(SPECTRAL_ERROR_DOMAIN_CORE, (int)err));
        spectral_log_backend_resolution(SPECTRAL_RESOLUTION_EVENT_FALLBACK, requested_backend, BACKEND_CPU, reason);
    }

    /* Fallback to CPU */
    return dispatch_cpu_fallback(sa, out_buffer, out_len, stretch, pitch, timbre,
                                 bank, n_threads, t_synth, out_effective_backend, out_effective_timbre);
}

SpectralError spectral_synth_dispatch(
    SegmentArray sa, float* out_buffer, size_t out_len,
    float stretch, float pitch, SpectralTimbre timbre,
    SynthBackend backend, const SpectralWavetableBank* bank,
    int n_threads, double* t_synth, SynthBackend* out_effective_backend)
{
    return spectral_synth_dispatch_ex(sa, out_buffer, out_len, stretch, pitch, timbre,
                                      backend, bank, n_threads, t_synth,
                                      out_effective_backend, NULL);
}
