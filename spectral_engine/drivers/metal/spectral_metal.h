/* spectral_metal.h - Metal driver public surface.
 *
 * HAS_METAL is decided HERE, next to the code it gates, not in the core API
 * header. Consumers: the backend dispatch registry (core/spectral_backend.c,
 * the one sanctioned core->driver edge) and cmd userland (capability text).
 */
#ifndef SPECTRAL_METAL_H
#define SPECTRAL_METAL_H

#include <stddef.h>

#include "spectral_common.h"
#include "spectral_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__APPLE__) && !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
#define HAS_METAL 1

void metal_init(void);
int  metal_available(void);
SpectralError synth_metal(SegmentArray sa, float* out_buffer, size_t out_len,
                          float stretch, float pitch, SpectralTimbre timbre,
                          double* t_synth);
void metal_cleanup(void);

/* MSL payload, generated from the C synthesis contract into
 * spectral_osc_metal_generated.h and compiled by the payload TU
 * (spectral_osc_metal_payload.c) — only the Metal driver carries the
 * shader strings; no other target holds them as rodata. */
extern const char* gpu_structs_metal_source;
extern const char* oscillator_metal_source;
extern const char* spectral_segment_math_metal_source;

#else
#define HAS_METAL 0
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_METAL_H */
