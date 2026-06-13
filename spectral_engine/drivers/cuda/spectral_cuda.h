/* spectral_cuda.h - CUDA driver public surface.
 *
 * HAS_CUDA is decided HERE, next to the code it gates, not in the core API
 * header. Consumers: the backend dispatch registry (core/spectral_backend.c,
 * the one sanctioned core->driver edge) and cmd userland (capability text,
 * the VRAM info line).
 */
#ifndef SPECTRAL_CUDA_H
#define SPECTRAL_CUDA_H

#include <stddef.h>

#include "spectral_common.h"
#include "spectral_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(USE_CUDA) && !SPECTRAL_EMBEDDED && !SPECTRAL_RESTRICTED_MODE
#define HAS_CUDA 1

void cuda_init(void);
int  cuda_available(void);
size_t cuda_vram_usage_bytes(void);
SpectralError synth_cuda(SegmentArray sa, float* out_buffer, size_t out_len,
                         float stretch, float pitch, SpectralTimbre timbre,
                         double* t_synth);
void cuda_cleanup(void);

#else
#define HAS_CUDA 0
#endif

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_CUDA_H */
