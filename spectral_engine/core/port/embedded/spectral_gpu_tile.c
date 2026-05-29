/* spectral_gpu_tile.c (embedded profile) - GPU tile preprocessing stub.
 *
 * Build-selected for the embedded/simulation profile, which has no GPU backend
 * (Metal/CUDA). The shared gpu_tile_preprocess() declaration in
 * spectral_synth_internal.h is satisfied here by a stub that reports the
 * capability is unavailable, so the device-agnostic wrapper
 * gpu_tile_preprocess_cached() can call it unconditionally. The real host
 * implementation lives in core/port/host/spectral_gpu_tile.c. This mirrors the
 * former #else (SPECTRAL_ERR_BACKEND_UNAVAIL) arm of the wrapper's profile #if.
 */

#include "spectral_synth_internal.h"

SpectralError gpu_tile_preprocess(
    SegmentArray sa, float stretch, uint32_t tile_size, size_t out_len,
    GpuTileData* out)
{
    (void)sa;
    (void)stretch;
    (void)tile_size;
    (void)out_len;
    if (out) *out = (GpuTileData){0};
    return SPECTRAL_ERR_BACKEND_UNAVAIL;
}
