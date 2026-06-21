/* spectral_gpu_tile.c (embedded profile) - GPU tile preprocessing stub.
 *
 * The embedded/simulation profile has no GPU backend, so it satisfies the shared
 * gpu_tile_preprocess() declaration (spectral_synth_internal.h) with a stub that
 * reports the capability unavailable -- letting gpu_tile_preprocess_cached() call
 * it unconditionally. The real host implementation is arch/simd/spectral_gpu_tile.c.
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
