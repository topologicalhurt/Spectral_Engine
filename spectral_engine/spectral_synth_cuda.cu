/*
 * spectral_synth_cuda.cu - CUDA tile-parallel synthesis backend
 *
 * Uses shared memory segment caching and tile-parallel dispatch,
 * matching the Metal backend architecture. No atomicAdd needed.
 *
 * Legacy segment-parallel kernel preserved in legacy_spectral_synth_cuda.cu.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "spectral_common.h"
#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include "oscillator.h"

#define TILE_SIZE           SPECTRAL_GPU_TILE_SIZE
#define SEGMENT_CACHE_SIZE  SPECTRAL_METAL_SEG_CACHE_SIZE
#define FADE_SAMPLES        64

/* Segment fade envelope — matches CPU/Metal Hann-window ramp */
__device__ __forceinline__ float fade_envelope(float j, float seg_len) {
    float fade_len = fminf(seg_len * 0.25f, (float)FADE_SAMPLES);
    if (fade_len < 1.0f) fade_len = 1.0f;
    float inv_fade = 1.0f / fade_len;
    if (j < fade_len) {
        return 0.5f * (1.0f - oscillator_fast_sin_cuda((j * inv_fade - 0.5f) * SPECTRAL_PI));
    }
    float from_end = seg_len - 1.0f - j;
    if (from_end < fade_len) {
        return 0.5f * (1.0f - oscillator_fast_sin_cuda((from_end * inv_fade - 0.5f) * SPECTRAL_PI));
    }
    return 1.0f;
}

/* Persistent device buffer cache — avoids per-call cudaMalloc/cudaFree */
static Segment*  d_segments   = NULL;
static uint32_t* d_tile_ids   = NULL;
static TileRange* d_tile_ranges = NULL;
static float*    d_output     = NULL;
static size_t d_seg_cap = 0, d_tile_ids_cap = 0, d_tile_ranges_cap = 0, d_output_cap = 0;

static cudaStream_t g_stream = NULL;
static int g_cuda_available = -1;  /* -1 = not checked, 0 = no, 1 = yes */

/* Grow-only device buffer allocation with 1.5x headroom */
#define CUDA_GROW_BUFFER(ptr, cap, needed, type) do { \
    if ((cap) < (needed)) { \
        if (ptr) cudaFree(ptr); \
        (cap) = (needed) + (needed) / 2; \
        cudaError_t _err = cudaMalloc((void**)&(ptr), (cap)); \
        if (_err != cudaSuccess) { (ptr) = NULL; (cap) = 0; goto cleanup; } \
    } \
} while (0)

/*
 * Tile-parallel synthesis kernel
 *
 * Each threadgroup processes one tile of TILE_SIZE output samples.
 * Segments are loaded cooperatively into shared memory in chunks.
 */
__global__ void synthesize_tile_kernel(
    const Segment* __restrict__ segments,
    const uint32_t* __restrict__ tile_segment_ids,
    const TileRange* __restrict__ tile_ranges,
    float* __restrict__ output,
    uint32_t out_len, float stretch, float inv_stretch,
    float inv_stretch_sq, float pitch_factor, int timbre
) {
    __shared__ Segment seg_cache[SEGMENT_CACHE_SIZE];

    uint32_t tile_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    TileRange range = tile_ranges[tile_idx];
    uint32_t sample_idx = tile_idx * TILE_SIZE + tid;
    float sample_pos = (float)sample_idx;
    float sum = 0.0f;

    for (uint32_t chunk_start = 0; chunk_start < range.count; chunk_start += SEGMENT_CACHE_SIZE) {
        uint32_t chunk_size = min((uint32_t)SEGMENT_CACHE_SIZE, range.count - chunk_start);

        if (tid < chunk_size) {
            uint32_t seg_idx = tile_segment_ids[range.start + chunk_start + tid];
            seg_cache[tid] = segments[seg_idx];
        }
        __syncthreads();

        if (sample_idx < out_len) {
            for (uint32_t i = 0; i < chunk_size; i++) {
                Segment seg = seg_cache[i];
                float seg_start = seg.start * stretch;
                float seg_end = seg_start + seg.length * stretch;
                if (sample_pos < seg_start || sample_pos >= seg_end) continue;

                float j = sample_pos - seg_start;
                float seg_len = seg.length * stretch;
                float alpha = seg.omega * pitch_factor * inv_stretch;
                float beta = seg.df * pitch_factor * inv_stretch_sq;
                float d_a = seg.da * inv_stretch;
                float p = seg.phase + j * (alpha + beta * j);
                sum += (seg.amp + d_a * j) * fade_envelope(j, seg_len) * oscillator_cuda(p, timbre);
            }
        }
        __syncthreads();
    }

    if (sample_idx < out_len) {
        output[sample_idx] = sum;
    }
}

/*
 * Public API
 */

extern "C" void cuda_init(void) {
    if (g_cuda_available >= 0) return;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        g_cuda_available = 0;
        return;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        g_cuda_available = 0;
        return;
    }

    if (prop.major < 3) {
        fprintf(stderr, "CUDA: Device compute capability %d.%d too old (need 3.0+)\n",
                prop.major, prop.minor);
        g_cuda_available = 0;
        return;
    }

    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) {
        g_cuda_available = 0;
        return;
    }

    printf("CUDA: %s (Compute %d.%d, %.1f GB, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.multiProcessorCount);

    g_cuda_available = 1;
}

extern "C" int cuda_available(void) {
    if (g_cuda_available < 0) cuda_init();
    return g_cuda_available;
}

extern "C" SpectralError synth_cuda(
    SegmentArray sa,
    float* out_buffer,
    size_t out_len,
    float stretch,
    float pitch,
    SpectralTimbre timbre,
    double* t_synth
) {
    /* Shared input validation */
    if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;
    }

    /* Check CUDA availability */
    if (!cuda_available()) {
        memset(out_buffer, 0, out_len * sizeof(float));
        *t_synth = 0;
        return SPECTRAL_OK;
    }

    /* Check timbre support - fall back to CPU for unsupported timbres */
    if (!gpu_check_timbre_or_fallback("CUDA", sa, out_buffer, out_len, stretch, pitch, timbre, t_synth)) {
        return SPECTRAL_OK;
    }

    /* Tile preprocessing on CPU */
    TileRange* tile_ranges_host = NULL;
    uint32_t* tile_segment_ids_host = NULL;
    uint32_t num_tiles = 0, total_refs = 0;

    /* Start async segment upload while we preprocess tiles */
    size_t seg_size = sa.count * sizeof(Segment);
    CUDA_GROW_BUFFER(d_segments, d_seg_cap, seg_size, Segment);
    cudaMemcpyAsync(d_segments, sa.segs, seg_size, cudaMemcpyHostToDevice, g_stream);

    SpectralError tile_err = gpu_tile_preprocess(
        sa, stretch, TILE_SIZE, out_len,
        &tile_ranges_host, &tile_segment_ids_host, &num_tiles, &total_refs);
    if (tile_err != SPECTRAL_OK) {
        cudaStreamSynchronize(g_stream);
        memset(out_buffer, 0, out_len * sizeof(float));
        *t_synth = 0;
        return tile_err;
    }

    size_t tile_ids_size = total_refs * sizeof(uint32_t);
    size_t tile_ranges_size = num_tiles * sizeof(TileRange);
    size_t out_size = out_len * sizeof(float);

    /* Grow device buffers as needed */
    CUDA_GROW_BUFFER(d_tile_ids, d_tile_ids_cap, tile_ids_size, uint32_t);
    CUDA_GROW_BUFFER(d_tile_ranges, d_tile_ranges_cap, tile_ranges_size, TileRange);
    CUDA_GROW_BUFFER(d_output, d_output_cap, out_size, float);

    /* Async upload tile data */
    cudaMemcpyAsync(d_tile_ids, tile_segment_ids_host, tile_ids_size, cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(d_tile_ranges, tile_ranges_host, tile_ranges_size, cudaMemcpyHostToDevice, g_stream);

    /* Precompute synthesis parameters */
    float inv_stretch = 1.0f / stretch;
    float inv_stretch_sq = inv_stretch * inv_stretch;
    float pitch_factor = powf(2.0f, pitch / 12.0f);

    /* Launch tile-parallel kernel */
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start, g_stream);
    synthesize_tile_kernel<<<num_tiles, TILE_SIZE, 0, g_stream>>>(
        d_segments, d_tile_ids, d_tile_ranges, d_output,
        (uint32_t)out_len, stretch, inv_stretch, inv_stretch_sq,
        pitch_factor, (int)timbre);
    cudaEventRecord(ev_stop, g_stream);

    /* Async copy result back */
    cudaMemcpyAsync(out_buffer, d_output, out_size, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(kernel_err));
        gpu_tile_preprocess_free(tile_ranges_host, tile_segment_ids_host);
        memset(out_buffer, 0, out_len * sizeof(float));
        *t_synth = 0;
        return SPECTRAL_ERR_GPU_INIT;
    }

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    *t_synth = gpu_ms / 1000.0;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    gpu_tile_preprocess_free(tile_ranges_host, tile_segment_ids_host);
    return SPECTRAL_OK;

cleanup:
    gpu_tile_preprocess_free(tile_ranges_host, tile_segment_ids_host);
    memset(out_buffer, 0, out_len * sizeof(float));
    *t_synth = 0;
    return SPECTRAL_ERR_MEMORY;
}

extern "C" void cuda_cleanup(void) {
    if (d_segments)    { cudaFree(d_segments);    d_segments = NULL; }
    if (d_tile_ids)    { cudaFree(d_tile_ids);    d_tile_ids = NULL; }
    if (d_tile_ranges) { cudaFree(d_tile_ranges); d_tile_ranges = NULL; }
    if (d_output)      { cudaFree(d_output);      d_output = NULL; }
    d_seg_cap = d_tile_ids_cap = d_tile_ranges_cap = d_output_cap = 0;

    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }
}
