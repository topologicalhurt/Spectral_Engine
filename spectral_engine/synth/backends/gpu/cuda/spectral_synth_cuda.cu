/*
 * spectral_synth_cuda.cu - CUDA tile-parallel synthesis backend
 *
 * Uses shared memory segment caching and tile-parallel dispatch,
 * matching the Metal backend architecture. No atomicAdd needed.
 *
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "spectral_common.h"
#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include "oscillator.h"
#include "spectral_osc_formulas.h"
#include "spectral_segment_math.h"
#include "spectral_log.h"
#include "spectral_omp.h"
#include "spectral_utils.h"

enum {
    CUDA_TILE_SIZE = SPECTRAL_GPU_TILE_SIZE,
    CUDA_SEGMENT_CACHE_SIZE = SPECTRAL_METAL_SEG_CACHE_SIZE,
    CUDA_FADE_SAMPLES = SPECTRAL_FADE_SAMPLES_DESKTOP
};

/* Segment fade envelope — delegates to canonical formula in spectral_osc_formulas.h */
__device__ __forceinline__ float fade_envelope(float j, float seg_len) {
    float fade_len = fminf(seg_len * 0.25f, (float)CUDA_FADE_SAMPLES);
    if (fade_len < 1.0f) fade_len = 1.0f;
    float inv_fade = 1.0f / fade_len;
    return spectral_fade_envelope_gpu(j, seg_len, fade_len, inv_fade);
}

/* Persistent device buffer cache — avoids per-call cudaMalloc/cudaFree */
typedef struct {
    Segment*   d_segments;
    uint32_t*  d_tile_ids;
    TileRange* d_tile_ranges;
    float*     d_output;
    size_t     seg_cap, tile_ids_cap, tile_ranges_cap, output_cap;
    cudaStream_t stream;
    int        available;   /* -1=unchecked, 0=no, 1=yes */
} CudaDeviceCache;

static CudaDeviceCache g_cuda = { .available = -1 };

static int cuda_grow_buffer(void** field, size_t* capacity, size_t required) {
    size_t next_capacity = 0;
    cudaError_t err;

    if (!field || !capacity) return 0;
    if (*capacity >= required) return 1;
    if (!spectral_next_capacity_3_over_2(required, &next_capacity)) return 0;

    if (*field) {
        cudaFree(*field);
        *field = NULL;
    }
    *capacity = 0;

    err = cudaMalloc(field, next_capacity);
    if (err != cudaSuccess) return 0;
    *capacity = next_capacity;
    return 1;
}

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
    __shared__ Segment seg_cache[CUDA_SEGMENT_CACHE_SIZE];

    uint32_t tile_idx = blockIdx.x;
    uint32_t tid = threadIdx.x;
    TileRange range = tile_ranges[tile_idx];
    uint32_t sample_idx = tile_idx * CUDA_TILE_SIZE + tid;
    float sample_pos = (float)sample_idx;
    float sum = 0.0f;

    for (uint32_t chunk_start = 0; chunk_start < range.count; chunk_start += CUDA_SEGMENT_CACHE_SIZE) {
        uint32_t chunk_size = min((uint32_t)CUDA_SEGMENT_CACHE_SIZE, range.count - chunk_start);

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
                float alpha = spectral_segment_alpha_f32(seg.omega, pitch_factor, inv_stretch);
                float beta = spectral_segment_beta_f32(seg.df, pitch_factor, inv_stretch_sq);
                float d_amp = spectral_segment_d_amp_f32(seg.da, inv_stretch);
                float p = spectral_segment_phase_at_f32(seg.phase, alpha, beta, j);
                float amp = spectral_segment_amp_at_f32(seg.amp, d_amp, j);
                sum += amp * fade_envelope(j, seg_len) * oscillator_cuda(p, timbre);
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
    if (g_cuda.available >= 0) return;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);

    if (err != cudaSuccess || device_count == 0) {
        g_cuda.available = 0;
        return;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        g_cuda.available = 0;
        return;
    }

    if (prop.major < 3) {
        SPECTRAL_LOG_ERROR_STDERR("CUDA: Device compute capability %d.%d too old (need 3.0+)",
                                  prop.major, prop.minor);
        g_cuda.available = 0;
        return;
    }

    err = cudaStreamCreate(&g_cuda.stream);
    if (err != cudaSuccess) {
        g_cuda.available = 0;
        return;
    }

    SPECTRAL_LOG_INFO("CUDA: %s (Compute %d.%d, %.1f GB, %d SMs)",
                      prop.name, prop.major, prop.minor,
                      prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                      prop.multiProcessorCount);

    g_cuda.available = 1;
}

extern "C" int cuda_available(void) {
    if (g_cuda.available < 0) cuda_init();
    return g_cuda.available;
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
    size_t out_size = 0;

    /* Shared preflight: validate + params + timing */
    SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
    if (!pf.ok) return SPECTRAL_OK;
    if (!spectral_array_bytes(out_len, sizeof(float), &out_size)) {
        *t_synth = 0;
        return SPECTRAL_ERR_OVERFLOW;
    }

    /* Check CUDA availability */
    if (!cuda_available()) {
        memset(out_buffer, 0, out_size);
        *t_synth = 0;
        return SPECTRAL_OK;
    }

    /* Check timbre support - fall back to CPU for unsupported timbres */
    {
        int continue_backend = 1;
        SpectralError gate_err = gpu_check_timbre_or_fallback(
            "CUDA", sa, out_buffer, out_len, stretch, pitch, timbre,
            omp_get_max_threads(), t_synth, &continue_backend);
        if (gate_err != SPECTRAL_OK) {
            return gate_err;
        }
        if (!continue_backend) {
            return SPECTRAL_OK;
        }
    }

    /* Tile preprocessing on CPU */
    GpuTileData td = {0};

    /* Start async segment upload while we preprocess tiles */
    size_t seg_size = 0;
    if (!spectral_array_bytes((size_t)sa.count, sizeof(Segment), &seg_size)) {
        memset(out_buffer, 0, out_size);
        *t_synth = 0;
        return SPECTRAL_ERR_OVERFLOW;
    }
    if (!cuda_grow_buffer((void**)&g_cuda.d_segments, &g_cuda.seg_cap, seg_size)) {
        goto cleanup;
    }
    cudaMemcpyAsync(g_cuda.d_segments, sa.segs, seg_size, cudaMemcpyHostToDevice, g_cuda.stream);

    SpectralError tile_err = gpu_tile_preprocess(sa, stretch, CUDA_TILE_SIZE, out_len, &td);
    if (tile_err != SPECTRAL_OK) {
        cudaStreamSynchronize(g_cuda.stream);
        memset(out_buffer, 0, out_size);
        *t_synth = 0;
        return tile_err;
    }

    size_t tile_ids_size = 0;
    size_t tile_ranges_size = 0;
    if (!spectral_array_bytes((size_t)td.total_refs, sizeof(uint32_t), &tile_ids_size) ||
        !spectral_array_bytes((size_t)td.num_tiles, sizeof(TileRange), &tile_ranges_size)) {
        gpu_tile_data_free(&td);
        memset(out_buffer, 0, out_size);
        *t_synth = 0;
        return SPECTRAL_ERR_OVERFLOW;
    }

    /* Grow device buffers as needed */
    if (!cuda_grow_buffer((void**)&g_cuda.d_tile_ids, &g_cuda.tile_ids_cap, tile_ids_size)) {
        goto cleanup;
    }
    if (!cuda_grow_buffer((void**)&g_cuda.d_tile_ranges, &g_cuda.tile_ranges_cap, tile_ranges_size)) {
        goto cleanup;
    }
    if (!cuda_grow_buffer((void**)&g_cuda.d_output, &g_cuda.output_cap, out_size)) {
        goto cleanup;
    }

    /* Async upload tile data */
    cudaMemcpyAsync(g_cuda.d_tile_ids, td.segment_ids, tile_ids_size, cudaMemcpyHostToDevice, g_cuda.stream);
    cudaMemcpyAsync(g_cuda.d_tile_ranges, td.ranges, tile_ranges_size, cudaMemcpyHostToDevice, g_cuda.stream);

    /* Pack GPU params and launch tile-parallel kernel */
    GpuSynthParams gp = gpu_synth_params_pack(&pf.params, CUDA_TILE_SIZE, timbre);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start, g_cuda.stream);
    synthesize_tile_kernel<<<td.num_tiles, CUDA_TILE_SIZE, 0, g_cuda.stream>>>(
        g_cuda.d_segments, g_cuda.d_tile_ids, g_cuda.d_tile_ranges, g_cuda.d_output,
        gp.out_len, gp.stretch, gp.inv_stretch,
        gp.inv_stretch_sq, gp.pitch_factor, (int)timbre);
    cudaEventRecord(ev_stop, g_cuda.stream);

    /* Async copy result back */
    cudaMemcpyAsync(out_buffer, g_cuda.d_output, out_size, cudaMemcpyDeviceToHost, g_cuda.stream);
    cudaStreamSynchronize(g_cuda.stream);

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        SPECTRAL_LOG_ERROR_STDERR("CUDA kernel error: %s", cudaGetErrorString(kernel_err));
        gpu_tile_data_free(&td);
        memset(out_buffer, 0, out_size);
        *t_synth = 0;
        return SPECTRAL_ERR_GPU_INIT;
    }

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
    *t_synth = gpu_ms / SPECTRAL_MILLIS_PER_SECOND_D;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    gpu_tile_data_free(&td);
    return SPECTRAL_OK;

cleanup:
    gpu_tile_data_free(&td);
    memset(out_buffer, 0, out_size);
    *t_synth = 0;
    return SPECTRAL_ERR_MEMORY;
}

extern "C" void cuda_cleanup(void) {
    if (g_cuda.d_segments)    cudaFree(g_cuda.d_segments);
    if (g_cuda.d_tile_ids)    cudaFree(g_cuda.d_tile_ids);
    if (g_cuda.d_tile_ranges) cudaFree(g_cuda.d_tile_ranges);
    if (g_cuda.d_output)      cudaFree(g_cuda.d_output);
    if (g_cuda.stream)        cudaStreamDestroy(g_cuda.stream);
    g_cuda = (CudaDeviceCache){ .available = -1 };
}
