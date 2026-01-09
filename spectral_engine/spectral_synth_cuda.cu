/*
 * spectral_synth_cuda.cu - CUDA synthesis backend for integration with main binary
 */

#include <cuda_runtime.h>
#include "spectral_common.h"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
}

static int g_cuda_available = -1;  // -1 = not checked, 0 = no, 1 = yes

// ============================================================================
// CUDA Kernel: Segment-parallel synthesis with native atomic float adds
// ============================================================================

__global__ void synthesize_kernel(
    const Segment* __restrict__ segments,
    float* __restrict__ output,
    SynthParams params
) {
    unsigned int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_idx >= params.num_segments) return;
    
    Segment seg = segments[seg_idx];
    
    // Calculate output range for this segment
    unsigned int start_idx = (unsigned int)(seg.start * params.stretch);
    unsigned int seg_len = (unsigned int)(seg.length * params.stretch);
    
    if (start_idx >= params.out_len) return;
    if (start_idx + seg_len > params.out_len) {
        seg_len = params.out_len - start_idx;
    }
    
    // Precompute synthesis parameters
    float alpha = seg.freq_hz * params.pitch_factor * params.inv_stretch;
    float beta = seg.df * params.pitch_factor * params.inv_stretch_sq;
    float d_a = seg.da * params.inv_stretch;
    float phase = seg.phase;
    float amp = seg.amp;
    
    // Synthesize each sample in this segment
    for (unsigned int j = 0; j < seg_len; j++) {
        float p = phase + (float)j * (alpha + beta * (float)j);
        float val = (amp + d_a * (float)j) * fast_sin_device(p);
        
        // Native atomic float add - hardware accelerated on modern NVIDIA GPUs
        atomicAdd(&output[start_idx + j], val);
    }
}

// ============================================================================
// Public API for integration with main binary
// ============================================================================

extern "C" void cuda_init(void) {
    if (g_cuda_available >= 0) return;  // Already checked
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        g_cuda_available = 0;
        return;
    }
    
    // Try to get device properties to verify it's usable
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        g_cuda_available = 0;
        return;
    }
    
    // Require compute capability 3.0+ for atomicAdd on float
    if (prop.major < 3) {
        fprintf(stderr, "CUDA: Device compute capability %d.%d too old (need 3.0+)\n", 
                prop.major, prop.minor);
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

extern "C" void synth_cuda(
    SegmentArray sa,
    float* out_buffer,
    size_t out_len,
    float stretch,
    float pitch,
    double* t_synth
) {
    if (!cuda_available()) {
        fprintf(stderr, "CUDA: Not available\n");
        *t_synth = 0;
        return;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    Segment* d_segments;
    float* d_output;
    
    size_t seg_size = sa.count * sizeof(Segment);
    size_t out_size = out_len * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_segments, seg_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    
    // Copy segments to device
    CUDA_CHECK(cudaMemcpy(d_segments, sa.segs, seg_size, cudaMemcpyHostToDevice));
    
    // Zero output buffer on device
    CUDA_CHECK(cudaMemset(d_output, 0, out_size));
    
    // Setup parameters
    SynthParams params = {
        .stretch = stretch,
        .inv_stretch = 1.0f / stretch,
        .inv_stretch_sq = 1.0f / (stretch * stretch),
        .pitch_factor = powf(2.0f, pitch / 12.0f),
        .out_len = (unsigned int)out_len,
        .num_segments = (unsigned int)sa.count
    };
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (sa.count + threads_per_block - 1) / threads_per_block;
    
    cudaEventRecord(start);
    synthesize_kernel<<<num_blocks, threads_per_block>>>(d_segments, d_output, params);
    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(kernel_err));
        cudaFree(d_segments);
        cudaFree(d_output);
        *t_synth = 0;
        return;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(out_buffer, d_output, out_size, cudaMemcpyDeviceToHost));
    
    *t_synth = gpu_ms / 1000.0;
    
    // Cleanup
    cudaFree(d_segments);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
