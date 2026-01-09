/*
 * CUDA GPU Synthesis for Spectral Processing
 * 
 * COMPILE: nvcc -O3 -arch=sm_70 partial_cuda.cu -o spectral_cuda -lsndfile
 * USAGE:   ./spectral_cuda segments.bin output.wav
 *          ./spectral_cuda --demo  (run benchmark)
 * 
 * Uses segment-parallel approach with native atomicAdd for float accumulation.
 * Each CUDA thread processes one segment, atomically adding to output buffer.
 */

#include <cuda_runtime.h>
#include <sndfile.h>
#include "spectral_common.h"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

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
// Host API
// ============================================================================

extern "C" void render_cuda(
    Segment* h_segments,
    size_t num_segments,
    float* h_output,
    size_t out_len,
    float stretch,
    float pitch,
    double* t_synth
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate device memory
    Segment* d_segments;
    float* d_output;
    
    size_t seg_size = num_segments * sizeof(Segment);
    size_t out_size = out_len * sizeof(float);
    
    printf("CUDA: Allocating %.1f MB for segments, %.1f MB for output\n",
           seg_size / (1024.0 * 1024.0), out_size / (1024.0 * 1024.0));
    
    CUDA_CHECK(cudaMalloc(&d_segments, seg_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    
    // Copy segments to device
    CUDA_CHECK(cudaMemcpy(d_segments, h_segments, seg_size, cudaMemcpyHostToDevice));
    
    // Zero output buffer
    CUDA_CHECK(cudaMemset(d_output, 0, out_size));
    
    // Setup parameters
    SynthParams params = {
        .stretch = stretch,
        .inv_stretch = 1.0f / stretch,
        .inv_stretch_sq = 1.0f / (stretch * stretch),
        .pitch_factor = powf(2.0f, pitch / 12.0f),
        .out_len = (unsigned int)out_len,
        .num_segments = (unsigned int)num_segments
    };
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (num_segments + threads_per_block - 1) / threads_per_block;
    
    printf("CUDA: Launching %d blocks x %d threads\n", num_blocks, threads_per_block);
    
    cudaEventRecord(start);
    synthesize_kernel<<<num_blocks, threads_per_block>>>(d_segments, d_output, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));
    
    printf("CUDA: GPU kernel time = %.1f ms\n", gpu_ms);
    *t_synth = gpu_ms / 1000.0;
    
    // Cleanup
    cudaFree(d_segments);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main - standalone synthesis from segment file
// ============================================================================

void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("CUDA Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("  Memory: %.1f GB, SMs: %d\n", 
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), prop.multiProcessorCount);
}

void run_demo() {
    printf("\n--- CUDA Synthesis Demo ---\n");
    
    size_t num_segments = 10000000;  // 10M segments
    size_t out_len = 44100 * 10;     // 10 seconds @ 44.1kHz
    
    printf("Generating %zu random segments...\n", num_segments);
    
    Segment* segments = (Segment*)malloc(num_segments * sizeof(Segment));
    float* output = (float*)calloc(out_len, sizeof(float));
    
    srand(42);
    for (size_t i = 0; i < num_segments; i++) {
        segments[i].start = (float)(rand() % (int)(out_len * 0.9f));
        segments[i].length = 32.0f + (float)(rand() % 128);
        segments[i].phase = (float)(rand() % 1000) / 1000.0f * TWO_PI;
        segments[i].freq_hz = 100.0f + (float)(rand() % 2000);
        segments[i].df = 0.0f;
        segments[i].amp = 0.001f + (float)(rand() % 100) / 10000.0f;
        segments[i].da = 0.0f;
        segments[i].width = 1.0f;
    }
    
    printf("Test: %zu segments, %zu output samples (density=%.1f)\n", 
           num_segments, out_len, (float)num_segments / out_len);
    
    double t_synth;
    render_cuda(segments, num_segments, output, out_len, 1.0f, 0.0f, &t_synth);
    
    printf("\nResults:\n");
    printf("  Synthesis time: %.1f ms\n", t_synth * 1000.0);
    printf("  Throughput: %.1f M segments/sec\n", num_segments / t_synth / 1e6);
    printf("  Effective rate: %.1f GB/s\n", 
           (num_segments * sizeof(Segment) + out_len * sizeof(float)) / t_synth / 1e9);
    
    free(segments);
    free(output);
}

int main(int argc, char** argv) {
    print_device_info();
    
    if (argc < 2 || strcmp(argv[1], "--demo") == 0) {
        run_demo();
        return 0;
    }
    
    // Load segments from file
    const char* seg_path = argv[1];
    const char* out_path = (argc > 2) ? argv[2] : "out_cuda.wav";
    
    SegmentArray sa;
    int sr;
    float stretch, pitch;
    
    printf("Loading segments from %s...\n", seg_path);
    int err = segments_load(seg_path, &sa, &sr, &stretch, &pitch);
    if (err) {
        fprintf(stderr, "Error loading segments: %d\n", err);
        return 1;
    }
    
    printf("Loaded %zu segments (sr=%d, stretch=%.2f, pitch=%.1f)\n", 
           sa.count, sr, stretch, pitch);
    
    size_t out_len = (size_t)(sa.count > 0 ? 
                              (sa.segs[sa.count-1].start + sa.segs[sa.count-1].length) * stretch * 1.1 
                              : sr * 10);
    float* output = (float*)calloc(out_len, sizeof(float));
    
    double t_synth;
    render_cuda(sa.segs, sa.count, output, out_len, stretch, pitch, &t_synth);
    
    // Normalize
    float max_amp = 0.0f;
    for (size_t i = 0; i < out_len; i++) {
        float a = fabsf(output[i]);
        if (a > max_amp) max_amp = a;
    }
    if (max_amp > 0) {
        float scale = 0.95f / max_amp;
        for (size_t i = 0; i < out_len; i++) output[i] *= scale;
    }
    
    // Write output
    SF_INFO info = {0};
    info.samplerate = sr;
    info.channels = 1;
    info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE* outfile = sf_open(out_path, SFM_WRITE, &info);
    sf_writef_float(outfile, output, out_len);
    sf_close(outfile);
    
    printf("Wrote %s (%.1f sec)\n", out_path, (float)out_len / sr);
    
    free(sa.segs);
    free(output);
    return 0;
}
