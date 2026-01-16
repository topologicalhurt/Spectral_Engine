/* spectral_cuda.cu - Standalone CUDA synthesis tool
 * 
 * Uses tile-based parallel synthesis to reduce atomicAdd contention:
 *   1. Pre-compute which segments affect each output tile
 *   2. Each thread block processes one tile with shared memory accumulation
 *   3. Single atomic write per sample (not per segment)
 */

#include <cuda_runtime.h>
#include <sndfile.h>
#include "spectral_common.h"
#include "spectral_segment_parser.h"

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

/* Tile configuration */
#define TILE_SIZE           256
#define THREADS_PER_TILE    256
#define SEGMENT_CACHE_SIZE  128

/* Tile range: start index and count of segments affecting this tile */
typedef struct {
    unsigned int start;
    unsigned int count;
} TileRange;

__device__ __forceinline__ float fast_sin_device(float x) {
    x = x - TWO_PI * floorf(x * INV_TWO_PI + 0.5f);
    float x2 = x * x;
    return x * (PI_SQ - x2) / (PI_SQ + 0.25f * x2);
}

/* Tile-parallel synthesis kernel
 * 
 * Each thread block handles one tile of TILE_SIZE output samples.
 * Uses shared memory segment cache to reduce global memory access.
 * Eliminates per-sample atomicAdd contention by using local accumulation.
 */
__global__ void synthesize_tile_kernel(
    const Segment* __restrict__ segments,
    const unsigned int* __restrict__ tile_segment_ids,
    const TileRange* __restrict__ tile_ranges,
    float* __restrict__ output,
    SynthParams params
) {
    __shared__ Segment seg_cache[SEGMENT_CACHE_SIZE];
    __shared__ float tile_output[TILE_SIZE];
    
    unsigned int tile_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int sample_idx = tile_idx * TILE_SIZE + tid;
    
    /* Initialize shared output to zero */
    if (tid < TILE_SIZE) {
        tile_output[tid] = 0.0f;
    }
    __syncthreads();
    
    TileRange range = tile_ranges[tile_idx];
    float sample_pos = (float)sample_idx;
    float local_sum = 0.0f;
    
    /* Process segments in chunks that fit in shared memory cache */
    for (unsigned int chunk_start = 0; chunk_start < range.count; chunk_start += SEGMENT_CACHE_SIZE) {
        unsigned int chunk_size = min((unsigned int)SEGMENT_CACHE_SIZE, range.count - chunk_start);
        
        /* Cooperatively load segment chunk into shared memory */
        if (tid < chunk_size) {
            unsigned int seg_idx = tile_segment_ids[range.start + chunk_start + tid];
            seg_cache[tid] = segments[seg_idx];
        }
        __syncthreads();
        
        /* Each thread processes its sample against all cached segments */
        if (sample_idx < params.out_len && tid < TILE_SIZE) {
            for (unsigned int i = 0; i < chunk_size; i++) {
                Segment seg = seg_cache[i];
                
                float seg_start = seg.start * params.stretch;
                float seg_end = seg_start + seg.length * params.stretch;
                
                /* Skip if sample outside segment bounds */
                if (sample_pos < seg_start || sample_pos >= seg_end) continue;
                
                float j = sample_pos - seg_start;
                float alpha = seg.freq_hz * params.pitch_factor * params.inv_stretch;
                float beta = seg.df * params.pitch_factor * params.inv_stretch_sq;
                float d_a = seg.da * params.inv_stretch;
                
                float p = seg.phase + j * (alpha + beta * j);
                local_sum += (seg.amp + d_a * j) * fast_sin_device(p);
            }
        }
        __syncthreads();
    }
    
    /* Write to shared memory tile output */
    if (sample_idx < params.out_len && tid < TILE_SIZE) {
        tile_output[tid] = local_sum;
    }
    __syncthreads();
    
    /* Single coalesced write to global memory (no atomics needed for tile output) */
    if (sample_idx < params.out_len && tid < TILE_SIZE) {
        output[sample_idx] = tile_output[tid];
    }
}

/* Legacy kernel */
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
    
    /* Setup parameters */
    SynthParams params = {
        .stretch = stretch,
        .inv_stretch = 1.0f / stretch,
        .inv_stretch_sq = 1.0f / (stretch * stretch),
        .pitch_factor = powf(2.0f, pitch / 12.0f),
        .out_len = (unsigned int)out_len,
        .num_segments = (unsigned int)num_segments
    };
    
    /* Calculate tile count */
    unsigned int num_tiles = (out_len + TILE_SIZE - 1) / TILE_SIZE;
    
    printf("CUDA: Building tile index for %u tiles...\n", num_tiles);
    
    /* Pass 1: Count segments per tile (CPU side) */
    unsigned int* tile_counts = (unsigned int*)calloc(num_tiles, sizeof(unsigned int));
    if (!tile_counts) {
        fprintf(stderr, "CUDA: Failed to allocate tile_counts\n");
        memset(h_output, 0, out_len * sizeof(float));
        *t_synth = 0;
        return;
    }
    
    for (size_t i = 0; i < num_segments; i++) {
        float seg_start = h_segments[i].start * stretch;
        float seg_end = seg_start + h_segments[i].length * stretch;
        
        int start_tile = (int)(seg_start / TILE_SIZE);
        int end_tile = (int)(seg_end / TILE_SIZE);
        
        if (start_tile < 0) start_tile = 0;
        if (start_tile >= (int)num_tiles) continue;
        if (end_tile >= (int)num_tiles) end_tile = num_tiles - 1;
        
        for (int t = start_tile; t <= end_tile; t++) {
            tile_counts[t]++;
        }
    }
    
    /* Build tile ranges and segment ID list */
    TileRange* tile_ranges = (TileRange*)malloc(num_tiles * sizeof(TileRange));
    if (!tile_ranges) {
        free(tile_counts);
        memset(h_output, 0, out_len * sizeof(float));
        *t_synth = 0;
        return;
    }
    
    unsigned int total_refs = 0;
    for (unsigned int t = 0; t < num_tiles; t++) {
        tile_ranges[t].start = total_refs;
        tile_ranges[t].count = tile_counts[t];
        total_refs += tile_counts[t];
    }
    
    unsigned int* tile_segment_ids = (unsigned int*)malloc(total_refs * sizeof(unsigned int));
    unsigned int* tile_cursors = (unsigned int*)calloc(num_tiles, sizeof(unsigned int));
    if (!tile_segment_ids || !tile_cursors) {
        free(tile_counts);
        free(tile_ranges);
        free(tile_segment_ids);
        free(tile_cursors);
        memset(h_output, 0, out_len * sizeof(float));
        *t_synth = 0;
        return;
    }
    
    /* Pass 2: Fill segment IDs per tile */
    for (size_t i = 0; i < num_segments; i++) {
        float seg_start = h_segments[i].start * stretch;
        float seg_end = seg_start + h_segments[i].length * stretch;
        
        int start_tile = (int)(seg_start / TILE_SIZE);
        int end_tile = (int)(seg_end / TILE_SIZE);
        
        if (start_tile < 0) start_tile = 0;
        if (start_tile >= (int)num_tiles) continue;
        if (end_tile >= (int)num_tiles) end_tile = num_tiles - 1;
        
        for (int t = start_tile; t <= end_tile; t++) {
            unsigned int pos = tile_cursors[t]++;
            tile_segment_ids[tile_ranges[t].start + pos] = (unsigned int)i;
        }
    }
    
    free(tile_counts);
    free(tile_cursors);
    
    /* Calculate average segments per tile */
    float avg_segs = (float)total_refs / num_tiles;
    printf("CUDA: %zu segments, %u tiles, avg %.0f segs/tile\n", 
           num_segments, num_tiles, avg_segs);
    
    /* Allocate device memory */
    Segment* d_segments;
    unsigned int* d_tile_segment_ids;
    TileRange* d_tile_ranges;
    float* d_output;
    
    size_t seg_size = num_segments * sizeof(Segment);
    size_t ids_size = total_refs * sizeof(unsigned int);
    size_t ranges_size = num_tiles * sizeof(TileRange);
    size_t out_size = out_len * sizeof(float);
    
    printf("CUDA: Allocating %.1f MB for segments, %.1f MB for output\n",
           seg_size / (1024.0 * 1024.0), out_size / (1024.0 * 1024.0));
    
    CUDA_CHECK(cudaMalloc(&d_segments, seg_size));
    CUDA_CHECK(cudaMalloc(&d_tile_segment_ids, ids_size));
    CUDA_CHECK(cudaMalloc(&d_tile_ranges, ranges_size));
    CUDA_CHECK(cudaMalloc(&d_output, out_size));
    
    /* Copy data to device */
    CUDA_CHECK(cudaMemcpy(d_segments, h_segments, seg_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_segment_ids, tile_segment_ids, ids_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tile_ranges, tile_ranges, ranges_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, out_size));
    
    /* Launch tile-parallel kernel */
    printf("CUDA: Launching %u blocks x %d threads (tile-parallel)\n", num_tiles, THREADS_PER_TILE);
    
    cudaEventRecord(start);
    synthesize_tile_kernel<<<num_tiles, THREADS_PER_TILE>>>(
        d_segments, d_tile_segment_ids, d_tile_ranges, d_output, params);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    
    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));
    
    printf("CUDA: GPU kernel time = %.1f ms\n", gpu_ms);
    *t_synth = gpu_ms / 1000.0;
    
    /* Cleanup */
    cudaFree(d_segments);
    cudaFree(d_tile_segment_ids);
    cudaFree(d_tile_ranges);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(tile_ranges);
    free(tile_segment_ids);
}

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
    if (!segments || !output) {
        fprintf(stderr, "Error: Failed to allocate memory for demo\n");
        free(segments);
        free(output);
        return;
    }
    
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
    if (!output) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        free(sa.segs);
        return 1;
    }
    
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
    if (!outfile) {
        fprintf(stderr, "Error: Cannot create output file %s\n", out_path);
        free(sa.segs);
        free(output);
        return 1;
    }
    sf_writef_float(outfile, output, out_len);
    sf_close(outfile);
    
    printf("Wrote %s (%.1f sec)\n", out_path, (float)out_len / sr);
    
    free(sa.segs);
    free(output);
    return 0;
}
