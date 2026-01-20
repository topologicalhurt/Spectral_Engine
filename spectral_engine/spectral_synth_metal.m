/* spectral_synth_metal.m - Metal GPU synthesis (macOS)
 * 
 * Uses oscillator functions from oscillator.h for GPU waveform generation.
 * The Metal shader source (oscillator_metal_source) is defined in oscillator.c.
 * 
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include "spectral_utils.h"
#include "oscillator.h"
#include <omp.h>

/* Metal kernel source - struct definitions and synthesis kernel.
 * Oscillator functions come from oscillator_metal_source (defined in oscillator.c) */
static const char* metalKernelStructs = 
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct Segment {\n"
"    float start;\n"
"    float length;\n"
"    float phase;\n"
"    float omega;\n"
"    float df;\n"
"    float amp;\n"
"    float da;\n"
"    float width;\n"
"    float _pad[8];\n"
"};\n"
"\n"
"struct SynthParams {\n"
"    float stretch;\n"
"    float inv_stretch;\n"
"    float inv_stretch_sq;\n"
"    float pitch_factor;\n"
"    uint out_len;\n"
"    uint num_segments;\n"
"    uint tile_size;\n"
"    uint timbre;\n"
"};\n"
"\n"
"struct TileRange {\n"
"    uint start;\n"
"    uint count;\n"
"};\n"
"\n";

static const char* metalKernelCode = 
"#define THREADS_PER_TILE 512\n"
"#define SEGMENT_CACHE_SIZE 128\n"
"\n"
"kernel void synthesize_tile_parallel(\n"
"    device const Segment* segments [[buffer(0)]],\n"
"    device const uint* tile_segment_ids [[buffer(1)]],\n"
"    device const TileRange* tile_ranges [[buffer(2)]],\n"
"    device float* output [[buffer(3)]],\n"
"    constant SynthParams& params [[buffer(4)]],\n"
"    uint tile_idx [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]]\n"
") {\n"
"    threadgroup Segment seg_cache[SEGMENT_CACHE_SIZE];\n"
"    threadgroup float tile_output[THREADS_PER_TILE];\n"
"    \n"
"    TileRange range = tile_ranges[tile_idx];\n"
"    uint sample_idx = tile_idx * params.tile_size + tid;\n"
"    float sample_pos = (float)sample_idx;\n"
"    uint timbre = params.timbre;\n"
"    \n"
"    float sum = 0.0f;\n"
"    \n"
"    for (uint chunk_start = 0; chunk_start < range.count; chunk_start += SEGMENT_CACHE_SIZE) {\n"
"        uint chunk_size = min((uint)SEGMENT_CACHE_SIZE, range.count - chunk_start);\n"
"        \n"
"        if (tid < chunk_size) {\n"
"            uint seg_idx = tile_segment_ids[range.start + chunk_start + tid];\n"
"            seg_cache[tid] = segments[seg_idx];\n"
"        }\n"
"        \n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        \n"
"        if (sample_idx < params.out_len) {\n"
"            for (uint i = 0; i < chunk_size; i++) {\n"
"                Segment seg = seg_cache[i];\n"
"                \n"
"                float seg_start = seg.start * params.stretch;\n"
"                float seg_end = seg_start + seg.length * params.stretch;\n"
"                \n"
"                if (sample_pos < seg_start || sample_pos >= seg_end) continue;\n"
"                \n"
"                float j = sample_pos - seg_start;\n"
"                float alpha = seg.omega * params.pitch_factor * params.inv_stretch;\n"
"                float beta = seg.df * params.pitch_factor * params.inv_stretch_sq;\n"
"                float d_a = seg.da * params.inv_stretch;\n"
"                \n"
"                float p = seg.phase + j * (alpha + beta * j);\n"
"                sum += (seg.amp + d_a * j) * oscillator(p, timbre);\n"
"            }\n"
"        }\n"
"        \n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    }\n"
"    \n"
"    if (sample_idx < params.out_len) {\n"
"        output[sample_idx] = sum;\n"
"    }\n"
"}\n";

static id<MTLDevice> metalDevice = nil;
static id<MTLCommandQueue> metalQueue = nil;
static id<MTLComputePipelineState> metalSynthPipeline = nil;
static bool metalInitialized = false;
static bool metalIsAvailable = false;

typedef struct {
    uint32_t start;
    uint32_t count;
} TileRange;

void metal_init(void) {
    if (metalInitialized) return;
    metalInitialized = true;
    
    @autoreleasepool {
        NSArray<id<MTLDevice>>* allDevices = MTLCopyAllDevices();
        if (allDevices && [allDevices count] > 0) {
            metalDevice = allDevices[0];
        } else {
            metalDevice = MTLCreateSystemDefaultDevice();
        }
        
        if (!metalDevice) {
            SPECTRAL_WARN("Metal: No GPU found");
            return;
        }
        
        SPECTRAL_DBG("Metal: %s", [[metalDevice name] UTF8String]);
        
        metalQueue = [metalDevice newCommandQueue];
        
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        
        /* Combine shader sources: structs + oscillator (from oscillator.c) + kernel */
        NSString* source = [NSString stringWithFormat:@"%s%s%s", 
                           metalKernelStructs, 
                           oscillator_metal_source,  /* From oscillator.c */
                           metalKernelCode];
        id<MTLLibrary> library = [metalDevice newLibraryWithSource:source options:options error:&error];
        if (error) {
            SPECTRAL_WARN("Metal: Shader error: %s", [[error localizedDescription] UTF8String]);
            return;
        }
        
        id<MTLFunction> synthKernel = [library newFunctionWithName:@"synthesize_tile_parallel"];
        metalSynthPipeline = [metalDevice newComputePipelineStateWithFunction:synthKernel error:&error];
        if (error) {
            SPECTRAL_WARN("Metal: Pipeline error: %s", [[error localizedDescription] UTF8String]);
            return;
        }
        
        metalIsAvailable = true;
    }
}

int metal_available(void) {
    return metalIsAvailable ? 1 : 0;
}

void synth_metal(SegmentArray sa, float* out_buffer, size_t out_len,
                 float stretch, float pitch, SpectralTimbre timbre, double* t_synth) {
    @autoreleasepool {
        /* Shared input validation */
        if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, &t_synth)) {
            return;
        }
        
        /* Check timbre support - fall back to CPU for unsupported timbres */
        if (!gpu_check_timbre_or_fallback("Metal", sa, out_buffer, out_len, stretch, pitch, timbre, t_synth)) {
            return;
        }
        
        double synth_start = omp_get_wtime();
        uint32_t num_tiles = ((uint32_t)out_len + SPECTRAL_GPU_TILE_SIZE - 1) / SPECTRAL_GPU_TILE_SIZE;
        int n_threads = omp_get_max_threads();
        
        uint32_t** thread_counts = malloc(n_threads * sizeof(uint32_t*));
        if (!thread_counts) {
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return;
        }
        for (int t = 0; t < n_threads; t++) {
            thread_counts[t] = calloc(num_tiles, sizeof(uint32_t));
            if (!thread_counts[t]) {
                for (int i = 0; i < t; i++) free(thread_counts[i]);
                free(thread_counts);
                memset(out_buffer, 0, out_len * sizeof(float));
                *t_synth = 0;
                return;
            }
        }
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint32_t* my_counts = thread_counts[tid];
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < sa.count; i++) {
                float start = sa.segs[i].start * stretch;
                float end = start + sa.segs[i].length * stretch;
                
                int start_tile = (int)(start / SPECTRAL_GPU_TILE_SIZE);
                int end_tile = (int)(end / SPECTRAL_GPU_TILE_SIZE);
                if (start_tile < 0) start_tile = 0;
                if (start_tile >= (int)num_tiles) continue;
                if (end_tile >= (int)num_tiles) end_tile = num_tiles - 1;
                
                for (int t = start_tile; t <= end_tile; t++) {
                    my_counts[t]++;
                }
            }
        }
        
        uint32_t* tile_counts = calloc(num_tiles, sizeof(uint32_t));
        if (!tile_counts) {
            for (int t = 0; t < n_threads; t++) free(thread_counts[t]);
            free(thread_counts);
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return;
        }
        for (int t = 0; t < n_threads; t++) {
            for (uint32_t i = 0; i < num_tiles; i++) {
                tile_counts[i] += thread_counts[t][i];
            }
            free(thread_counts[t]);
        }
        free(thread_counts);
        
        TileRange* tile_ranges = malloc(num_tiles * sizeof(TileRange));
        if (!tile_ranges) {
            free(tile_counts);
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return;
        }
        uint32_t total_refs = 0;
        for (uint32_t t = 0; t < num_tiles; t++) {
            tile_ranges[t].start = total_refs;
            tile_ranges[t].count = tile_counts[t];
            total_refs += tile_counts[t];
        }
        
        uint32_t* tile_segment_ids = malloc(total_refs * sizeof(uint32_t));
        uint32_t* tile_cursors = calloc(num_tiles, sizeof(uint32_t));
        if (!tile_segment_ids || !tile_cursors) {
            free(tile_counts);
            free(tile_ranges);
            free(tile_segment_ids);
            free(tile_cursors);
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return;
        }
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            float start = sa.segs[i].start * stretch;
            float end = start + sa.segs[i].length * stretch;
            
            int start_tile = (int)(start / SPECTRAL_GPU_TILE_SIZE);
            int end_tile = (int)(end / SPECTRAL_GPU_TILE_SIZE);
            if (start_tile < 0) start_tile = 0;
            if (start_tile >= (int)num_tiles) continue;
            if (end_tile >= (int)num_tiles) end_tile = num_tiles - 1;
            
            for (int t = start_tile; t <= end_tile; t++) {
                uint32_t pos;
                #pragma omp atomic capture
                pos = tile_cursors[t]++;
                tile_segment_ids[tile_ranges[t].start + pos] = (uint32_t)i;
            }
        }
        
        free(tile_counts);
        free(tile_cursors);
        
#if SPECTRAL_DEBUG && !defined(NDEBUG)
        float avg_segs = (float)total_refs / num_tiles;
#endif
        
        struct {
            float stretch;
            float inv_stretch;
            float inv_stretch_sq;
            float pitch_factor;
            uint32_t out_len;
            uint32_t num_segments;
            uint32_t tile_size;
            uint32_t timbre;
        } params = {
            .stretch = stretch,
            .inv_stretch = 1.0f / stretch,
            .inv_stretch_sq = 1.0f / (stretch * stretch),
            .pitch_factor = SPECTRAL_PITCH_FACTOR(pitch),
            .out_len = (uint32_t)out_len,
            .num_segments = sa.count,
            .tile_size = SPECTRAL_GPU_TILE_SIZE,
            .timbre = (uint32_t)timbre
        };
        
        size_t segment_buf_size = sa.count * sizeof(Segment);
        size_t tile_ids_size = total_refs * sizeof(uint32_t);
        size_t tile_ranges_size = num_tiles * sizeof(TileRange);
        size_t output_size = out_len * sizeof(float);
        
        SPECTRAL_DBG("Metal: %u segs, %u tiles, avg %.0f segs/tile", 
                sa.count, num_tiles, avg_segs);
        
        id<MTLBuffer> segmentBuffer = [metalDevice newBufferWithBytes:sa.segs
                                                               length:segment_buf_size
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> tileIdsBuffer = [metalDevice newBufferWithBytes:tile_segment_ids
                                                               length:tile_ids_size
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> tileRangesBuffer = [metalDevice newBufferWithBytes:tile_ranges
                                                                  length:tile_ranges_size
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [metalDevice newBufferWithLength:output_size
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> paramsBuffer = [metalDevice newBufferWithBytes:&params
                                                              length:sizeof(params)
                                                             options:MTLResourceStorageModeShared];

        if (!segmentBuffer || !tileIdsBuffer || !tileRangesBuffer || !outputBuffer || !paramsBuffer) {
            free(tile_ranges);
            free(tile_segment_ids);
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return;
        }
        
        id<MTLCommandBuffer> cmdBuffer = [metalQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:metalSynthPipeline];
        [encoder setBuffer:segmentBuffer offset:0 atIndex:0];
        [encoder setBuffer:tileIdsBuffer offset:0 atIndex:1];
        [encoder setBuffer:tileRangesBuffer offset:0 atIndex:2];
        [encoder setBuffer:outputBuffer offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];
        
        [encoder dispatchThreadgroups:MTLSizeMake(num_tiles, 1, 1) 
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        memcpy(out_buffer, [outputBuffer contents], output_size);
        *t_synth = omp_get_wtime() - synth_start;
        
        free(tile_ranges);
        free(tile_segment_ids);
    }
}
