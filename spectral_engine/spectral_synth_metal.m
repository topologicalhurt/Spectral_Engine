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
#include "spectral_omp.h"

/* Metal kernel source - struct definitions and synthesis kernel.
 * Oscillator functions come from oscillator_metal_source (defined in oscillator.c) */
static const char* metalKernelStructs = 
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
/* NOTE: Must match Segment layout in spectral_common.h (no compile-time check possible) */
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
"#define THREADS_PER_TILE " SPECTRAL_STR(SPECTRAL_GPU_TILE_SIZE) "\n"
"#define SEGMENT_CACHE_SIZE " SPECTRAL_STR(SPECTRAL_METAL_SEG_CACHE_SIZE) "\n"
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
"                float seg_len = seg.length * params.stretch;\n"
"                float p = seg.phase + j * (alpha + beta * j);\n"
"                sum += (seg.amp + d_a * j) * fade_envelope(j, seg_len) * oscillator(p, timbre);\n"
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

/* Persistent buffer cache — avoids per-call VM page allocation */
typedef struct {
    id<MTLBuffer> segBuf, tileIdsBuf, tileRangesBuf, outputBuf;
    size_t segCap, tileIdsCap, tileRangesCap, outputCap;
} MetalBufferCache;

static MetalBufferCache g_mtl;

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
        options.languageVersion = MTLLanguageVersion2_4;
        
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

SpectralError synth_metal(SegmentArray sa, float* out_buffer, size_t out_len,
                          float stretch, float pitch, SpectralTimbre timbre, double* t_synth) {
    @autoreleasepool {
        /* Shared preflight: validate + params + timing */
        SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
        if (!pf.ok) return SPECTRAL_OK;

        /* Check timbre support - fall back to CPU for unsupported timbres */
        if (!gpu_check_timbre_or_fallback("Metal", sa, out_buffer, out_len, stretch, pitch, timbre, t_synth)) {
            return SPECTRAL_OK;
        }

        GpuTileData td = {0};
        SpectralError tile_err = gpu_tile_preprocess(
            sa, stretch, SPECTRAL_GPU_TILE_SIZE, out_len, &td);
        if (tile_err != SPECTRAL_OK) {
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return tile_err;
        }

#if SPECTRAL_DEBUG && !defined(NDEBUG)
        float avg_segs = (float)td.total_refs / td.num_tiles;
#endif
        
        GpuSynthParams params = gpu_synth_params_pack(&pf.params, SPECTRAL_GPU_TILE_SIZE, timbre);
        
        size_t segment_buf_size = sa.count * sizeof(Segment);
        size_t tile_ids_size = td.total_refs * sizeof(uint32_t);
        size_t tile_ranges_size = td.num_tiles * sizeof(TileRange);
        size_t output_size = out_len * sizeof(float);

        SPECTRAL_DBG("Metal: %u segs, %u tiles, avg %.0f segs/tile",
                sa.count, td.num_tiles, avg_segs);

        /* Reuse cached buffers when capacity suffices; grow with 1.5x headroom */
        if (g_mtl.segCap < segment_buf_size) {
            g_mtl.segBuf = nil;
            g_mtl.segCap = segment_buf_size + segment_buf_size / 2;
            g_mtl.segBuf = [metalDevice newBufferWithLength:g_mtl.segCap
                                                   options:MTLResourceStorageModeShared];
        }
        if (g_mtl.tileIdsCap < tile_ids_size) {
            g_mtl.tileIdsBuf = nil;
            g_mtl.tileIdsCap = tile_ids_size + tile_ids_size / 2;
            g_mtl.tileIdsBuf = [metalDevice newBufferWithLength:g_mtl.tileIdsCap
                                                       options:MTLResourceStorageModeShared];
        }
        if (g_mtl.tileRangesCap < tile_ranges_size) {
            g_mtl.tileRangesBuf = nil;
            g_mtl.tileRangesCap = tile_ranges_size + tile_ranges_size / 2;
            g_mtl.tileRangesBuf = [metalDevice newBufferWithLength:g_mtl.tileRangesCap
                                                           options:MTLResourceStorageModeShared];
        }
        if (g_mtl.outputCap < output_size) {
            g_mtl.outputBuf = nil;
            g_mtl.outputCap = output_size + output_size / 2;
            g_mtl.outputBuf = [metalDevice newBufferWithLength:g_mtl.outputCap
                                                      options:MTLResourceStorageModeShared];
        }

        if (!g_mtl.segBuf || !g_mtl.tileIdsBuf || !g_mtl.tileRangesBuf || !g_mtl.outputBuf) {
            gpu_tile_data_free(&td);
            memset(out_buffer, 0, out_len * sizeof(float));
            *t_synth = 0;
            return SPECTRAL_ERR_MEMORY;
        }

        /* Copy data into cached buffers */
        memcpy([g_mtl.segBuf contents], sa.segs, segment_buf_size);
        memcpy([g_mtl.tileIdsBuf contents], td.segment_ids, tile_ids_size);
        memcpy([g_mtl.tileRangesBuf contents], td.ranges, tile_ranges_size);

        /* Params buffer: small (36 bytes), allocated per-call */
        id<MTLBuffer> paramsBuffer = [metalDevice newBufferWithBytes:&params
                                                              length:sizeof(params)
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuffer = [metalQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:metalSynthPipeline];
        [encoder setBuffer:g_mtl.segBuf offset:0 atIndex:0];
        [encoder setBuffer:g_mtl.tileIdsBuf offset:0 atIndex:1];
        [encoder setBuffer:g_mtl.tileRangesBuf offset:0 atIndex:2];
        [encoder setBuffer:g_mtl.outputBuf offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        [encoder dispatchThreadgroups:MTLSizeMake(td.num_tiles, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SPECTRAL_GPU_TILE_SIZE, 1, 1)];
        [encoder endEncoding];
        
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        
        memcpy(out_buffer, [g_mtl.outputBuf contents], output_size);
        *t_synth = omp_get_wtime() - pf.start_time;

        gpu_tile_data_free(&td);
    }
    return SPECTRAL_OK;
}

void metal_cleanup(void) {
    g_mtl = (MetalBufferCache){0};
}
