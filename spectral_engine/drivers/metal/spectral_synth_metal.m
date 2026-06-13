/* spectral_synth_metal.m - Metal GPU synthesis (macOS)
 * 
 * Uses oscillator functions from oscillator.h for GPU waveform generation.
 * The Metal shader source (oscillator_metal_source) is defined in the payload TU (spectral_osc_metal_payload.c).
 * 
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "spectral_metal.h"   /* the driver surface this TU implements */
#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include "spectral_utils.h"
#include "oscillator.h"
#include "spectral_omp.h"

/* The oscillator and segment-math MSL functions are CODEGEN'd from the canonical
 * C contract (spectral_osc_formulas.h / spectral_segment_math.h) into
 * drivers/metal/spectral_osc_metal_generated.h by tools/.../generators/metal_osc.py and
 * exposed as oscillator_metal_source + spectral_segment_math_metal_source.  The
 * CMake verify_metal_osc target enforces they match the C formulas every build,
 * which replaces the old _Static_assert(VERSION==N) drift reminders. */

/* The struct definitions (SegmentGpu/SynthParams/TileRange) come from the
 * generated gpu_structs_metal_source — codegen from the C layouts, so the
 * old "must match, no compile-time check possible" hand mirror is gone. */

static const char* metalKernelCode = 
"#define THREADS_PER_TILE " SPECTRAL_STR(SPECTRAL_GPU_TILE_SIZE) "\n"
"#define SEGMENT_CACHE_SIZE " SPECTRAL_STR(SPECTRAL_GPU_SEG_CACHE_SIZE) "\n"
"/* 32-byte SegmentGpu = 2x entries in same threadgroup budget */\n"
"\n"
"/* segment_alpha/beta/d_amp/phase_at/amp_at are provided by the generated\n"
" * spectral_segment_math_metal_source string (prepended at compile time). */\n"
"kernel void synthesize_tile_parallel(\n"
"    device const SegmentGpu* segments [[buffer(0)]],\n"
"    device const uint* tile_segment_ids [[buffer(1)]],\n"
"    device const TileRange* tile_ranges [[buffer(2)]],\n"
"    device float* output [[buffer(3)]],\n"
"    constant SynthParams& params [[buffer(4)]],\n"
"    uint tile_idx [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]]\n"
") {\n"
"    threadgroup SegmentGpu seg_cache[SEGMENT_CACHE_SIZE];\n"
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
"                SegmentGpu seg = seg_cache[i];\n"
"                \n"
"                float seg_start = seg.start * params.stretch;\n"
"                float seg_end = seg_start + seg.length * params.stretch;\n"
"                \n"
"                if (sample_pos < seg_start || sample_pos >= seg_end) continue;\n"
"                \n"
"                float j = sample_pos - seg_start;\n"
"                float alpha = segment_alpha(seg.omega, params.pitch_factor, params.inv_stretch);\n"
"                float beta = segment_beta(seg.df, params.pitch_factor, params.inv_stretch_sq);\n"
"                float d_a = segment_d_amp(seg.da, params.inv_stretch);\n"
"                \n"
"                float seg_len = seg.length * params.stretch;\n"
"                float p = segment_phase_at(seg.phase, alpha, beta, j);\n"
"                sum += segment_amp_at(seg.amp, d_a, j) * fade_envelope(j, seg_len) * oscillator(p, timbre);\n"
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
    id<MTLBuffer> segBufNoCopy;       /* zero-copy wrapper for mmap'd SegmentGpu */
    size_t segCap, tileIdsCap, tileRangesCap, outputCap;
} MetalBufferCache;

static MetalBufferCache g_mtl;

static int metal_grow_buffer(id<MTLDevice> device,
                             __strong id<MTLBuffer>* buffer,
                             size_t* capacity,
                             size_t required_size) {
    size_t next_capacity = 0;

    if (!device || !buffer || !capacity) return 0;
    if (*capacity >= required_size) return 1;
    if (!spectral_next_capacity_3_over_2(required_size, &next_capacity)) return 0;

    *buffer = nil;
    *capacity = 0;
    *buffer = [device newBufferWithLength:next_capacity
                                  options:MTLResourceStorageModeShared];
    if (!*buffer) return 0;
    *capacity = next_capacity;
    return 1;
}

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
#if defined(SPECTRAL_METAL_FAST_MATH) && SPECTRAL_METAL_FAST_MATH
        options.fastMathEnabled = YES;
#else
        options.fastMathEnabled = NO;
#endif
        options.languageVersion = MTLLanguageVersion2_4;
        
        if (!gpu_structs_metal_source || !oscillator_metal_source ||
            !spectral_segment_math_metal_source) {
            SPECTRAL_WARN("Metal: generated MSL source is NULL (payload TU not linked?)");
            return;
        }

        /* Combine shader sources: generated structs + generated oscillator +
         * generated segment math + kernel.  The first three strings are
         * codegen'd from the C contract (spectral_osc_metal_generated.h). */
        NSString* source = [NSString stringWithFormat:@"%s%s%s%s",
                           gpu_structs_metal_source,
                           oscillator_metal_source,
                           spectral_segment_math_metal_source,
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
        size_t output_size = 0;
        SpectralError return_err = SPECTRAL_OK;
        id<MTLBuffer> segBufForDispatch = nil;
        int seg_buf_is_nocopy = 0;
        SpectralGpuDispatchPlan plan = {0};
        id<MTLBuffer> paramsBuffer = nil;
        id<MTLCommandBuffer> cmdBuffer = nil;
        id<MTLComputeCommandEncoder> encoder = nil;

        SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
        if (!pf.ok) return pf.error;
        if (!spectral_array_bytes(out_len, sizeof(float), &output_size)) {
            *t_synth = 0;
            return SPECTRAL_ERR_OVERFLOW;
        }

        {
            int continue_backend = 1;
            SpectralError gate_err = gpu_check_timbre_or_fallback(
                "Metal", sa, out_buffer, out_len, stretch, pitch, timbre,
                spectral_omp_effective_thread_count(), t_synth, &continue_backend);
            if (gate_err != SPECTRAL_OK) return gate_err;
            if (!continue_backend) return SPECTRAL_OK;
        }

        return_err = spectral_gpu_dispatch_plan_init(&plan, sa, &pf.params, stretch, timbre, out_len);
        if (return_err != SPECTRAL_OK) goto cleanup;

        if (plan.zero_output) {
            memset(out_buffer, 0, output_size);
            *t_synth = omp_get_wtime() - pf.start_time;
            goto cleanup;
        }

        if (plan.segment_source) {
            g_mtl.segBufNoCopy = [metalDevice
                newBufferWithBytesNoCopy:(void*)plan.segment_source
                                 length:plan.segment_bytes
                                options:MTLResourceStorageModeShared
                            deallocator:nil];
            if (g_mtl.segBufNoCopy) {
                segBufForDispatch = g_mtl.segBufNoCopy;
                seg_buf_is_nocopy = 1;
            }
        }

        if (!segBufForDispatch) {
            if (!metal_grow_buffer(metalDevice, &g_mtl.segBuf, &g_mtl.segCap, plan.segment_bytes)) {
                return_err = SPECTRAL_ERR_MEMORY;
                goto cleanup;
            }
            spectral_segment_pack_gpu_array(sa.segs, sa.count, (SegmentGpu*)[g_mtl.segBuf contents]);
            segBufForDispatch = g_mtl.segBuf;
        }

#if SPECTRAL_DEBUG && !defined(NDEBUG)
        {
            float avg_segs = (float)plan.tiles.total_refs / plan.tiles.num_tiles;
            SPECTRAL_DBG("Metal: %u segs, %u tiles, avg %.0f segs/tile",
                         sa.count, plan.tiles.num_tiles, avg_segs);
        }
#endif

        if (!metal_grow_buffer(metalDevice, &g_mtl.tileIdsBuf, &g_mtl.tileIdsCap, plan.tile_ids_bytes) ||
            !metal_grow_buffer(metalDevice, &g_mtl.tileRangesBuf, &g_mtl.tileRangesCap, plan.tile_ranges_bytes) ||
            !metal_grow_buffer(metalDevice, &g_mtl.outputBuf, &g_mtl.outputCap, output_size)) {
            return_err = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }

        memcpy([g_mtl.tileIdsBuf contents], plan.tiles.segment_ids, plan.tile_ids_bytes);
        memcpy([g_mtl.tileRangesBuf contents], plan.tiles.ranges, plan.tile_ranges_bytes);

        paramsBuffer = [metalDevice newBufferWithBytes:&plan.params
                                                length:sizeof(plan.params)
                                               options:MTLResourceStorageModeShared];
        if (!paramsBuffer) {
            return_err = SPECTRAL_ERR_GPU_INIT;
            goto cleanup;
        }

        cmdBuffer = [metalQueue commandBuffer];
        if (!cmdBuffer) {
            return_err = SPECTRAL_ERR_GPU_INIT;
            goto cleanup;
        }
        encoder = [cmdBuffer computeCommandEncoder];
        if (!encoder) {
            return_err = SPECTRAL_ERR_GPU_INIT;
            goto cleanup;
        }

        [encoder setComputePipelineState:metalSynthPipeline];
        [encoder setBuffer:segBufForDispatch offset:0 atIndex:0];
        [encoder setBuffer:g_mtl.tileIdsBuf offset:0 atIndex:1];
        [encoder setBuffer:g_mtl.tileRangesBuf offset:0 atIndex:2];
        [encoder setBuffer:g_mtl.outputBuf offset:0 atIndex:3];
        [encoder setBuffer:paramsBuffer offset:0 atIndex:4];

        [encoder dispatchThreadgroups:MTLSizeMake(plan.tiles.num_tiles, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(SPECTRAL_GPU_TILE_SIZE, 1, 1)];
        [encoder endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        if ([cmdBuffer status] != MTLCommandBufferStatusCompleted) {
            if ([cmdBuffer error]) {
                SPECTRAL_WARN("Metal: command buffer failed: %s",
                              [[[cmdBuffer error] localizedDescription] UTF8String]);
            } else {
                SPECTRAL_WARN("Metal: command buffer failed");
            }
            return_err = SPECTRAL_ERR_GPU_INIT;
            goto cleanup;
        }

        memcpy(out_buffer, [g_mtl.outputBuf contents], output_size);
        *t_synth = omp_get_wtime() - pf.start_time;

    cleanup:
        if (return_err != SPECTRAL_OK) {
            memset(out_buffer, 0, output_size);
            *t_synth = 0;
        }
        spectral_gpu_dispatch_plan_free(&plan);
        if (seg_buf_is_nocopy) g_mtl.segBufNoCopy = nil;

        return return_err;
    }
}
void metal_cleanup(void) {
    g_mtl = (MetalBufferCache){0};
}
