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
#include "spectral_segment_math.h"
#include "spectral_osc_formulas.h"

/* Compile-time parity guard: if the canonical C formulas change, bump their
 * version constant and update the MSL strings below to match.  A mismatch
 * here means the Metal shader is out of sync with CPU/CUDA backends. */
_Static_assert(SPECTRAL_SEGMENT_MATH_VERSION == 1,
    "Metal segment_math MSL strings are stale — update metalKernelCode and bump version");
_Static_assert(SPECTRAL_OSC_FORMULAS_VERSION == 1,
    "Metal oscillator MSL strings are stale — update oscillator_metal_source and bump version");

/* Metal kernel source - struct definitions and synthesis kernel.
 * Oscillator functions come from oscillator_metal_source (defined in oscillator.c).
 * Uses SegmentGpu (32 bytes) for parity with CUDA — doubles threadgroup cache. */
static const char* metalKernelStructs = 
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
/* NOTE: Must match SegmentGpu layout in spectral_common.h (no compile-time check possible) */
"struct SegmentGpu {\n"
"    float start;\n"
"    float length;\n"
"    float phase;\n"
"    float omega;\n"
"    float df;\n"
"    float amp;\n"
"    float da;\n"
"    float _pad;\n"
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
"#define SEGMENT_CACHE_SIZE " SPECTRAL_STR(SPECTRAL_GPU_SEG_CACHE_SIZE) "\n"
"/* 32-byte SegmentGpu = 2x entries in same threadgroup budget */\n"
"\n"
"/* Backend parity note:\n"
" * MSL source is compiled from this runtime string, so it cannot include\n"
" * core/spectral_segment_math.h directly.  Keep these helpers in lockstep\n"
" * with spectral_segment_{alpha,beta,d_amp,phase_at,amp_at}_f32 formulas.\n"
" *\n"
" * WARNING: any change to spectral_segment_math.h MUST be mirrored here.\n"
" * CUDA includes the header directly; Metal cannot.  If formulas drift,\n"
" * Metal output will silently differ from CPU/CUDA. */\n"
"inline float segment_alpha(float omega, float pitch_factor, float inv_stretch) {\n"
"    return omega * pitch_factor * inv_stretch;\n"
"}\n"
"inline float segment_beta(float df, float pitch_factor, float inv_stretch_sq) {\n"
"    return df * pitch_factor * inv_stretch_sq;\n"
"}\n"
"inline float segment_d_amp(float da, float inv_stretch) {\n"
"    return da * inv_stretch;\n"
"}\n"
"inline float segment_phase_at(float phase0, float alpha, float beta, float sample_offset) {\n"
"    return phase0 + sample_offset * (alpha + beta * sample_offset);\n"
"}\n"
"inline float segment_amp_at(float amp0, float d_amp, float sample_offset) {\n"
"    return amp0 + d_amp * sample_offset;\n"
"}\n"
" \n"
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
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion2_4;
        
        if (!oscillator_metal_source) {
            SPECTRAL_WARN("Metal: oscillator_metal_source is NULL (oscillator.c not linked?)");
            return;
        }

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
        size_t output_size = 0;

        /* Shared preflight: validate + params + timing */
        SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
        if (!pf.ok) return SPECTRAL_OK;
        if (!spectral_array_bytes(out_len, sizeof(float), &output_size)) {
            *t_synth = 0;
            return SPECTRAL_ERR_OVERFLOW;
        }

        /* Check timbre support - fall back to CPU for unsupported timbres */
        {
            int continue_backend = 1;
            SpectralError gate_err = gpu_check_timbre_or_fallback(
                "Metal", sa, out_buffer, out_len, stretch, pitch, timbre,
                omp_get_max_threads(), t_synth, &continue_backend);
            if (gate_err != SPECTRAL_OK) {
                return gate_err;
            }
            if (!continue_backend) {
                return SPECTRAL_OK;
            }
        }

        SpectralError return_err = SPECTRAL_OK;

        /* ── Segment upload ─────────────────────────────────────────────
         * Prefer pre-packed SegmentGpu from cache (zero-copy via UMA),
         * fall back to packing from the 64-byte Segment array. */
        size_t gpu_seg_size = 0;
        if (!spectral_array_bytes((size_t)sa.count, sizeof(SegmentGpu), &gpu_seg_size)) {
            return_err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }

        id<MTLBuffer> segBufForDispatch = nil;
        int seg_buf_is_nocopy = 0;
        {
            const SegmentGpu* cached_gpu_segs = NULL;
            if (gpu_seg_cache_try_get(sa.count, &cached_gpu_segs)) {
                /* Fast path: wrap mmap'd SegmentGpu directly — zero memcpy on UMA. */
                g_mtl.segBufNoCopy = [metalDevice
                    newBufferWithBytesNoCopy:(void*)cached_gpu_segs
                                     length:gpu_seg_size
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
                if (g_mtl.segBufNoCopy) {
                    segBufForDispatch = g_mtl.segBufNoCopy;
                    seg_buf_is_nocopy = 1;
                }
            }
            if (!segBufForDispatch) {
                /* Slow path: pack 64-byte Segments → 32-byte SegmentGpu into Metal buffer. */
                if (!metal_grow_buffer(metalDevice, &g_mtl.segBuf, &g_mtl.segCap, gpu_seg_size)) {
                    return_err = SPECTRAL_ERR_MEMORY;
                    goto cleanup;
                }
                /* Pack directly into UMA memory without an intermediate buffer. */
                spectral_segment_pack_gpu_array(sa.segs, sa.count, (SegmentGpu*)[g_mtl.segBuf contents]);
                segBufForDispatch = g_mtl.segBuf;
            }
        }

        /* ── Tile preprocessing (or cache) ──────────────────────────── */
        GpuTileData td = {0};
        int owns_tile_data = 0;
        SpectralError tile_err = gpu_tile_preprocess_cached(sa, stretch, SPECTRAL_GPU_TILE_SIZE, out_len, &td, &owns_tile_data);
        if (tile_err != SPECTRAL_OK) {
            return_err = tile_err;
            goto cleanup;
        }

#if SPECTRAL_DEBUG && !defined(NDEBUG)
        float avg_segs = (float)td.total_refs / td.num_tiles;
#endif
        
        GpuSynthParams params = gpu_synth_params_pack(&pf.params, SPECTRAL_GPU_TILE_SIZE, timbre);
        
        size_t tile_ids_size = 0;
        size_t tile_ranges_size = 0;
        if (!spectral_array_bytes((size_t)td.total_refs, sizeof(uint32_t), &tile_ids_size) ||
            !spectral_array_bytes((size_t)td.num_tiles, sizeof(TileRange), &tile_ranges_size)) {
            return_err = SPECTRAL_ERR_OVERFLOW;
            goto cleanup;
        }

        SPECTRAL_DBG("Metal: %u segs, %u tiles, avg %.0f segs/tile",
                sa.count, td.num_tiles, avg_segs);

        /* Grow cached Metal buffers (tile ids, tile ranges, output).
         * metal_grow_buffer sets the buffer to nil and capacity to 0 on
         * failure, so successful earlier allocations remain in the cache
         * for reuse on the next call. */
        if (!metal_grow_buffer(metalDevice, &g_mtl.tileIdsBuf, &g_mtl.tileIdsCap, tile_ids_size) ||
            !metal_grow_buffer(metalDevice, &g_mtl.tileRangesBuf, &g_mtl.tileRangesCap, tile_ranges_size) ||
            !metal_grow_buffer(metalDevice, &g_mtl.outputBuf, &g_mtl.outputCap, output_size)) {
            return_err = SPECTRAL_ERR_MEMORY;
            goto cleanup;
        }

        /* Copy tile data into cached buffers */
        memcpy([g_mtl.tileIdsBuf contents], td.segment_ids, tile_ids_size);
        memcpy([g_mtl.tileRangesBuf contents], td.ranges, tile_ranges_size);

        /* Params buffer: small (36 bytes), allocated per-call */
        id<MTLBuffer> paramsBuffer = [metalDevice newBufferWithBytes:&params
                                                              length:sizeof(params)
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> cmdBuffer = [metalQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:metalSynthPipeline];
        [encoder setBuffer:segBufForDispatch offset:0 atIndex:0];
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

    cleanup:
        if (return_err != SPECTRAL_OK) {
            memset(out_buffer, 0, output_size);
            *t_synth = 0;
        }
        if (owns_tile_data) gpu_tile_data_free(&td);
        if (seg_buf_is_nocopy) g_mtl.segBufNoCopy = nil;

        return return_err;
    }
}

void metal_cleanup(void) {
    g_mtl = (MetalBufferCache){0};
}
