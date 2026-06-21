/* GPU Q15 double-pack audit bench (host/Apple/Metal only; manual benchmark).
 *
 * Reproduction harness for docs/core_audit/archive/GPU_Q15_DOUBLEPACK_AUDIT.md.
 * Answers the measure-first question: on this GPU, does packing two 16-bit lanes per
 * 32-bit register (Metal `half` / `half2`) make the synth oscillator faster than the
 * shipping fp32 kernel — by enough to justify half's precision loss?
 *
 * It JIT-compiles MSL via newLibraryWithSource (exactly as production
 * spectral_synth_metal.m does — the offline `metal` tool is absent on CLT-only Macs,
 * but the runtime compiler in Metal.framework is present), then times on the GPU:
 *   (A) PURE sin throughput   — sin_f32 vs sin_f16 vs sin_f16x2  (isolates the SFU question)
 *   (B) FAITHFUL synth inner   — synth_f32 vs synth_f16x2 (fp32 phase + fp32 accumulate,
 *       only the oscillator sin narrowed to half — the real Q-island upper bound)
 * and reports GPU time, speedup, and precision (max|diff| + RMS dBFS) of f16x2 vs f32.
 *
 * Run: cmake --build build --target bench_metal_q15pack
 *      && build/bin/bench_metal_q15pack
 *
 * NO production code is wired; this only measures, per measure-first / decline-on-data. */
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <math.h>
#include <stdio.h>

/* ---- MSL source (JIT-compiled at runtime, like production) -------------------------
 * Constants mirror spectral_consts.h / spectral_osc_metal_generated.h. The oscillator
 * sin is the hardware sin() per GPU policy. Phase is computed fp32 (it accumulates over
 * a segment); only the waveform sin is narrowed to half in the packed variants. */
static const char* kMSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"#define TWO_PI 6.283185307179586f\n"
"#define INV_TWO_PI 0.159154943091895f\n"
"inline float norm_phase(float p){ return p - TWO_PI*floor(p*INV_TWO_PI + 0.5f); }\n"
"\n"
/* (A) PURE sin throughput: K sins per thread, accumulate, write once. ALU/SFU-bound. */
"kernel void sin_f32(device float* out [[buffer(0)]], constant uint& K [[buffer(1)]],\n"
"                    uint gid [[thread_position_in_grid]]) {\n"
"    float x = (float)gid * 1e-4f + 0.123f; float acc = 0.0f;\n"
"    for (uint i=0;i<K;i++){ x += 0.000113f; acc += sin(norm_phase(x*(1.0f+0.001f*i))); }\n"
"    out[gid] = acc;\n"
"}\n"
"kernel void sin_f16(device float* out [[buffer(0)]], constant uint& K [[buffer(1)]],\n"
"                    uint gid [[thread_position_in_grid]]) {\n"
"    float x = (float)gid * 1e-4f + 0.123f; float acc = 0.0f;\n"
"    for (uint i=0;i<K;i++){ x += 0.000113f; half h = (half)norm_phase(x*(1.0f+0.001f*i)); acc += (float)sin(h); }\n"
"    out[gid] = acc;\n"
"}\n"
/* half2: TWO independent phases per thread (the double-pack). Half the threads launched. */
"kernel void sin_f16x2(device float* out [[buffer(0)]], constant uint& K [[buffer(1)]],\n"
"                      uint gid [[thread_position_in_grid]]) {\n"
"    float x0 = (float)(2*gid) * 1e-4f + 0.123f;\n"
"    float x1 = (float)(2*gid+1) * 1e-4f + 0.123f;\n"
"    float2 acc = 0.0f;\n"
"    for (uint i=0;i<K;i++){ x0 += 0.000113f; x1 += 0.000113f;\n"
"        half2 h = half2((half)norm_phase(x0*(1.0f+0.001f*i)), (half)norm_phase(x1*(1.0f+0.001f*i)));\n"
"        half2 s = sin(h); acc += float2(s); }\n"
"    out[2*gid] = acc.x; out[2*gid+1] = acc.y;\n"
"}\n"
"\n"
/* (B) FAITHFUL synth inner loop: S segments summed per sample. phase_at + fade + osc.
 * fp32 phase + fp32 accumulate in BOTH; only the oscillator sin narrows in f16x2. */
"struct Seg { float phase, alpha, beta, amp, da; };\n"
"inline float fade_env(float j, float seg_len){\n"
"    float fl = min(seg_len*0.25f, 64.0f); if (fl<1.0f) fl=1.0f; float inv=1.0f/fl;\n"
"    if (j<fl) return 0.5f*(1.0f+sin((j*inv-0.5f)*3.14159265f));\n"
"    float fe = seg_len-1.0f-j; if (fe<fl) return 0.5f*(1.0f+sin((fe*inv-0.5f)*3.14159265f));\n"
"    return 1.0f; }\n"
"kernel void synth_f32(device float* out [[buffer(0)]], device const Seg* segs [[buffer(1)]],\n"
"                      constant uint& S [[buffer(2)]], constant float& seglen [[buffer(3)]],\n"
"                      uint gid [[thread_position_in_grid]]) {\n"
"    float j = (float)gid; float sum = 0.0f;\n"
"    for (uint s=0;s<S;s++){ Seg g = segs[s];\n"
"        float p = g.phase + j*(g.alpha + g.beta*j);\n"
"        sum += (g.amp + g.da*j) * fade_env(j, seglen) * sin(norm_phase(p)); }\n"
"    out[gid] = sum;\n"
"}\n"
"kernel void synth_f16x2(device float* out [[buffer(0)]], device const Seg* segs [[buffer(1)]],\n"
"                        constant uint& S [[buffer(2)]], constant float& seglen [[buffer(3)]],\n"
"                        uint gid [[thread_position_in_grid]]) {\n"
"    float j0 = (float)(2*gid); float j1 = (float)(2*gid+1);\n"
"    float2 sum = 0.0f;\n"
"    for (uint s=0;s<S;s++){ Seg g = segs[s];\n"
"        float p0 = g.phase + j0*(g.alpha + g.beta*j0);\n"
"        float p1 = g.phase + j1*(g.alpha + g.beta*j1);\n"
"        half2 wv = sin(half2((half)norm_phase(p0), (half)norm_phase(p1)));\n"
"        sum.x += (g.amp + g.da*j0) * fade_env(j0, seglen) * (float)wv.x;\n"
"        sum.y += (g.amp + g.da*j1) * fade_env(j1, seglen) * (float)wv.y; }\n"
"    out[2*gid] = sum.x; out[2*gid+1] = sum.y;\n"
"}\n";

static double gpu_time_ms(id<MTLCommandBuffer> cb) {
    return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
}

int main(void) {
@autoreleasepool {
    id<MTLDevice> dev = nil;
    NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();   /* like production spectral_synth_metal.m */
    if (all && [all count] > 0) dev = all[0];
    if (!dev) dev = MTLCreateSystemDefaultDevice();
    if (!dev) { printf("no Metal device (GPU access blocked? try outside sandbox)\n"); return 1; }
    printf("\n=== Phase 5 GPU Q15 double-pack audit (%s, Metal) ===\n\n", [[dev name] UTF8String]);

    id<MTLCommandQueue> q = [dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* opt = [[MTLCompileOptions alloc] init];
    opt.fastMathEnabled = NO;               /* match production default */
    opt.languageVersion = MTLLanguageVersion2_4;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMSL]
                                           options:opt error:&err];
    if (!lib) { printf("compile error: %s\n", [[err localizedDescription] UTF8String]); return 1; }

    NSArray* names = @[@"sin_f32", @"sin_f16", @"sin_f16x2", @"synth_f32", @"synth_f16x2"];
    NSMutableDictionary* pipe = [NSMutableDictionary dictionary];
    for (NSString* n in names) {
        id<MTLFunction> f = [lib newFunctionWithName:n];
        id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
        if (!p) { printf("pipeline %s error\n", [n UTF8String]); return 1; }
        pipe[n] = p;
    }

    const uint32_t N = 1u << 20;            /* 1,048,576 output samples */
    const int REP = 30;                     /* GPU-timed repetitions (min taken) */
    id<MTLBuffer> outBuf = [dev newBufferWithLength:N * sizeof(float)
                                            options:MTLResourceStorageModeShared];

    /* ---------- (A) pure sin throughput ---------- */
    printf("(A) PURE sin throughput  (K sins/thread, N=%u threads-equiv)\n", N);
    printf("%-12s %10s %10s %9s\n", "kernel", "K", "GPU ms", "speedup");
    uint32_t Ks[] = { 64, 256 };
    for (int ki = 0; ki < 2; ki++) {
        uint32_t K = Ks[ki];
        double tf32 = 1e30, tf16 = 1e30, tf16x2 = 1e30;
        for (int variant = 0; variant < 3; variant++) {
            NSString* n = names[variant];
            id<MTLComputePipelineState> p = pipe[n];
            uint32_t threads = (variant == 2) ? (N / 2) : N;   /* half2 -> half the threads */
            double best = 1e30;
            for (int r = 0; r < REP; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p];
                [e setBuffer:outBuf offset:0 atIndex:0];
                [e setBytes:&K length:sizeof(K) atIndex:1];
                MTLSize tg = MTLSizeMake(256, 1, 1);
                MTLSize grid = MTLSizeMake((threads + 255) / 256 * 256, 1, 1);
                [e dispatchThreads:grid threadsPerThreadgroup:tg];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double ms = gpu_time_ms(cb);
                if (ms < best) best = ms;
            }
            if (variant == 0) tf32 = best; else if (variant == 1) tf16 = best; else tf16x2 = best;
        }
        printf("%-12s %10u %10.4f %8s\n", "sin_f32",   K, tf32,   "1.00x");
        printf("%-12s %10u %10.4f %8.2fx\n", "sin_f16",   K, tf16,   tf32 / tf16);
        printf("%-12s %10u %10.4f %8.2fx\n", "sin_f16x2", K, tf16x2, tf32 / tf16x2);
    }

    /* ---------- (B) faithful synth inner loop + precision ---------- */
    printf("\n(B) FAITHFUL synth  (S segments/sample, fp32 phase+accum, osc-sin narrowed)\n");
    printf("%-12s %8s %10s %9s  %-11s %-11s\n", "kernel", "S", "GPU ms", "speedup", "max|diff|", "rms dBFS");
    uint32_t Ss[] = { 64, 256 };
    for (int si = 0; si < 2; si++) {
        uint32_t S = Ss[si];
        float seglen = (float)N;
        id<MTLBuffer> segBuf = [dev newBufferWithLength:S * 5 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        float* seg = (float*)segBuf.contents;
        for (uint32_t s = 0; s < S; s++) {
            float omega = 0.01f + 0.45f * (float)s / (float)S;   /* spread partials */
            seg[s*5+0] = 0.1f * s;            /* phase  */
            seg[s*5+1] = omega;               /* alpha  */
            seg[s*5+2] = 1e-9f;               /* beta (tiny chirp) */
            seg[s*5+3] = 0.5f / (1.0f + s);   /* amp (1/n rolloff) */
            seg[s*5+4] = 0.0f;                /* da     */
        }
        id<MTLBuffer> outF32 = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> outF16 = [dev newBufferWithLength:N*sizeof(float) options:MTLResourceStorageModeShared];

        double tf32 = 1e30, tf16x2 = 1e30;
        for (int variant = 0; variant < 2; variant++) {
            NSString* n = (variant == 0) ? @"synth_f32" : @"synth_f16x2";
            id<MTLComputePipelineState> p = pipe[n];
            id<MTLBuffer> ob = (variant == 0) ? outF32 : outF16;
            uint32_t threads = (variant == 1) ? (N / 2) : N;
            double best = 1e30;
            for (int r = 0; r < REP; r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p];
                [e setBuffer:ob offset:0 atIndex:0];
                [e setBuffer:segBuf offset:0 atIndex:1];
                [e setBytes:&S length:sizeof(S) atIndex:2];
                [e setBytes:&seglen length:sizeof(seglen) atIndex:3];
                MTLSize tg = MTLSizeMake(256, 1, 1);
                MTLSize grid = MTLSizeMake((threads + 255) / 256 * 256, 1, 1);
                [e dispatchThreads:grid threadsPerThreadgroup:tg];
                [e endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                double ms = gpu_time_ms(cb);
                if (ms < best) best = ms;
            }
            if (variant == 0) tf32 = best; else tf16x2 = best;
        }
        /* precision: f16x2 vs f32 */
        const float* a = (const float*)outF32.contents;
        const float* b = (const float*)outF16.contents;
        double mx = 0.0, ss = 0.0;
        for (uint32_t i = 0; i < N; i++) {
            double d = (double)a[i] - (double)b[i];
            double ad = fabs(d); if (ad > mx) mx = ad; ss += d*d;
        }
        double rms = sqrt(ss / (double)N);
        double rms_dbfs = 20.0 * log10(rms > 1e-30 ? rms : 1e-30);
        printf("%-12s %8u %10.4f %8s  %-11s %-11s\n", "synth_f32", S, tf32, "1.00x", "-", "-");
        printf("%-12s %8u %10.4f %8.2fx  %-11.3e % -10.1f\n",
               "synth_f16x2", S, tf16x2, tf32 / tf16x2, mx, rms_dbfs);
    }
    printf("\n(speedup >1 means the half2 double-pack is faster; rms dBFS is f16x2 vs f32 output)\n");
}
    return 0;
}
