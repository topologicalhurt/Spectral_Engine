/* spectral_synth_cpu.c - CPU Synthesis Backend
 * 
 * Multi-threaded additive synthesis using OpenMP.
 * Each thread accumulates into a private buffer, then all buffers
 * are reduced (summed) into the final output.
 * 
 * Optimization Strategies:
 *   - Thread-local buffers avoid synchronization in inner loop
 *   - Cache-aligned allocation for buffer access
 *   - SIMD hints (#pragma omp simd) for sine synthesis
 *   - Segment prefetching to hide memory latency
 *   - vDSP vector add for final buffer reduction (macOS)
 * 
 * Timbre Support:
 *   - Timbre 0 (sine): uses fast_sin() with SIMD
 *   - Timbres 1-7: scalar timbre_oscillator() function
 *   - Wavetable: LUT-based lookup with interpolation
 */

#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include <omp.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#if SPECTRAL_USE_VDSP
#include <Accelerate/Accelerate.h>
#endif

/* Backend queries */

int spectral_backend_supports_timbre(SynthBackend backend, int timbre_id) {
    switch (backend) {
        case BACKEND_CPU:   return (timbre_id >= TIMBRE_MIN && timbre_id <= BACKEND_CPU_TIMBRE_MAX);
        case BACKEND_METAL: return (timbre_id >= TIMBRE_MIN && timbre_id <= BACKEND_METAL_TIMBRE_MAX);
        case BACKEND_CUDA:  return (timbre_id >= TIMBRE_MIN && timbre_id <= BACKEND_CUDA_TIMBRE_MAX);
        default:            return 1;
    }
}

int spectral_backend_max_timbre(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:   return BACKEND_CPU_TIMBRE_MAX;
        case BACKEND_METAL: return BACKEND_METAL_TIMBRE_MAX;
        case BACKEND_CUDA:  return BACKEND_CUDA_TIMBRE_MAX;
        default:            return TIMBRE_MAX;
    }
}

const char* spectral_backend_name(SynthBackend backend) {
    static const char* names[] = {"Auto", "CPU", "Metal", "CUDA", "Export"};
    return (backend <= BACKEND_EXPORT) ? names[backend] : "Unknown";
}

int spectral_backend_supports_wavetable(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:   return BACKEND_CPU_WAVETABLE_SUPPORT;
        case BACKEND_METAL: return BACKEND_METAL_WAVETABLE_SUPPORT;
        case BACKEND_CUDA:  return BACKEND_CUDA_WAVETABLE_SUPPORT;
        default:            return 0;
    }
}

int spectral_backend_available(SynthBackend backend) {
    switch (backend) {
        case BACKEND_CPU:
            return 1;  /* Always available */
        case BACKEND_METAL:
            #if HAS_METAL
            return metal_available();
            #else
            return 0;
            #endif
        case BACKEND_CUDA:
            #if HAS_CUDA
            return cuda_available();
            #else
            return 0;
            #endif
        case BACKEND_EXPORT:
            return 1;
        case BACKEND_AUTO:
            return 1;
        default:
            return 0;
    }
}

SpectralBackendCaps spectral_backend_get_caps(SynthBackend backend) {
    SpectralBackendCaps caps = {0};
    caps.id = backend;
    caps.name = spectral_backend_name(backend);
    caps.available = spectral_backend_available(backend);
    caps.max_timbre = spectral_backend_max_timbre(backend);
    caps.has_wavetable = spectral_backend_supports_wavetable(backend);
    
    switch (backend) {
        case BACKEND_CPU:
            caps.is_gpu = 0;
            caps.is_parallel = 1;
            caps.max_segments = 0;     /* Limited only by system memory */
            caps.max_output_len = 0;   /* Unlimited */
            break;
        case BACKEND_METAL:
            caps.is_gpu = 1;
            caps.is_parallel = 1;
            caps.max_segments = 0;
            caps.max_output_len = 0;
            break;
        case BACKEND_CUDA:
            caps.is_gpu = 1;
            caps.is_parallel = 1;
            caps.max_segments = 0;      /* Limited only by VRAM */
            caps.max_output_len = 0;
            break;
        case BACKEND_EXPORT:
            caps.is_gpu = 0;
            caps.is_parallel = 0;
            caps.max_segments = 0;
            caps.max_output_len = 0;
            break;
        default:
            break;
    }
    return caps;
}

SynthBackend spectral_backend_select_for_timbre(int timbre_id, int prefer_gpu) {
    if (prefer_gpu) {
        /* Try Metal first (more timbre support than CUDA) */
        #if HAS_METAL
        if (metal_available() && spectral_backend_supports_timbre(BACKEND_METAL, timbre_id)) {
            return BACKEND_METAL;
        }
        #endif
        #if HAS_CUDA
        if (cuda_available() && spectral_backend_supports_timbre(BACKEND_CUDA, timbre_id)) {
            return BACKEND_CUDA;
        }
        #endif
    }
    /* Fall back to CPU which supports all timbres */
    return BACKEND_CPU;
}

/* Thread buffer management
 * 
 * Uses a single contiguous arena allocation for all thread buffers.
 * Benefits:
 *   - Single syscall instead of n_threads allocations
 *   - Better cache locality (contiguous memory)
 *   - Simpler cleanup on failure
 *   - Each buffer is cache-line aligned within the arena
 */

typedef struct {
    void*  arena;       /* Single contiguous allocation */
    void** bufs;        /* Pointers into arena for each thread */
    int    n_threads;
    size_t buf_stride;  /* Distance between thread buffers (aligned) */
    size_t buf_size;    /* Actual used size per buffer */
} ThreadBuffers;

static ThreadBuffers thread_buffers_alloc(int n_threads, size_t out_len, size_t element_size) {
    ThreadBuffers tb = {0};
    
    /* Calculate buffer size and aligned stride */
    tb.buf_size = out_len * element_size;
    tb.buf_stride = (tb.buf_size + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1);
    tb.n_threads = n_threads;
    
    /* Allocate pointer array */
    tb.bufs = malloc(n_threads * sizeof(void*));
    if (!tb.bufs) {
        tb.n_threads = 0;
        return tb;
    }
    
    /* Single arena allocation for all thread buffers */
    size_t arena_size = tb.buf_stride * n_threads + CACHE_ALIGN;
    tb.arena = malloc(arena_size);
    if (!tb.arena) {
        free(tb.bufs);
        tb.bufs = NULL;
        tb.n_threads = 0;
        return tb;
    }
    
    /* Zero the entire arena at once */
    memset(tb.arena, 0, arena_size);
    
    /* Align arena start and set up per-thread buffer pointers */
    char* aligned_base = (char*)(((uintptr_t)tb.arena + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1));
    for (int t = 0; t < n_threads; t++) {
        tb.bufs[t] = aligned_base + (t * tb.buf_stride);
    }
    
    return tb;
}

static void thread_buffers_free(ThreadBuffers* tb) {
    if (!tb) return;
    free(tb->arena);   /* Single deallocation */
    free(tb->bufs);
    tb->arena = NULL;
    tb->bufs = NULL;
    tb->n_threads = 0;
}

static void thread_buffers_reduce_float(const ThreadBuffers* tb, float* out_buffer, size_t out_len) {
#if SPECTRAL_USE_VDSP
    memcpy(out_buffer, tb->bufs[0], out_len * sizeof(float));
    for (int t = 1; t < tb->n_threads; t++) {
        vDSP_vadd(out_buffer, 1, (float*)tb->bufs[t], 1, out_buffer, 1, out_len);
    }
#else
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < out_len; j++) {
        float sum = ((float*)tb->bufs[0])[j];
        for (int t = 1; t < tb->n_threads; t++) sum += ((float*)tb->bufs[t])[j];
        out_buffer[j] = sum;
    }
#endif
}

static void thread_buffers_reduce_native(const ThreadBuffers* tb, spectral_sample_t* out_buffer, size_t out_len) {
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < out_len; j++) {
        spectral_sample_t sum = ((spectral_sample_t*)tb->bufs[0])[j];
        for (int t = 1; t < tb->n_threads; t++) {
            sum = SPECTRAL_SAMPLE_ADD(sum, ((spectral_sample_t*)tb->bufs[t])[j]);
        }
        out_buffer[j] = sum;
    }
}

/* Float output synthesis */

int synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len, 
              float stretch, float pitch, SpectralTimbre timbre, int n_threads,
              double* t_synth) {
    
    if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;  /* Early exit is not an error */
    }
    
    SynthParams params = make_synth_params(stretch, pitch, out_len, sa.count);
    double synth_start = omp_get_wtime();
    ThreadBuffers tb = thread_buffers_alloc(n_threads, out_len, sizeof(float));
    if (!tb.bufs) {
        memset(out_buffer, 0, out_len * sizeof(float));
        *t_synth = 0;
        return SPECTRAL_ERR_MEMORY;
    }
    
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        float* __restrict__ dst_base = (float*)tb.bufs[tid];
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            if (i + 4 < sa.count) PREFETCH_READ(&sa.segs[i + 4]);
            
            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;
            
            float* __restrict__ dst = dst_base + lp.start_idx;
            
            /* NOTE: For SIMD functions, we are basically guaranteed to oversaturate the vector units
            when thread count > core count since SIMD is per core on most CPUS */
            timbre_synth_segment(dst, &lp, timbre);
        }
    }
    
    thread_buffers_reduce_float(&tb, out_buffer, out_len);
    thread_buffers_free(&tb);
    *t_synth = omp_get_wtime() - synth_start;
    return SPECTRAL_OK;
}

/* Wavetable synthesis */

int synth_cpu_wavetable(SegmentArray sa, float* out_buffer, size_t out_len,
                        float stretch, float pitch,
                        const SpectralWavetableBank* bank, SpectralTimbre timbre,
                        int n_threads, double* t_synth) {
    
    /* No wavetable bank - fall back to regular synthesis */
    if (!bank) {
        return synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }
    
    if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;
    }
    
    const SpectralWavetable* table = spectral_wavetable_get(bank, (uint8_t)timbre);
    if (!table || !table->valid) {
        return synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }
    
    SynthParams params = make_synth_params(stretch, pitch, out_len, sa.count);
    double synth_start = omp_get_wtime();
    ThreadBuffers tb = thread_buffers_alloc(n_threads, out_len, sizeof(float));
    if (!tb.bufs) {
        memset(out_buffer, 0, out_len * sizeof(float));
        *t_synth = 0;
        return SPECTRAL_ERR_MEMORY;
    }
    
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        float* __restrict__ dst_base = (float*)tb.bufs[tid];
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            if (i + 4 < sa.count) PREFETCH_READ(&sa.segs[i + 4]);
            
            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;
            
            float* __restrict__ dst = dst_base + lp.start_idx;
            
            for (size_t j = 0; j < lp.length; j++) {
                float p = lp.phase + j * (lp.alpha + lp.beta * j);
                float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
                spectral_sample_t sample = spectral_wavetable_lookup_f(table, phase_norm);
                dst[j] += (lp.amp + lp.d_amp * j) * SPECTRAL_SAMPLE_TO_FLOAT(sample);
            }
        }
    }
    
    thread_buffers_reduce_float(&tb, out_buffer, out_len);
    thread_buffers_free(&tb);
    *t_synth = omp_get_wtime() - synth_start;
    return SPECTRAL_OK;
}

/* Native sample type synthesis */

void synth_cpu_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                      float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                      double* t_synth) {
    
    if (!SYNTH_VALIDATE_NATIVE(out_buffer, out_len, sa, &t_synth)) {
        return;
    }
    
    SynthParams params = make_synth_params(stretch, pitch, out_len, sa.count);
    double synth_start = omp_get_wtime();
    ThreadBuffers tb = thread_buffers_alloc(n_threads, out_len, sizeof(spectral_sample_t));
    if (!tb.bufs) {
        memset(out_buffer, 0, out_len * sizeof(spectral_sample_t));
        *t_synth = 0;
        return;
    }
    
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        spectral_sample_t* __restrict__ dst_base = (spectral_sample_t*)tb.bufs[tid];
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            if (i + 4 < sa.count) PREFETCH_READ(&sa.segs[i + 4]);
            
            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;
            
            spectral_sample_t* __restrict__ dst = dst_base + lp.start_idx;
            
            for (size_t j = 0; j < lp.length; j++) {
                float p = lp.phase + j * (lp.alpha + lp.beta * j);
                float sample_f = timbre_oscillator(p, lp.amp + lp.d_amp * j, timbre, lp.width);
                dst[j] = SPECTRAL_SAMPLE_ADD(dst[j], FLOAT_TO_SPECTRAL_SAMPLE(sample_f));
            }
        }
    }
    
    thread_buffers_reduce_native(&tb, out_buffer, out_len);
    thread_buffers_free(&tb);
    *t_synth = omp_get_wtime() - synth_start;
}

void synth_cpu_wavetable_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                                float stretch, float pitch,
                                const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                int n_threads, double* t_synth) {
    
    /* No wavetable bank - fall back to regular synthesis */
    if (!bank) {
        synth_cpu_native(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
        return;
    }
    
    if (!SYNTH_VALIDATE_NATIVE(out_buffer, out_len, sa, &t_synth)) {
        return;
    }
    
    const SpectralWavetable* table = spectral_wavetable_get(bank, (uint8_t)timbre);
    if (!table || !table->valid) {
        synth_cpu_native(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
        return;
    }
    
    SynthParams params = make_synth_params(stretch, pitch, out_len, sa.count);
    double synth_start = omp_get_wtime();
    ThreadBuffers tb = thread_buffers_alloc(n_threads, out_len, sizeof(spectral_sample_t));
    if (!tb.bufs) {
        memset(out_buffer, 0, out_len * sizeof(spectral_sample_t));
        *t_synth = 0;
        return;
    }
    
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        spectral_sample_t* __restrict__ dst_base = (spectral_sample_t*)tb.bufs[tid];
        
        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            if (i + 4 < sa.count) PREFETCH_READ(&sa.segs[i + 4]);
            
            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;
            
            spectral_sample_t* __restrict__ dst = dst_base + lp.start_idx;
            
            for (size_t j = 0; j < lp.length; j++) {
                float p = lp.phase + j * (lp.alpha + lp.beta * j);
                float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
                
                spectral_sample_t sample = spectral_wavetable_lookup_f(table, phase_norm);
                spectral_sample_t amp_native = FLOAT_TO_SPECTRAL_SAMPLE(lp.amp + lp.d_amp * j);
                spectral_sample_t scaled = SPECTRAL_SAMPLE_MUL(sample, amp_native);
                
                dst[j] = SPECTRAL_SAMPLE_ADD(dst[j], scaled);
            }
        }
    }
    
    thread_buffers_reduce_native(&tb, out_buffer, out_len);
    thread_buffers_free(&tb);
    *t_synth = omp_get_wtime() - synth_start;
}
