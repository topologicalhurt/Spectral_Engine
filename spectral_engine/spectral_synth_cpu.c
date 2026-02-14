/* spectral_synth_cpu.c - Multi-threaded CPU synthesis with OpenMP */

#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include "spectral_vector_ops.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "spectral_omp.h"

/* Thread-local buffer arena — single contiguous allocation, cache-line aligned */

typedef struct {
    void*  arena;       /* Single contiguous allocation */
    void** bufs;        /* Pointers into arena for each thread */
    int    n_threads;
    size_t buf_stride;  /* Distance between thread buffers (aligned) */
    size_t buf_size;    /* Actual used size per buffer */
} ThreadBuffers;

static ThreadBuffers thread_buffers_alloc(int n_threads, size_t out_len, size_t element_size) {
    ThreadBuffers tb = {0};

    if (n_threads <= 0 || element_size == 0 || out_len == 0) {
        return tb;
    }
    if (out_len > SIZE_MAX / element_size) {
        return tb;
    }

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
    if (tb.buf_stride > SIZE_MAX / (size_t)n_threads) {
        free(tb.bufs);
        tb.bufs = NULL;
        tb.n_threads = 0;
        return tb;
    }
    size_t arena_size = tb.buf_stride * (size_t)n_threads + CACHE_ALIGN;
    tb.arena = malloc(arena_size);
    if (!tb.arena) {
        free(tb.bufs);
        tb.bufs = NULL;
        tb.n_threads = 0;
        return tb;
    }

    memset(tb.arena, 0, arena_size);

    /* Set up per-thread buffer pointers */
    char* aligned_base = (char*)(((uintptr_t)tb.arena + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1));
    for (int t = 0; t < n_threads; t++) {
        tb.bufs[t] = aligned_base + (t * tb.buf_stride);
    }

    return tb;
}

static void thread_buffers_free(ThreadBuffers* tb) {
    if (!tb) return;
    free(tb->arena);
    free(tb->bufs);
    tb->arena = NULL;
    tb->bufs = NULL;
    tb->n_threads = 0;
}

static void thread_buffers_reduce_float(const ThreadBuffers* tb, float* out_buffer, size_t out_len) {
    memcpy(out_buffer, tb->bufs[0], out_len * sizeof(float));
    for (int t = 1; t < tb->n_threads; t++) {
        spectral_vadd(out_buffer, (float*)tb->bufs[t], out_buffer, out_len);
    }
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

/* All CPU synth variants: validate -> alloc ThreadBuffers -> parallel loop -> reduce -> free */

typedef void (*SegmentSynthFn)(void* dst, const SegmentLoopParams* lp, const void* ctx);
typedef void (*ReduceFn)(const ThreadBuffers* tb, void* out_buffer, size_t out_len);

static SpectralError synth_cpu_driver(
    SegmentArray sa, void* out_buffer, size_t out_len, size_t elem_size,
    float stretch, float pitch, int n_threads, double* t_synth,
    SegmentSynthFn segment_fn, const void* fn_ctx,
    ReduceFn reduce_fn
) {
    SynthParams params = make_synth_params(stretch, pitch, out_len, sa.count);
    double synth_start = omp_get_wtime();
    ThreadBuffers tb = thread_buffers_alloc(n_threads, out_len, elem_size);
    if (!tb.bufs) {
        memset(out_buffer, 0, out_len * elem_size);
        *t_synth = 0;
        return SPECTRAL_ERR_MEMORY;
    }

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        char* dst_base = (char*)tb.bufs[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < sa.count; i++) {
            if (i + 4 < sa.count) PREFETCH_READ(&sa.segs[i + 4]);

            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;

            void* dst = dst_base + lp.start_idx * elem_size;
            segment_fn(dst, &lp, fn_ctx);
        }
    }

    reduce_fn(&tb, out_buffer, out_len);
    thread_buffers_free(&tb);
    *t_synth = omp_get_wtime() - synth_start;
    return SPECTRAL_OK;
}

static void reduce_float_wrapper(const ThreadBuffers* tb, void* out, size_t len) {
    thread_buffers_reduce_float(tb, (float*)out, len);
}

static void reduce_native_wrapper(const ThreadBuffers* tb, void* out, size_t len) {
    thread_buffers_reduce_native(tb, (spectral_sample_t*)out, len);
}

/* Per-segment callbacks */

typedef struct { SpectralTimbre timbre; } TimbreCtx;
typedef struct { const SpectralWavetable* table; } WavetableCtx;
typedef struct { const SpectralWavetable* table; } NativeWavetableCtx;

static void segment_fn_timbre(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const TimbreCtx* tc = (const TimbreCtx*)ctx;
    timbre_synth_segment((float*)dst, lp, tc->timbre);
}

static void segment_fn_wavetable_float(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const WavetableCtx* wc = (const WavetableCtx*)ctx;
    float* out = (float*)dst;
    for (size_t j = 0; j < lp->length; j++) {
        float p = lp->phase + j * (lp->alpha + lp->beta * j);
        float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
        spectral_sample_t sample = spectral_wavetable_lookup_f(wc->table, phase_norm);
        out[j] += (lp->amp + lp->d_amp * j) * SPECTRAL_SAMPLE_TO_FLOAT(sample);
    }
}

static void segment_fn_native_timbre(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const TimbreCtx* tc = (const TimbreCtx*)ctx;
    spectral_sample_t* out = (spectral_sample_t*)dst;
    for (size_t j = 0; j < lp->length; j++) {
        float p = lp->phase + j * (lp->alpha + lp->beta * j);
        float sample_f = timbre_oscillator(p, lp->amp + lp->d_amp * j, tc->timbre, lp->width);
        out[j] = SPECTRAL_SAMPLE_ADD(out[j], FLOAT_TO_SPECTRAL_SAMPLE(sample_f));
    }
}

static void segment_fn_native_wavetable(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const NativeWavetableCtx* nwc = (const NativeWavetableCtx*)ctx;
    spectral_sample_t* out = (spectral_sample_t*)dst;
    for (size_t j = 0; j < lp->length; j++) {
        float p = lp->phase + j * (lp->alpha + lp->beta * j);
        float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
        spectral_sample_t sample = spectral_wavetable_lookup_f(nwc->table, phase_norm);
        spectral_sample_t amp_native = FLOAT_TO_SPECTRAL_SAMPLE(lp->amp + lp->d_amp * j);
        spectral_sample_t scaled = SPECTRAL_SAMPLE_MUL(sample, amp_native);
        out[j] = SPECTRAL_SAMPLE_ADD(out[j], scaled);
    }
}

/* Public synthesis functions */

/* When SPECTRAL_USE_EMBEDDED_SYNTH is defined, synth_cpu is macro-redirected
 * to synth_arm32_emulation (defined in spectral_synth_emulator.c) */
#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
SpectralError synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len,
                        float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                        double* t_synth) {
    if (n_threads < 1) n_threads = 1;
    if (!SYNTH_VALIDATE_FLOAT(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;
    }
    TimbreCtx ctx = { .timbre = timbre };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(float),
                            stretch, pitch, n_threads, t_synth,
                            segment_fn_timbre, &ctx, reduce_float_wrapper);
}
#endif

#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
SpectralError synth_cpu_wavetable(SegmentArray sa, float* out_buffer, size_t out_len,
                                  float stretch, float pitch,
                                  const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                  int n_threads, double* t_synth) {
    if (n_threads < 1) n_threads = 1;
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
    WavetableCtx ctx = { .table = table };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(float),
                            stretch, pitch, n_threads, t_synth,
                            segment_fn_wavetable_float, &ctx, reduce_float_wrapper);
}
#endif

SpectralError synth_cpu_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                               float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                               double* t_synth) {
    if (n_threads < 1) n_threads = 1;
    if (!SYNTH_VALIDATE_NATIVE(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;
    }
    TimbreCtx ctx = { .timbre = timbre };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(spectral_sample_t),
                            stretch, pitch, n_threads, t_synth,
                            segment_fn_native_timbre, &ctx, reduce_native_wrapper);
}

SpectralError synth_cpu_wavetable_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                                         float stretch, float pitch,
                                         const SpectralWavetableBank* bank, SpectralTimbre timbre,
                                         int n_threads, double* t_synth) {
    if (n_threads < 1) n_threads = 1;
    if (!bank) {
        return synth_cpu_native(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }
    if (!SYNTH_VALIDATE_NATIVE(out_buffer, out_len, sa, &t_synth)) {
        return SPECTRAL_OK;
    }
    const SpectralWavetable* table = spectral_wavetable_get(bank, (uint8_t)timbre);
    if (!table || !table->valid) {
        return synth_cpu_native(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }
    NativeWavetableCtx ctx = { .table = table };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(spectral_sample_t),
                            stretch, pitch, n_threads, t_synth,
                            segment_fn_native_wavetable, &ctx, reduce_native_wrapper);
}

