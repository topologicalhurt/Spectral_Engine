/* spectral_synth_cpu.c - Multi-threaded CPU synthesis with OpenMP */

#include "spectral_synth.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_utils.h"
#include "spectral_contracts.h"
#include "spectral_vector_ops.h"
#include "spectral_wavetable.h"
#include "oscillator.h"
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include "spectral_omp.h"

#if SPECTRAL_EMBEDDED
#define SPECTRAL_SYNTH_CPU_FADE_SAMPLES SPECTRAL_FADE_SAMPLES_EMBEDDED
#else
#define SPECTRAL_SYNTH_CPU_FADE_SAMPLES SPECTRAL_FADE_SAMPLES_DESKTOP
#endif

/* Thread-local buffer arena — single contiguous allocation, cache-line aligned */

typedef struct {
    void*  arena;       /* Single contiguous allocation */
    void** bufs;        /* Pointers into arena for each thread */
    int    n_threads;
    size_t buf_stride;  /* Distance between thread buffers (aligned) */
    size_t buf_size;    /* Actual used size per buffer */
} ThreadBuffers;

static int synth_partition_count(size_t seg_count, int n_threads) {
    size_t n = seg_count;
    size_t thread_cap = (size_t)((n_threads > 0) ? n_threads : 1);
    if (n == 0) return 1;
    if (n > thread_cap) {
        n = thread_cap;
    }
#if SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS > 0
    if (n > (size_t)SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS) {
        n = (size_t)SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS;
    }
#endif
    return (int)n;
}

static ThreadBuffers thread_buffers_alloc(int n_threads, size_t out_len, size_t element_size,
                                          SpectralError* out_error) {
    ThreadBuffers tb = {0};
    size_t arena_payload_size = 0;
    size_t arena_size = 0;
    size_t out_bytes = 0;
    size_t stride_with_padding = 0;
    size_t cache_align = (size_t)SPECTRAL_CACHE_ALIGN;
    size_t align_mask = 0;

    if (out_error) *out_error = SPECTRAL_OK;
    if (!out_error) {
        return tb;
    }

    if (n_threads <= 0 || element_size == 0 || out_len == 0) {
        *out_error = SPECTRAL_ERR_PARAM;
        return tb;
    }
    if (cache_align == 0u || (cache_align & (cache_align - 1u)) != 0u) {
        *out_error = SPECTRAL_ERR_PARAM;
        return tb;
    }
    align_mask = cache_align - 1u;

    if (!spectral_size_mul(out_len, element_size, &out_bytes)) {
        *out_error = SPECTRAL_ERR_OVERFLOW;
        return tb;
    }

    /* Checked align-up: stride = align_up(out_bytes, SPECTRAL_CACHE_ALIGN).
     * The power-of-two validation above makes the mask form valid; the
     * spectral_size_add() guard proves the padding addition cannot wrap.
     *
     * Calculate buffer size and aligned stride without raw addition.
     * Pass 22 proved out_len * element_size; this pass proves the cache-line
     * padding arithmetic before allocating the per-thread arena. */
    tb.buf_size = out_bytes;
    if (!spectral_size_add(tb.buf_size, align_mask, &stride_with_padding)) {
        *out_error = SPECTRAL_ERR_OVERFLOW;
        return tb;
    }
    tb.buf_stride = stride_with_padding & ~align_mask;
    if (tb.buf_stride < tb.buf_size) {
        *out_error = SPECTRAL_ERR_OVERFLOW;
        return tb;
    }
    tb.n_threads = n_threads;

    /* Allocate pointer array */
    tb.bufs = spectral_malloc_array((size_t)n_threads, sizeof(void*));
    if (!tb.bufs) {
        tb.n_threads = 0;
        *out_error = SPECTRAL_ERR_MEMORY;
        return tb;
    }

    /* Manual aligned-base adjustment: calloc() does not promise cache-line
     * alignment, so reserve up to align_mask bytes after the checked payload.
     * That extra space is for moving the base forward, not for per-thread data. */
    if (!spectral_size_mul(tb.buf_stride, (size_t)n_threads, &arena_payload_size) ||
        !spectral_size_add(arena_payload_size, align_mask, &arena_size)) {
        free(tb.bufs);
        tb.bufs = NULL;
        tb.n_threads = 0;
        *out_error = SPECTRAL_ERR_OVERFLOW;
        return tb;
    }
    tb.arena = spectral_calloc_array(arena_size, 1u);
    if (!tb.arena) {
        free(tb.bufs);
        tb.bufs = NULL;
        tb.n_threads = 0;
        *out_error = SPECTRAL_ERR_MEMORY;
        return tb;
    }

    /* Set up per-thread buffer pointers */
    {
        uintptr_t base = (uintptr_t)tb.arena;
        uintptr_t aligned_base_u = 0u;
        char* aligned_base = NULL;

        /* Pointer-add guard: the integer addition used to compute the aligned
         * base is checked before forming `(base + align_mask)`. Without this,
         * an address near UINTPTR_MAX could wrap before the mask is applied. */
        if (base > UINTPTR_MAX - (uintptr_t)align_mask) {
            free(tb.arena);
            free(tb.bufs);
            tb = (ThreadBuffers){0};
            *out_error = SPECTRAL_ERR_OVERFLOW;
            return tb;
        }

        aligned_base_u = (base + (uintptr_t)align_mask) & ~(uintptr_t)align_mask;
        aligned_base = (char*)aligned_base_u;
        for (int t = 0; t < n_threads; t++) {
            tb.bufs[t] = aligned_base + ((size_t)t * tb.buf_stride);
        }
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

#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
#endif


#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
static SpectralError thread_buffers_reduce_float(const ThreadBuffers* tb, float* out_buffer,
                                                 size_t out_len, size_t out_bytes) {
    if (!tb || !tb->bufs || !out_buffer || out_len == 0u) {
        return SPECTRAL_ERR_PARAM;
    }
    if (tb->buf_size < out_bytes) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    memcpy(out_buffer, tb->bufs[0], out_bytes);
    for (int t = 1; t < tb->n_threads; t++) {
        spectral_vadd(out_buffer, (float*)tb->bufs[t], out_buffer, out_len);
    }
    if (!spectral_f32_span_finite(out_buffer, out_len)) {
        return SPECTRAL_ERR_PARAM;
    }
    return SPECTRAL_OK;
}

#endif

static SpectralError thread_buffers_reduce_native(const ThreadBuffers* tb, spectral_sample_t* out_buffer,
                                                  size_t out_len, size_t out_bytes) {
    if (!tb || !tb->bufs || !out_buffer || out_len == 0u) {
        return SPECTRAL_ERR_PARAM;
    }
    if (tb->buf_size < out_bytes) {
        return SPECTRAL_ERR_OVERFLOW;
    }

    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < out_len; j++) {
        spectral_sample_t sum = ((spectral_sample_t*)tb->bufs[0])[j];
        for (int t = 1; t < tb->n_threads; t++) {
            sum = SPECTRAL_SAMPLE_ADD(sum, ((spectral_sample_t*)tb->bufs[t])[j]);
        }
        out_buffer[j] = sum;
    }
    return SPECTRAL_OK;
}


/* All CPU synth variants: validate -> alloc ThreadBuffers -> parallel loop -> reduce -> free */

typedef void (*SegmentSynthFn)(void* dst, const SegmentLoopParams* lp, const void* ctx);
typedef SpectralError (*ReduceFn)(const ThreadBuffers* tb, void* out_buffer, size_t out_len, size_t out_bytes);

static SpectralError synth_cpu_driver(
    SegmentArray sa, void* out_buffer, size_t out_len, size_t elem_size,
    const SynthParams* params_in, double synth_start, int n_threads, double* t_synth,
    SegmentSynthFn segment_fn, const void* fn_ctx,
    ReduceFn reduce_fn
) {
    SynthParams params = *params_in;
    size_t out_bytes = 0;
    SpectralError tb_err = SPECTRAL_OK;
    SpectralError reduce_err = SPECTRAL_OK;
    int n_parts = synth_partition_count(sa.count, n_threads);
    ThreadBuffers tb = {0};

    if (!spectral_size_mul(out_len, elem_size, &out_bytes)) {
        *t_synth = 0;
        return SPECTRAL_ERR_OVERFLOW;
    }

    tb = thread_buffers_alloc(n_parts, out_len, elem_size, &tb_err);
    if (!tb.bufs) {
        memset(out_buffer, 0, out_bytes);
        *t_synth = 0;
        return tb_err != SPECTRAL_OK ? tb_err : SPECTRAL_ERR_MEMORY;
    }

    #pragma omp parallel for schedule(static) num_threads(n_parts)
    for (int p = 0; p < n_parts; p++) {
        size_t seg_start = ((size_t)p * sa.count) / (size_t)n_parts;
        size_t seg_end = ((size_t)(p + 1) * sa.count) / (size_t)n_parts;
        char* dst_base = (char*)tb.bufs[p];

        for (size_t i = seg_start; i < seg_end; i++) {
            if (i + 4 < seg_end) SPECTRAL_PREFETCH_READ(&sa.segs[i + 4]);

            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;

            void* dst = dst_base + lp.start_idx * elem_size;
            segment_fn(dst, &lp, fn_ctx);
        }
    }

    reduce_err = reduce_fn(&tb, out_buffer, out_len, out_bytes);
    if (reduce_err != SPECTRAL_OK) {
        memset(out_buffer, 0, out_bytes);
        thread_buffers_free(&tb);
        *t_synth = 0;
        return reduce_err;
    }
    thread_buffers_free(&tb);
    *t_synth = omp_get_wtime() - synth_start;
    return SPECTRAL_OK;
}

#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
static SpectralError reduce_float_wrapper(const ThreadBuffers* tb, void* out, size_t len, size_t out_bytes) {
    return thread_buffers_reduce_float(tb, (float*)out, len, out_bytes);
}
#endif

static SpectralError reduce_native_wrapper(const ThreadBuffers* tb, void* out, size_t len, size_t out_bytes) {
    return thread_buffers_reduce_native(tb, (spectral_sample_t*)out, len, out_bytes);
}

/* Per-segment callbacks */

typedef struct { SpectralTimbre timbre; } TimbreCtx;
#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
typedef struct { const SpectralWavetable* table; } WavetableCtx;
#endif
typedef struct { const SpectralWavetable* table; } NativeWavetableCtx;

#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
static void segment_fn_timbre(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const TimbreCtx* tc = (const TimbreCtx*)ctx;
    timbre_synth_segment((float*)dst, lp, tc->timbre);
}

static void segment_fn_wavetable_float(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const WavetableCtx* wc = (const WavetableCtx*)ctx;
    float* out = (float*)dst;
    FadeParams fp = fade_params_init(lp->length, SPECTRAL_SYNTH_CPU_FADE_SAMPLES);
    for (size_t j = 0; j < lp->length; j++) {
        float p = spectral_segment_phase_at_f32(lp->phase, lp->alpha, lp->beta, (float)j);
        float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
        float amp = spectral_segment_amp_at_f32(lp->amp, lp->d_amp, (float)j) * fade_envelope(j, &fp, lp->length);
        spectral_sample_t sample = spectral_wavetable_lookup_f(wc->table, phase_norm);
        out[j] += amp * SPECTRAL_SAMPLE_TO_FLOAT(sample);
    }
}
#endif

static void segment_fn_native_timbre(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const TimbreCtx* tc = (const TimbreCtx*)ctx;
    spectral_sample_t* out = (spectral_sample_t*)dst;
    FadeParams fp = fade_params_init(lp->length, SPECTRAL_SYNTH_CPU_FADE_SAMPLES);
    for (size_t j = 0; j < lp->length; j++) {
        float p = spectral_segment_phase_at_f32(lp->phase, lp->alpha, lp->beta, (float)j);
        float amp = spectral_segment_amp_at_f32(lp->amp, lp->d_amp, (float)j) * fade_envelope(j, &fp, lp->length);
        float sample_f = timbre_oscillator(p, amp, tc->timbre, lp->width);
        out[j] = SPECTRAL_SAMPLE_ADD(out[j], FLOAT_TO_SPECTRAL_SAMPLE(sample_f));
    }
}

static void segment_fn_native_wavetable(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const NativeWavetableCtx* nwc = (const NativeWavetableCtx*)ctx;
    spectral_sample_t* out = (spectral_sample_t*)dst;
    FadeParams fp = fade_params_init(lp->length, SPECTRAL_SYNTH_CPU_FADE_SAMPLES);
    for (size_t j = 0; j < lp->length; j++) {
        float p = spectral_segment_phase_at_f32(lp->phase, lp->alpha, lp->beta, (float)j);
        float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
        float amp = spectral_segment_amp_at_f32(lp->amp, lp->d_amp, (float)j) * fade_envelope(j, &fp, lp->length);
        spectral_sample_t sample = spectral_wavetable_lookup_f(nwc->table, phase_norm);
        spectral_sample_t amp_native = FLOAT_TO_SPECTRAL_SAMPLE(amp);
        spectral_sample_t scaled = SPECTRAL_SAMPLE_MUL(sample, amp_native);
        out[j] = SPECTRAL_SAMPLE_ADD(out[j], scaled);
    }
}

/* Public synthesis functions */

/* When SPECTRAL_USE_EMBEDDED_SYNTH is defined, synth_cpu is macro-redirected
 * to synth_arm32_simulation (defined in spectral_synth_simulation.c). */
#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
SpectralError synth_cpu(SegmentArray sa, float* out_buffer, size_t out_len,
                        float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                        double* t_synth) {
    if (n_threads < 1) n_threads = 1;
    SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
    if (!pf.ok) return pf.error;
    TimbreCtx ctx = { .timbre = timbre };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(float),
                            &pf.params, pf.start_time, n_threads, t_synth,
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
    SynthPreflight pf = synth_preflight_float(out_buffer, out_len, sa, stretch, pitch, &t_synth);
    if (!pf.ok) return pf.error;
    const SpectralWavetable* table = spectral_wavetable_get(bank, (uint8_t)timbre);
    if (!table || !table->valid) {
        return synth_cpu(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }
    WavetableCtx ctx = { .table = table };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(float),
                            &pf.params, pf.start_time, n_threads, t_synth,
                            segment_fn_wavetable_float, &ctx, reduce_float_wrapper);
}
#endif

SpectralError synth_cpu_native(SegmentArray sa, spectral_sample_t* out_buffer, size_t out_len,
                               float stretch, float pitch, SpectralTimbre timbre, int n_threads,
                               double* t_synth) {
    if (n_threads < 1) n_threads = 1;
    SynthPreflight pf = synth_preflight_native(out_buffer, out_len, sa, stretch, pitch, &t_synth);
    if (!pf.ok) return pf.error;
    TimbreCtx ctx = { .timbre = timbre };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(spectral_sample_t),
                            &pf.params, pf.start_time, n_threads, t_synth,
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
    SynthPreflight pf = synth_preflight_native(out_buffer, out_len, sa, stretch, pitch, &t_synth);
    if (!pf.ok) return pf.error;
    const SpectralWavetable* table = spectral_wavetable_get(bank, (uint8_t)timbre);
    if (!table || !table->valid) {
        return synth_cpu_native(sa, out_buffer, out_len, stretch, pitch, timbre, n_threads, t_synth);
    }
    NativeWavetableCtx ctx = { .table = table };
    return synth_cpu_driver(sa, out_buffer, out_len, sizeof(spectral_sample_t),
                            &pf.params, pf.start_time, n_threads, t_synth,
                            segment_fn_native_wavetable, &ctx, reduce_native_wrapper);
}
