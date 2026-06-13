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
    /* Output tiling: each partition's segments occupy a contiguous output
     * span [span_lo, span_lo+span_len).  Per-thread buffers are sized to the
     * widest span (buf_size) instead of the full output, and the combine pass
     * sums each span back at its offset — eliminating the O(threads*out_len)
     * private-buffer footprint and full reduce pass.  Borrowed, driver-owned. */
    const size_t* span_lo;   /* per-partition output start offset (samples) */
    const size_t* span_len;  /* per-partition span length (samples) */
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
     * spectral_size_add() guard proves the padding addition cannot wrap. */
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
static SpectralError thread_buffers_combine_float(const ThreadBuffers* tb, float* __restrict__ out_buffer,
                                                  size_t out_len, size_t out_bytes) {
    if (!tb || !tb->bufs || !out_buffer || out_len == 0u) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!tb->span_lo || !tb->span_len) {
        return SPECTRAL_ERR_PARAM;
    }

    /* Partitions tile the output: zero the whole buffer, then add each span back
     * at its offset in ascending partition order.  This reproduces the legacy
     * reduce exactly — a partition contributes 0 outside its span (an exact IEEE
     * identity under +), and the per-element accumulation order is unchanged. */
    memset(out_buffer, 0, out_bytes);
    for (int t = 0; t < tb->n_threads; t++) {
        const size_t lo = tb->span_lo[t];
        const size_t len = tb->span_len[t];
        if (len == 0u) continue;
        if (lo > out_len || len > out_len - lo) {
            return SPECTRAL_ERR_OVERFLOW;
        }
        if (len > tb->buf_size / sizeof(float)) {
            return SPECTRAL_ERR_OVERFLOW;
        }
        spectral_vadd(out_buffer + lo, (float*)tb->bufs[t], out_buffer + lo, len);
    }
    if (!spectral_f32_span_finite(out_buffer, out_len)) {
        return SPECTRAL_ERR_PARAM;
    }
    return SPECTRAL_OK;
}

#endif

static SpectralError thread_buffers_combine_native(const ThreadBuffers* tb, spectral_sample_t* __restrict__ out_buffer,
                                                   size_t out_len, size_t out_bytes) {
    if (!tb || !tb->bufs || !out_buffer || out_len == 0u) {
        return SPECTRAL_ERR_PARAM;
    }
    if (!tb->span_lo || !tb->span_len) {
        return SPECTRAL_ERR_PARAM;
    }

    memset(out_buffer, 0, out_bytes);
    for (int t = 0; t < tb->n_threads; t++) {
        const size_t lo = tb->span_lo[t];
        const size_t len = tb->span_len[t];
        if (len == 0u) continue;
        if (lo > out_len || len > out_len - lo) {
            return SPECTRAL_ERR_OVERFLOW;
        }
        if (len > tb->buf_size / sizeof(spectral_sample_t)) {
            return SPECTRAL_ERR_OVERFLOW;
        }
        const spectral_sample_t* __restrict__ src = (const spectral_sample_t*)tb->bufs[t];
        spectral_sample_t* __restrict__ dst = out_buffer + lo;
        for (size_t j = 0; j < len; j++) {
            dst[j] = spectral_sample_add(dst[j], src[j]);
        }
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
    size_t* span_lo = NULL;
    size_t* span_len = NULL;
    size_t max_span = 0;

    if (!spectral_size_mul(out_len, elem_size, &out_bytes)) {
        *t_synth = 0;
        return SPECTRAL_ERR_OVERFLOW;
    }

    /* Output tiling: each partition owns segments by contiguous INDEX range;
     * those segments write a contiguous output window [span_lo, span_lo+span_len).
     * Sizing per-thread buffers to the widest window (max_span) instead of the
     * full output replaces the O(threads*out_len) private-buffer footprint with
     * O(threads*max_span), and the combine pass adds each window back at its
     * offset.  Bit-identical: outside its window a partition contributes exactly
     * +0.0 (an exact additive identity), and the ascending-partition combine
     * order matches the legacy full reduce. */
    span_lo = spectral_calloc_array((size_t)n_parts, sizeof(size_t));
    span_len = spectral_calloc_array((size_t)n_parts, sizeof(size_t));
    if (!span_lo || !span_len) {
        free(span_lo);
        free(span_len);
        memset(out_buffer, 0, out_bytes);
        *t_synth = 0;
        return SPECTRAL_ERR_MEMORY;
    }

    /* Span pre-pass: same partitioning and same segment_loop_params_init as the
     * synth loop below, so the window bounds it derives exactly contain every
     * write that loop performs. */
    #pragma omp parallel for schedule(static) num_threads(n_parts)
    for (int p = 0; p < n_parts; p++) {
        size_t seg_start = ((size_t)p * sa.count) / (size_t)n_parts;
        size_t seg_end = ((size_t)(p + 1) * sa.count) / (size_t)n_parts;
        size_t lo = out_len;   /* sentinel: empty partition -> hi(0) <= lo(out_len) */
        size_t hi = 0;

        for (size_t i = seg_start; i < seg_end; i++) {
            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;   /* valid => length>=1, start_idx<out_len */
            size_t s = lp.start_idx;
            size_t e = lp.start_idx + lp.length;   /* clamped so e <= out_len */
            if (s < lo) lo = s;
            if (e > hi) hi = e;
        }

        if (hi > lo) {
            span_lo[p] = lo;
            span_len[p] = hi - lo;
        } else {
            span_lo[p] = 0;
            span_len[p] = 0;
        }
    }

    for (int p = 0; p < n_parts; p++) {
        if (span_len[p] > max_span) max_span = span_len[p];
    }

    if (max_span == 0) {
        /* Every partition is empty (all segments rejected): output is silence. */
        memset(out_buffer, 0, out_bytes);
        free(span_lo);
        free(span_len);
        *t_synth = omp_get_wtime() - synth_start;
        return SPECTRAL_OK;
    }

    tb = thread_buffers_alloc(n_parts, max_span, elem_size, &tb_err);
    if (!tb.bufs) {
        memset(out_buffer, 0, out_bytes);
        free(span_lo);
        free(span_len);
        *t_synth = 0;
        return tb_err != SPECTRAL_OK ? tb_err : SPECTRAL_ERR_MEMORY;
    }
    tb.span_lo = span_lo;
    tb.span_len = span_len;

    #pragma omp parallel for schedule(static) num_threads(n_parts)
    for (int p = 0; p < n_parts; p++) {
        size_t seg_start = ((size_t)p * sa.count) / (size_t)n_parts;
        size_t seg_end = ((size_t)(p + 1) * sa.count) / (size_t)n_parts;
        char* dst_base = (char*)tb.bufs[p];
        size_t lo = span_lo[p];

        for (size_t i = seg_start; i < seg_end; i++) {
            if (i + 4 < seg_end) SPECTRAL_PREFETCH_READ(&sa.segs[i + 4]);

            SegmentLoopParams lp = segment_loop_params_init(&sa.segs[i], &params, out_len);
            if (!lp.valid) continue;

            void* dst = dst_base + (lp.start_idx - lo) * elem_size;
            segment_fn(dst, &lp, fn_ctx);
        }
    }

    reduce_err = reduce_fn(&tb, out_buffer, out_len, out_bytes);
    if (reduce_err != SPECTRAL_OK) {
        memset(out_buffer, 0, out_bytes);
        thread_buffers_free(&tb);
        free(span_lo);
        free(span_len);
        *t_synth = 0;
        return reduce_err;
    }
    thread_buffers_free(&tb);
    free(span_lo);
    free(span_len);
    *t_synth = omp_get_wtime() - synth_start;
    return SPECTRAL_OK;
}

#ifndef SPECTRAL_USE_EMBEDDED_SYNTH
static SpectralError reduce_float_wrapper(const ThreadBuffers* tb, void* out, size_t len, size_t out_bytes) {
    return thread_buffers_combine_float(tb, (float*)out, len, out_bytes);
}
#endif

static SpectralError reduce_native_wrapper(const ThreadBuffers* tb, void* out, size_t len, size_t out_bytes) {
    return thread_buffers_combine_native(tb, (spectral_sample_t*)out, len, out_bytes);
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
        float p = spectral_segment_phase_at_cubic_f32(lp->phase, lp->alpha, lp->c2, lp->c3, (float)j);
        float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
        float amp = spectral_segment_amp_at_f32(lp->amp, lp->d_amp, (float)j) * fade_envelope(j, &fp, lp->length);
        spectral_sample_t sample = spectral_wavetable_lookup_f(wc->table, phase_norm);
        out[j] += amp * spectral_sample_to_float(sample);
    }
}
#endif

static void segment_fn_native_timbre(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const TimbreCtx* tc = (const TimbreCtx*)ctx;
    spectral_sample_t* out = (spectral_sample_t*)dst;
    FadeParams fp = fade_params_init(lp->length, SPECTRAL_SYNTH_CPU_FADE_SAMPLES);
    for (size_t j = 0; j < lp->length; j++) {
        float p = spectral_segment_phase_at_cubic_f32(lp->phase, lp->alpha, lp->c2, lp->c3, (float)j);
        float amp = spectral_segment_amp_at_f32(lp->amp, lp->d_amp, (float)j) * fade_envelope(j, &fp, lp->length);
        float sample_f = timbre_oscillator(p, amp, tc->timbre, lp->width);
        out[j] = spectral_sample_add(out[j], float_to_spectral_sample(sample_f));
    }
}

static void segment_fn_native_wavetable(void* dst, const SegmentLoopParams* lp, const void* ctx) {
    const NativeWavetableCtx* nwc = (const NativeWavetableCtx*)ctx;
    spectral_sample_t* out = (spectral_sample_t*)dst;
    FadeParams fp = fade_params_init(lp->length, SPECTRAL_SYNTH_CPU_FADE_SAMPLES);
    for (size_t j = 0; j < lp->length; j++) {
        float p = spectral_segment_phase_at_cubic_f32(lp->phase, lp->alpha, lp->c2, lp->c3, (float)j);
        float phase_norm = p * (float)SPECTRAL_INV_TWO_PI;
        float amp = spectral_segment_amp_at_f32(lp->amp, lp->d_amp, (float)j) * fade_envelope(j, &fp, lp->length);
        spectral_sample_t sample = spectral_wavetable_lookup_f(nwc->table, phase_norm);
        spectral_sample_t amp_native = float_to_spectral_sample(amp);
        spectral_sample_t scaled = spectral_sample_mul(sample, amp_native);
        out[j] = spectral_sample_add(out[j], scaled);
    }
}

/* Public synthesis functions */

/* Under SPECTRAL_USE_EMBEDDED_SYNTH the build selects the simulation TU's
 * synth_cpu definition instead; this float OpenMP body compiles itself out. */
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
