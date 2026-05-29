/* spectral_synth_arm32.c - Q15 Fixed-Point Synthesis for ARM Cortex-M
 * 
 * Optimized for ARM Cortex-M7 (Daisy Seed, STM32H7).
 * Uses DSP intrinsics when available for single-cycle saturating math.
 * 
 * Key Optimizations:
 *   - Dual MAC (SMLAD) processes 2 samples per instruction
 *   - Batch LUT lookups minimize cache misses
 *   - Phase accumulator is Q31 for precision, output is Q15
 *   - Active segment list avoids scanning inactive segments
 * 
 * Platform Detection:
 *   __ARM_ARCH_7EM__ - Cortex-M7 with DSP
 *   __ARM_FEATURE_DSP - DSP extension available
 *   ARM_MATH_CM7 - CMSIS-DSP library present
 * 
 * Memory Layout:
 *   Segment data in SDRAM (large, sequential access)
 *   Active state in SRAM (small, frequent access)
 *   LUT in DTCM or flash (read-only, cacheable)
 * 
 * Uses AoS layout for active segments. SoA is a future consideration
 * if profiling shows cache misses dominating at high voice counts.
 */

#include "spectral_synth_arm32.h"
#include "spectral_config.h"
#include "spectral_io.h"
#include "spectral_lut.h"
#include "spectral_perf_accounting.h"
#include "spectral_perf_model.h"
#include "spectral_utils.h"

#if SPECTRAL_EMBEDDED

#include <string.h>

/*
 * DMA prefetch from SDRAM to DTCM buffer.
 * User must provide dma_start_transfer() via HAL integration.
 */
#if SPECTRAL_HAS_DMA && SPECTRAL_ARM_M7
extern void dma_start_transfer(const void* src, void* dst, size_t bytes);
extern int  dma_transfer_complete(void);

#ifndef SPECTRAL_ARM32_DMA_BUFFER_DTCM
#define SPECTRAL_ARM32_DMA_BUFFER_DTCM 0
#endif
#ifndef SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE
#define SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE 0
#endif

#if SPECTRAL_ARM32_DMA_BUFFER_DTCM
#define SPECTRAL_ARM32_DMA_BUFFER_ATTR SPECTRAL_DTCM
#else
#define SPECTRAL_ARM32_DMA_BUFFER_ATTR
#endif

#if SPECTRAL_USE_CMSIS
#include "arm_math.h"
#endif

static SpectralSegmentQ15 dma_seg_buf[SPECTRAL_DMA_BATCH] SPECTRAL_ARM32_DMA_BUFFER_ATTR;
static uint32_t dma_prefetch_start = 0;
static uint32_t dma_prefetch_count = 0;
static int dma_prefetch_coherent = 0;

static void spectral_arm32_dma_rx_sync(const void* ptr, size_t bytes) {
#if SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE && SPECTRAL_USE_CMSIS && defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    const uintptr_t line = (uintptr_t)SPECTRAL_CACHE_LINE;
    if (ptr && bytes > 0u && line > 0u && (line & (line - 1u)) == 0u) {
        uintptr_t begin = (uintptr_t)ptr & ~(line - 1u);
        uintptr_t end = 0u;
        if ((uintptr_t)ptr <= UINTPTR_MAX - bytes &&
            (uintptr_t)ptr + bytes <= UINTPTR_MAX - (line - 1u)) {
            end = ((uintptr_t)ptr + bytes + (line - 1u)) & ~(line - 1u);
            if (end > begin && (end - begin) <= (uintptr_t)INT32_MAX) {
                SCB_InvalidateDCache_by_Addr((uint32_t*)begin, (int32_t)(end - begin));
            }
        }
    }
#else
    (void)ptr;
    (void)bytes;
#endif
#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
    __DSB();
#else
    __sync_synchronize();
#endif
}

static void spectral_arm32_dma_prefetch(SpectralArm32Ctx* ctx) {
    uint32_t next = ctx->next_seg_idx;
    uint32_t batch = ctx->num_segments - next;
    if (batch > SPECTRAL_DMA_BATCH) batch = SPECTRAL_DMA_BATCH;
    if (batch > 0) {
        dma_prefetch_start = next;
        dma_prefetch_count = batch;
        dma_prefetch_coherent = 0;
        dma_start_transfer(&ctx->segments[next], dma_seg_buf,
                           (size_t)batch * sizeof(SpectralSegmentQ15));
    }
}

/* Get segment pointer: use DMA buffer if segment was prefetched, else SDRAM.
 * DSB after DMA completion ensures the CPU sees coherent data even if the
 * DMA buffer is placed in cacheable SRAM rather than tightly-coupled DTCM. */
static inline const SpectralSegmentQ15* get_segment(
    const SpectralArm32Ctx* ctx, uint32_t idx)
{
    if (idx >= dma_prefetch_start &&
        idx < dma_prefetch_start + dma_prefetch_count) {
        if (!dma_prefetch_coherent && dma_transfer_complete()) {
            spectral_arm32_dma_rx_sync(dma_seg_buf,
                                       (size_t)dma_prefetch_count * sizeof(SpectralSegmentQ15));
            dma_prefetch_coherent = 1;
        }
        if (dma_prefetch_coherent) {
            return &dma_seg_buf[idx - dma_prefetch_start];
        }
    }
    return &ctx->segments[idx];
}
#else
/* No DMA: access segments directly from SDRAM */
static inline const SpectralSegmentQ15* get_segment(
    const SpectralArm32Ctx* ctx, uint32_t idx)
{
    return &ctx->segments[idx];
}
#endif /* SPECTRAL_HAS_DMA */

static inline int spectral_arm32_segment_chirp_supported(const SpectralSegmentQ15* seg) {
    if (!seg) return 0;
#if SPECTRAL_HAS_CHIRP
    /* The current ARM32 hot path stores df_q15 but does not yet consume it in
     * spectral_arm32_process().  Reject chirped embedded segments at load time
     * rather than silently rendering them as constant-frequency partials. */
    return seg->df_q15 == 0;
#else
    return 1;
#endif
}

#if SPECTRAL_USE_CMSIS
#include "arm_math.h"
#endif

/* Keep op/cycle accounting gating centralized. */
typedef struct SpectralPerfSegScanState {
    uint32_t start_cycles;
    uint32_t scan_start_idx;
    uint32_t activations;
} SpectralPerfSegScanState;

#if SPECTRAL_RESTRICTED_PROFILE
static SpectralPerfCounters s_perf_op_counts;

typedef struct {
    uint32_t total_cycles;
    uint32_t call_count;
    uint32_t min_cycles;
    uint32_t max_cycles;
} FunctionProfile;

static struct {
    FunctionProfile process;        /* spectral_arm32_process() */
    FunctionProfile segment_scan;   /* Active segment scanning */
    FunctionProfile oscillator;     /* LUT lookup + accumulation */
    FunctionProfile amplitude;      /* Amplitude scaling */
    uint32_t samples_processed;
    uint32_t segments_active_total;
} s_perf_stats = {0};

static uint32_t s_last_cycle_count = 0;
static uint32_t s_peak_cycle_count = 0;

static inline uint32_t get_cycles(void) {
#if defined(DWT) && defined(DWT_CYCCNT)
    return DWT->CYCCNT;
#else
    return 0;
#endif
}

static inline void profile_update(FunctionProfile* prof, uint32_t cycles) {
    prof->total_cycles += cycles;
    prof->call_count++;
    if (cycles < prof->min_cycles || prof->min_cycles == 0) prof->min_cycles = cycles;
    if (cycles > prof->max_cycles) prof->max_cycles = cycles;
}

static inline uint32_t perf_cache_miss_threshold(void) {
    const SpectralPerfModelProfile* p = spectral_perf_model_default_profile();
    return p ? p->cache_miss_threshold_active : 24u;
}

#endif

#if !SPECTRAL_RESTRICTED_PROFILE
static inline uint32_t get_cycles(void) {
    return 0;
}
#endif

static inline uint32_t spectral_perf_process_start(void) {
    return get_cycles();
}

static inline void spectral_perf_process_end(uint32_t start_cycles,
                                             uint32_t samples_processed,
                                             uint32_t active_segments) {
#if SPECTRAL_RESTRICTED_PROFILE
    uint32_t elapsed = get_cycles() - start_cycles;
    profile_update(&s_perf_stats.process, elapsed);
    spectral_perf_record_peak_block(&s_perf_op_counts, elapsed, active_segments);
    s_perf_stats.samples_processed += samples_processed;
#else
    (void)start_cycles;
    (void)samples_processed;
    (void)active_segments;
#endif
}

static inline void spectral_perf_track_active(uint32_t active_segments) {
#if SPECTRAL_RESTRICTED_PROFILE
    s_perf_stats.segments_active_total += active_segments;
#else
    (void)active_segments;
#endif
}

static inline SpectralPerfSegScanState spectral_perf_segment_scan_start(const SpectralArm32Ctx* ctx) {
    SpectralPerfSegScanState state = {0, 0, 0};
#if SPECTRAL_RESTRICTED_PROFILE
    state.start_cycles = get_cycles();
    state.scan_start_idx = ctx->next_seg_idx;
#else
    (void)ctx;
#endif
    return state;
}

static inline void spectral_perf_segment_activation(SpectralPerfSegScanState* state) {
#if SPECTRAL_RESTRICTED_PROFILE
    state->activations++;
#else
    (void)state;
#endif
}

static inline void spectral_perf_segment_scan_end(const SpectralArm32Ctx* ctx,
                                                  const SpectralPerfSegScanState* state) {
#if SPECTRAL_RESTRICTED_PROFILE
    profile_update(&s_perf_stats.segment_scan, get_cycles() - state->start_cycles);
    spectral_perf_count_segment_scan(&s_perf_op_counts, ctx->next_seg_idx - state->scan_start_idx);
    spectral_perf_count_segment_activations(&s_perf_op_counts, state->activations);
#else
    (void)ctx;
    (void)state;
#endif
}

static inline void spectral_perf_cache_pressure(uint32_t active_segments, uint32_t block_samples) {
#if SPECTRAL_RESTRICTED_PROFILE
    spectral_perf_count_cache_pressure(&s_perf_op_counts,
                                       active_segments,
                                       perf_cache_miss_threshold(),
                                       block_samples);
#else
    (void)active_segments;
    (void)block_samples;
#endif
}

static inline uint32_t spectral_perf_oscillator_start(void) {
    return get_cycles();
}

static inline void spectral_perf_oscillator_end(uint32_t start_cycles) {
#if SPECTRAL_RESTRICTED_PROFILE
    profile_update(&s_perf_stats.oscillator, get_cycles() - start_cycles);
#else
    (void)start_cycles;
#endif
}

static inline void spectral_perf_segment_samples(uint32_t segment_samples) {
#if SPECTRAL_RESTRICTED_PROFILE
    spectral_perf_count_segment_samples(&s_perf_op_counts, segment_samples);
#else
    (void)segment_samples;
#endif
}

static inline uint32_t spectral_perf_amplitude_start(void) {
    return get_cycles();
}

static inline void spectral_perf_amplitude_end(uint32_t start_cycles) {
#if SPECTRAL_RESTRICTED_PROFILE
    profile_update(&s_perf_stats.amplitude, get_cycles() - start_cycles);
#else
    (void)start_cycles;
#endif
}

/* Phase computation for 4-sample unrolling.
 * Backend parity note: this intentionally diverges from float
 * spectral_segment_phase_at_f32() by operating on fixed-point phase
 * accumulators for deterministic embedded timing and throughput. */
static inline void spectral_phase_batch4(q31_t phase,
                                         q31_t freq_inc,
                                         q31_t* p0,
                                         q31_t* p1,
                                         q31_t* p2,
                                         q31_t* p3) {
    const q31_t inc2 = freq_inc << 1;
    *p0 = phase;
    *p1 = phase + freq_inc;
    *p2 = phase + inc2;
    *p3 = phase + freq_inc + inc2;
}

/* Amplitude computation in Q15 domain.
 * Backend parity note: this intentionally diverges from float
 * spectral_segment_amp_at_f32() for fixed-point saturation behavior.
 *
 * All paths use saturating adds to prevent wrap-around when amp is
 * near Q15_MAX/Q15_MIN.  On ARM with DSP extensions spectral_qadd16()
 * compiles to a single QADD16 instruction — no extra cost. */
static inline void spectral_amp_batch4(q15_t amp,
                                       q15_t amp_delta,
                                       q15_t* a0,
                                       q15_t* a1,
                                       q15_t* a2,
                                       q15_t* a3) {
    *a0 = amp;
    *a1 = spectral_qadd16(amp, amp_delta);
    *a2 = spectral_qadd16(*a1, amp_delta);
    *a3 = spectral_qadd16(*a2, amp_delta);
}

static inline void spectral_accum_batch4(q31_t* accum,
                                         uint32_t j,
                                         const q15_t* samples,
                                         q15_t a0,
                                         q15_t a1,
                                         q15_t a2,
                                         q15_t a3) {
    accum[j] = spectral_mac_q15(accum[j], samples[0], a0);
    accum[j + 1] = spectral_mac_q15(accum[j + 1], samples[1], a1);
    accum[j + 2] = spectral_mac_q15(accum[j + 2], samples[2], a2);
    accum[j + 3] = spectral_mac_q15(accum[j + 3], samples[3], a3);
}

/* ARM M7 Prefetch & Cache */

#if SPECTRAL_ARM_M7 && defined(__GNUC__)

static SPECTRAL_MAYBE_UNUSED inline void prefetch_segment(const SpectralSegmentQ15* seg) {
    SPECTRAL_PREFETCH_READ(seg);
}

static inline void spectral_data_sync_barrier(void) {
#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
    __asm__ volatile ("dsb" ::: "memory");   /* real Cortex-M: full data sync barrier */
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST); /* host (incl. forced-M7 host-sim): ordering only */
#endif
}

#else
static SPECTRAL_MAYBE_UNUSED inline void prefetch_segment(const SpectralSegmentQ15* seg) {
    (void)seg;
}

static inline void spectral_data_sync_barrier(void) {
}
#endif

static inline uint32_t spectral_arm32_segment_end_sat_u32(uint32_t start, uint32_t length) {
    if (length > UINT32_MAX - start) return UINT32_MAX;
    return start + length;
}

static inline int spectral_arm32_segment_end_checked_u32(uint32_t start,
                                                          uint32_t length,
                                                          uint32_t* out_end) {
    if (!out_end || length == 0u || length > UINT32_MAX - start) return 0;
    *out_end = start + length;
    return 1;
}

static SpectralError spectral_arm32_validate_segment_data(const SpectralSegmentQ15* data,
                                                          uint32_t num_segments,
                                                          uint32_t output_len) {
    uint32_t first_live = 0u;
    uint32_t last_start = 0u;
    uint32_t last_end = 0u;

    if (num_segments == 0u) return SPECTRAL_OK;
    if (!data || output_len == 0u) return SPECTRAL_ERR_PARAM;

    for (uint32_t i = 0u; i < num_segments; i++) {
        uint32_t start = data[i].start;
        uint32_t end = 0u;

        if (!spectral_arm32_segment_end_checked_u32(start, data[i].length, &end)) {
            return SPECTRAL_ERR_OVERFLOW;
        }
        if (start >= output_len) {
            return SPECTRAL_ERR_PARAM;
        }
        if (i > 0u && (start < last_start || end < last_end)) {
            return SPECTRAL_ERR_PARAM;
        }
        if (!spectral_arm32_segment_chirp_supported(&data[i])) {
            return SPECTRAL_ERR_PARAM;
        }

        while (first_live < i) {
            uint32_t first_end = 0u;
            if (!spectral_arm32_segment_end_checked_u32(data[first_live].start,
                                                        data[first_live].length,
                                                        &first_end)) {
                return SPECTRAL_ERR_OVERFLOW;
            }
            if (first_end > start) break;
            first_live++;
        }
        if ((i - first_live + 1u) > (uint32_t)SPECTRAL_ARM32_MAX_ACTIVE) {
            return SPECTRAL_ERR_OVERFLOW;
        }

        last_start = start;
        last_end = end;
    }
    return SPECTRAL_OK;
}

static inline void spectral_arm32_zero_output(q15_t* out_left, q15_t* out_right, uint32_t num_samples) {
#if SPECTRAL_USE_CMSIS
    arm_fill_q15(0, out_left, num_samples);
    if (out_right) arm_fill_q15(0, out_right, num_samples);
#else
    if (out_left) memset(out_left, 0, (size_t)num_samples * sizeof(q15_t));
    if (out_right) memset(out_right, 0, (size_t)num_samples * sizeof(q15_t));
#endif
}

/* Initialization */

void spectral_arm32_init(SpectralArm32Ctx* ctx,
                            SpectralSegmentQ15* segments,
                            uint32_t capacity,
                            const q15_t* osc_lut,
                            uint32_t sample_rate) {
    if (!ctx) return;
    
    memset(ctx, 0, sizeof(SpectralArm32Ctx));
    ctx->segments = segments;
    ctx->segments_capacity = capacity;
    ctx->osc_lut = osc_lut;
    ctx->sample_rate = sample_rate;
    /* freq_q88 stores omega (rad/sample) * 256; phase is 0.32 turns (one cycle =
     * 2^32). The per-sample increment is (omega/2pi)*2^32 = freq_q88 * 2^24/(2pi),
     * INDEPENDENT of sample_rate. The previous 2^24/sample_rate divisor treated
     * freq_q88 as Hz and rendered ~sample_rate/(2pi) too low (the desktop float
     * backend and tests/arm_core both render the nominal frequency). */
    ctx->freq_inc_scale_q24 = (uint32_t)((double)(1u << 24) / SPECTRAL_TWO_PI + 0.5);
    ctx->amplitude_q15 = Q15_MAX;
    
    /* Ensure caches are coherent after init */
    spectral_data_sync_barrier();
}

void spectral_arm32_reset(SpectralArm32Ctx* ctx) {
    if (!ctx) return;
    ctx->output_position = 0;
    ctx->num_active = 0;
    ctx->next_seg_idx = 0;
    ctx->peak_active = 0;
}

void spectral_arm32_seek(SpectralArm32Ctx* ctx, uint32_t sample_pos) {
    if (!ctx) return;
    if (sample_pos > ctx->output_length) sample_pos = ctx->output_length;

    ctx->num_active = 0;
    ctx->output_position = sample_pos;
    ctx->next_seg_idx = 0;

    uint32_t lo = 0;
    uint32_t hi = ctx->num_segments;
    while (lo < hi) {
        uint32_t mid = lo + ((hi - lo) >> 1);
        const SpectralSegmentQ15* seg = &ctx->segments[mid];
        uint32_t seg_end = spectral_arm32_segment_end_sat_u32(seg->start, seg->length);
        if (seg_end <= sample_pos) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    ctx->next_seg_idx = lo;
}

/* Loading */

SpectralError spectral_arm32_load(SpectralArm32Ctx* ctx,
                                     const SpectralSegmentQ15* data,
                                     uint32_t num_segments,
                                     uint32_t output_len) {
    if (!ctx) return SPECTRAL_ERR_PARAM;
    if (num_segments > ctx->segments_capacity) return SPECTRAL_ERR_OVERFLOW;
    if (num_segments > 0u && (!data || !ctx->segments)) return SPECTRAL_ERR_PARAM;

    for (uint32_t i = 0; i < num_segments; i++) {
        if (!spectral_arm32_segment_chirp_supported(&data[i])) {
            return SPECTRAL_ERR_PARAM;
        }
    }

    if (num_segments > 0u) {
        memcpy((void*)ctx->segments, data, (size_t)num_segments * sizeof(SpectralSegmentQ15));
    }
    ctx->num_segments = num_segments;
    ctx->output_length = output_len;

    /* Ensure SDRAM writes are complete before synthesis */
    spectral_data_sync_barrier();

    spectral_arm32_reset(ctx);
    return SPECTRAL_OK;
}

/* Parameters */

void spectral_arm32_set_amplitude(SpectralArm32Ctx* ctx, float amplitude) {
    if (!ctx) return;
    amplitude = CLAMP(amplitude, 0.0f, 1.0f);
    ctx->amplitude_q15 = (q15_t)(amplitude * Q15_MAX);
}

void spectral_arm32_set_stretch(SpectralArm32Ctx* ctx, float stretch) {
    if (!ctx) return;
    (void)stretch;
}

/* Inner Synthesis Loop - ARM M7 Optimized */

#if SPECTRAL_ARM_M7

/* Core unrolled synthesis: accumulates samples from blk_start to blk_end.
 * No fade applied — used for sustain region and as building block for fade regions. */
SPECTRAL_ITCM
static inline void synth_core_m7(
    q31_t* restrict accum,
    const q15_t* restrict osc_lut,
    uint32_t blk_start,
    uint32_t blk_end,
    q31_t* restrict phase,
    q15_t* restrict amp,
    q31_t freq_inc,
    q15_t amp_delta
) {
    uint32_t j = blk_start;
    uint32_t len = blk_end - blk_start;
    uint32_t len4 = len & ~3U;
    uint32_t end4 = blk_start + len4;

    SPECTRAL_UNROLL_4
    for (; j < end4; j += 4) {
        SPECTRAL_PREFETCH_WRITE(&accum[j + 8]);

        q31_t p0, p1, p2, p3;
        spectral_phase_batch4(*phase, freq_inc, &p0, &p1, &p2, &p3);

        q15_t a0, a1, a2, a3;
        spectral_amp_batch4(*amp, amp_delta, &a0, &a1, &a2, &a3);

        q15_t s0, s1, s2, s3;
        s0 = spectral_lut_sin((uq16_t)(p0 >> 16), osc_lut);
        s1 = spectral_lut_sin((uq16_t)(p1 >> 16), osc_lut);
        s2 = spectral_lut_sin((uq16_t)(p2 >> 16), osc_lut);
        s3 = spectral_lut_sin((uq16_t)(p3 >> 16), osc_lut);

        accum[j]     = spectral_mac_q15(accum[j],     s0, a0);
        accum[j + 1] = spectral_mac_q15(accum[j + 1], s1, a1);
        accum[j + 2] = spectral_mac_q15(accum[j + 2], s2, a2);
        accum[j + 3] = spectral_mac_q15(accum[j + 3], s3, a3);

        *phase = p3 + freq_inc;
        *amp = spectral_qadd16(a3, amp_delta);
    }

    for (; j < blk_end; j++) {
        q15_t sample = spectral_lut_sin((uq16_t)(*phase >> 16), osc_lut);
        accum[j] = spectral_mac_q15(accum[j], sample, *amp);
        *phase += freq_inc;
        *amp = spectral_qadd16(*amp, amp_delta);
    }
}

/* Fade region synthesis: applies linear Q15 fade ramp to each sample.
 * fade_val/fade_step are Q15 ramp state (0->Q15_MAX for fade-in, Q15_MAX->0 for fade-out).
 * Backend parity note: envelope shape matches GPU/desktop fade semantics but uses
 * fixed-point arithmetic to preserve embedded determinism. */
SPECTRAL_ITCM
static inline void synth_fade_m7(
    q31_t* restrict accum,
    const q15_t* restrict osc_lut,
    uint32_t blk_start,
    uint32_t blk_end,
    q31_t* restrict phase,
    q15_t* restrict amp,
    q31_t freq_inc,
    q15_t amp_delta,
    q15_t fade_val,
    q15_t fade_step
) {
    for (uint32_t j = blk_start; j < blk_end; j++) {
        q15_t sample = spectral_lut_sin((uq16_t)(*phase >> 16), osc_lut);
        q15_t faded = spectral_mul_q15(sample, fade_val);
        accum[j] = spectral_mac_q15(accum[j], faded, *amp);
        *phase += freq_inc;
        *amp = spectral_qadd16(*amp, amp_delta);
        fade_val = spectral_qadd16(fade_val, fade_step);
    }
}

/* Full segment synthesis with linear fade envelope.
 * seg_offset: position within segment at blk_start
 * seg_length: total segment length in samples
 * Backend parity note: this path preserves the same three-region fade partition
 * semantics used in desktop/GPU backends, but executes fully in Q15/Q31. */
SPECTRAL_ITCM
static inline void synth_segment_m7(
    q31_t* restrict accum,
    const q15_t* restrict osc_lut,
    uint32_t blk_start,
    uint32_t blk_end,
    q31_t phase_start,
    q31_t freq_inc,
    q15_t amp_start,
    q15_t amp_delta,
    q31_t* phase_out,
    q15_t* amp_out,
    uint32_t seg_offset,
    uint32_t seg_length,
    uint32_t fade_len
) {
    q31_t phase = phase_start;
    q15_t amp = amp_start;

    uint32_t seg_fade_out_start = seg_length - fade_len;
    uint32_t blk_len = blk_end - blk_start;
    uint32_t seg_end_in_blk = seg_offset + blk_len;

    /* Compute 3 block-local boundaries: [blk_start, fi_end) [fi_end, fo_start) [fo_start, blk_end)
     * All offsets are relative to accumulator index space */
    uint32_t orig_blk_start = blk_start;

    /* Fade-in: segment positions [0, fade_len) mapped to block */
    uint32_t fi_end = blk_start;  /* no fade-in by default */
    if (seg_offset < fade_len) {
        fi_end = orig_blk_start + (fade_len - seg_offset);
        if (fi_end > blk_end) fi_end = blk_end;
    }

    /* Fade-out: segment positions [seg_fade_out_start, seg_length) mapped to block */
    uint32_t fo_start = blk_end;  /* no fade-out by default */
    if (seg_end_in_blk > seg_fade_out_start) {
        if (seg_offset < seg_fade_out_start) {
            fo_start = orig_blk_start + (seg_fade_out_start - seg_offset);
        } else {
            fo_start = orig_blk_start;  /* already in fade-out region */
        }
        if (fo_start < fi_end) fo_start = fi_end;
        if (fo_start > blk_end) fo_start = blk_end;
    }

    /* Fade-in region */
    if (fi_end > orig_blk_start) {
        q15_t fade_val = (q15_t)((int32_t)seg_offset * SPECTRAL_FADE_STEP_Q15);
        synth_fade_m7(accum, osc_lut, orig_blk_start, fi_end,
                      &phase, &amp, freq_inc, amp_delta,
                      fade_val, SPECTRAL_FADE_STEP_Q15);
    }

    /* Sustain region: no fade, full unrolled path */
    if (fi_end < fo_start) {
        synth_core_m7(accum, osc_lut, fi_end, fo_start,
                      &phase, &amp, freq_inc, amp_delta);
    }

    /* Fade-out region */
    if (fo_start < blk_end) {
        uint32_t fo_seg_pos = seg_offset + (fo_start - orig_blk_start);
        uint32_t samples_into_fade = fo_seg_pos - seg_fade_out_start;
        q15_t fade_val = Q15_MAX - (q15_t)((int32_t)samples_into_fade * SPECTRAL_FADE_STEP_Q15);
        synth_fade_m7(accum, osc_lut, fo_start, blk_end,
                      &phase, &amp, freq_inc, amp_delta,
                      fade_val, -SPECTRAL_FADE_STEP_Q15);
    }

    *phase_out = phase;
    *amp_out = amp;
}

#endif /* SPECTRAL_ARM_M7 */

SPECTRAL_ITCM
uint32_t spectral_arm32_process(SpectralArm32Ctx* ctx,
                                   q15_t* out_left,
                                   q15_t* out_right,
                                   uint32_t num_samples) {
    uint32_t perf_start = spectral_perf_process_start();
    uint32_t out_pos = 0;
    uint32_t out_end = 0;
    uint32_t remaining = 0;

    if (SPECTRAL_UNLIKELY(!ctx || !out_left || num_samples == 0u)) {
        spectral_perf_process_end(perf_start, 0, 0);
        return 0;
    }

    if (SPECTRAL_UNLIKELY(num_samples > 256u)) {
        SPECTRAL_DBG("arm32: block size %u truncated to 256", (unsigned)num_samples);
        num_samples = 256u;
    }

    if (SPECTRAL_UNLIKELY(ctx->output_position >= ctx->output_length ||
                          ctx->num_segments == 0u || !ctx->segments || !ctx->osc_lut)) {
        spectral_arm32_zero_output(out_left, out_right, num_samples);
        spectral_perf_process_end(perf_start, 0, 0);
        return 0;
    }

    remaining = ctx->output_length - ctx->output_position;
    if (num_samples > remaining) num_samples = remaining;
    if (SPECTRAL_UNLIKELY(num_samples == 0u)) {
        spectral_perf_process_end(perf_start, 0, 0);
        return 0;
    }

    out_pos = ctx->output_position;
    out_end = out_pos + num_samples;
    const q15_t* restrict osc_lut = ctx->osc_lut;
    const q15_t master_amp = ctx->amplitude_q15;

    /* Static accumulator in DTCM for zero wait-state access on Cortex-M7.
     * Safe for embedded: single-threaded audio callback, no reentrancy. */
#if defined(__GNUC__) || defined(__clang__)
    static q31_t accum[256] __attribute__((aligned(SPECTRAL_CACHE_LINE))) SPECTRAL_DTCM;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    static _Alignas(32) q31_t accum[256];
#else
    static q31_t accum[256];
#endif
    
#if SPECTRAL_USE_CMSIS
    arm_fill_q31(0, accum, num_samples);
#else
    memset(accum, 0, num_samples * sizeof(q31_t));
#endif
    
    /* Prefetch first few segments from SDRAM */
#if SPECTRAL_HAS_DMA && SPECTRAL_ARM_M7
    spectral_arm32_dma_prefetch(ctx);
#elif SPECTRAL_ARM_M7
    if (ctx->next_seg_idx < ctx->num_segments) {
        prefetch_segment(&ctx->segments[ctx->next_seg_idx]);
        if (ctx->next_seg_idx + 1 < ctx->num_segments)
            prefetch_segment(&ctx->segments[ctx->next_seg_idx + 1]);
    }
#endif
    
    /* Activate new segments that start within this block */
    SpectralPerfSegScanState seg_scan_state = spectral_perf_segment_scan_start(ctx);
    while (ctx->next_seg_idx < ctx->num_segments &&
           ctx->num_active < SPECTRAL_ARM32_MAX_ACTIVE) {
        const SpectralSegmentQ15* seg = get_segment(ctx, ctx->next_seg_idx);
        uint32_t seg_start = seg->start;
        uint32_t seg_length = seg->length;
        uint32_t seg_end = spectral_arm32_segment_end_sat_u32(seg_start, seg_length);
        if (seg_start >= out_end) break;

        /* Prefetch next segment while processing this one */
#if SPECTRAL_ARM_M7
        if (ctx->next_seg_idx + 2 < ctx->num_segments) {
            prefetch_segment(&ctx->segments[ctx->next_seg_idx + 2]);
        }
#endif

        if (seg_end > out_pos) {
            uint16_t slot = ctx->num_active;

            /* Calculate how many samples into the segment we are */
            uint32_t sample_offset = (out_pos > seg_start) ? (out_pos - seg_start) : 0;
            uint32_t fade_len = SPECTRAL_FADE_SAMPLES_EMBEDDED;
            if (fade_len > seg_length / 2) fade_len = seg_length / 2;
            if (fade_len == 0) fade_len = 1;

            /* Backend parity note:
             * This is the fixed-point counterpart of spectral_segment_alpha_f32().
             * Q8.8 frequency is mapped to Q31 phase increment for embedded synthesis. */
            q31_t freq_inc = (q31_t)((uint32_t)seg->freq_q88 * ctx->freq_inc_scale_q24);
            q31_t phase_acc = ((q31_t)seg->phase_q15 + 32768) << 16;
            q15_t amp_cur;

            if (sample_offset > 0) {
                phase_acc += sample_offset * freq_inc;
                q31_t amp_advance = (q31_t)seg->da_q15 * (q31_t)sample_offset;
                amp_cur = spectral_ssat16((q31_t)seg->amp_q15 + amp_advance);
            } else {
                amp_cur = seg->amp_q15;
            }

#if SPECTRAL_SOA_ACTIVE
            ctx->active_soa.seg_idx[slot] = ctx->next_seg_idx;
            ctx->active_soa.freq_inc[slot] = freq_inc;
            ctx->active_soa.phase_acc[slot] = phase_acc;
            ctx->active_soa.amp_current[slot] = amp_cur;
            ctx->active_soa.amp_delta[slot] = seg->da_q15;
            ctx->active_soa.seg_start[slot] = seg_start;
            ctx->active_soa.seg_end[slot] = seg_end;
            ctx->active_soa.seg_length[slot] = (uint16_t)seg_length;
            ctx->active_soa.fade_len[slot] = (uint16_t)fade_len;
#else
            SpectralActiveSegment* act = &ctx->active[slot];
            act->seg_idx = ctx->next_seg_idx;
            act->freq_inc = freq_inc;
            act->phase_acc = phase_acc;
            act->amp_current = amp_cur;
            act->amp_delta = seg->da_q15;
            act->seg_start = seg_start;
            act->seg_end = seg_end;
            act->seg_length = (uint16_t)seg_length;
            act->fade_len = (uint16_t)fade_len;
#endif
            ctx->num_active++;
            spectral_perf_segment_activation(&seg_scan_state);
        }
        ctx->next_seg_idx++;
    }
    spectral_perf_segment_scan_end(ctx, &seg_scan_state);
    
    /* Track peak polyphony */
    if (SPECTRAL_UNLIKELY(ctx->num_active > ctx->peak_active)) {
        ctx->peak_active = ctx->num_active;
    }
    spectral_perf_cache_pressure(ctx->num_active, num_samples);
    
    /* Process all active segments */
    uint32_t osc_start = spectral_perf_oscillator_start();
    uint16_t i = 0;
    while (i < ctx->num_active) {
#if SPECTRAL_SOA_ACTIVE
        uint32_t seg_start = ctx->active_soa.seg_start[i];
        uint32_t seg_end = ctx->active_soa.seg_end[i];
        uint32_t seg_length = ctx->active_soa.seg_length[i];
        uint32_t fade_len = ctx->active_soa.fade_len[i];
#else
        SpectralActiveSegment* act = &ctx->active[i];
        uint32_t seg_start = act->seg_start;
        uint32_t seg_end = act->seg_end;
        uint32_t seg_length = act->seg_length;
        uint32_t fade_len = act->fade_len;
#endif

        /* Remove expired segments */
        if (SPECTRAL_UNLIKELY(out_pos >= seg_end)) {
#if SPECTRAL_SOA_ACTIVE
            /* SoA removal: copy last element's fields into slot i */
            uint16_t last = --ctx->num_active;
            ctx->active_soa.phase_acc[i]   = ctx->active_soa.phase_acc[last];
            ctx->active_soa.freq_inc[i]    = ctx->active_soa.freq_inc[last];
            ctx->active_soa.amp_current[i] = ctx->active_soa.amp_current[last];
            ctx->active_soa.amp_delta[i]   = ctx->active_soa.amp_delta[last];
            ctx->active_soa.seg_start[i]   = ctx->active_soa.seg_start[last];
            ctx->active_soa.seg_end[i]     = ctx->active_soa.seg_end[last];
            ctx->active_soa.seg_length[i]  = ctx->active_soa.seg_length[last];
            ctx->active_soa.fade_len[i]    = ctx->active_soa.fade_len[last];
            ctx->active_soa.seg_idx[i]     = ctx->active_soa.seg_idx[last];
#if SPECTRAL_HAS_CHIRP
            ctx->active_soa.freq_delta[i]  = ctx->active_soa.freq_delta[last];
#endif
#else
            *act = ctx->active[--ctx->num_active];
#endif
            continue;
        }

        /* Compute block range */
        uint32_t blk_start = (seg_start > out_pos) ? (seg_start - out_pos) : 0;
        uint32_t blk_end = (seg_end < out_end) ? (seg_end - out_pos) : num_samples;
        uint32_t len = blk_end - blk_start;
        spectral_perf_segment_samples(len);

        /* Read current state */
#if SPECTRAL_SOA_ACTIVE
        q31_t phase = ctx->active_soa.phase_acc[i];
        q31_t freq_inc = ctx->active_soa.freq_inc[i];
        q15_t amp = ctx->active_soa.amp_current[i];
        q15_t d_amp = ctx->active_soa.amp_delta[i];
#else
        q31_t phase = act->phase_acc;
        q31_t freq_inc = act->freq_inc;
        q15_t amp = act->amp_current;
        q15_t d_amp = act->amp_delta;
#endif

#if SPECTRAL_ARM_M7
        /* Use ARM M7 optimized inner loop */
        {
            uint32_t seg_offset = (uint32_t)(out_pos + blk_start) - seg_start;
            synth_segment_m7(accum, osc_lut, blk_start, blk_end,
                             phase, freq_inc, amp, d_amp,
                             &phase, &amp, seg_offset, seg_length, fade_len);
        }
#else
        /* Generic inner loop with linear fade envelope */
        {
            uint32_t seg_offset = (uint32_t)(out_pos + blk_start) - seg_start;
            uint32_t seg_len = seg_length;

            uint32_t seg_fo_start = seg_len - fade_len;
            uint32_t blk_len = len;

            /* Map fade regions to block offsets */
            uint32_t fi_end = blk_start;
            if (seg_offset < fade_len) {
                fi_end = blk_start + (fade_len - seg_offset);
                if (fi_end > blk_end) fi_end = blk_end;
            }
            uint32_t fo_start = blk_end;
            if (seg_offset + blk_len > seg_fo_start) {
                fo_start = (seg_fo_start > seg_offset)
                    ? blk_start + (seg_fo_start - seg_offset) : blk_start;
                if (fo_start < fi_end) fo_start = fi_end;
            }

            /* Fade-in */
            q15_t fade_val = (q15_t)((int32_t)seg_offset * SPECTRAL_FADE_STEP_Q15);
            for (uint32_t j = blk_start; j < fi_end; j++) {
                q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                sample = spectral_mul_q15(sample, fade_val);
                accum[j] = spectral_mac_q15(accum[j], sample, amp);
                phase += freq_inc;
                amp = spectral_qadd16(amp, d_amp);
                fade_val = spectral_qadd16(fade_val, SPECTRAL_FADE_STEP_Q15);
            }

            /* Sustain - 4 samples at a time */
            uint32_t j = fi_end;
            uint32_t sustain_len = fo_start - fi_end;
            uint32_t sustain_len4 = sustain_len & ~3U;
            uint32_t end4 = fi_end + sustain_len4;

            SPECTRAL_UNROLL_4
            for (; j < end4; j += 4) {
                q31_t p0, p1, p2, p3;
                spectral_phase_batch4(phase, freq_inc, &p0, &p1, &p2, &p3);

                q15_t a0, a1, a2, a3;
                spectral_amp_batch4(amp, d_amp, &a0, &a1, &a2, &a3);

                q15_t samples[4];
                samples[0] = spectral_lut_sin((uq16_t)(p0 >> 16), osc_lut);
                samples[1] = spectral_lut_sin((uq16_t)(p1 >> 16), osc_lut);
                samples[2] = spectral_lut_sin((uq16_t)(p2 >> 16), osc_lut);
                samples[3] = spectral_lut_sin((uq16_t)(p3 >> 16), osc_lut);

                spectral_accum_batch4(accum, j, samples, a0, a1, a2, a3);

                phase = p3 + freq_inc;
                amp = spectral_qadd16(a3, d_amp);
            }
            for (; j < fo_start; j++) {
                q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                accum[j] = spectral_mac_q15(accum[j], sample, amp);
                phase += freq_inc;
                amp = spectral_qadd16(amp, d_amp);
            }

            /* Fade-out */
            if (fo_start < blk_end) {
                uint32_t fo_seg_pos = seg_offset + (fo_start - blk_start);
                uint32_t into_fade = fo_seg_pos - seg_fo_start;
                fade_val = Q15_MAX - (q15_t)((int32_t)into_fade * SPECTRAL_FADE_STEP_Q15);
                for (; j < blk_end; j++) {
                    q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                    sample = spectral_mul_q15(sample, fade_val);
                    accum[j] = spectral_mac_q15(accum[j], sample, amp);
                    phase += freq_inc;
                    amp = spectral_qadd16(amp, d_amp);
                    fade_val = spectral_qadd16(fade_val, -SPECTRAL_FADE_STEP_Q15);
                }
            }
        }
#endif

        /* Write back state */
#if SPECTRAL_SOA_ACTIVE
        ctx->active_soa.phase_acc[i] = phase;
        ctx->active_soa.amp_current[i] = amp;
#else
        act->phase_acc = phase;
        act->amp_current = amp;
#endif

        i++;
    }
    spectral_perf_oscillator_end(osc_start);
    spectral_perf_track_active(ctx->num_active);
    
    /* Convert the Q30 accumulator (sum of Q15*Q15 MAC products) to Q15 with the
     * master gain. Q30 -> Q15 is a >>15 shift (see spectral_q30_to_q15_scaled). */
    uint32_t amp_start = spectral_perf_amplitude_start();
    spectral_q30_to_q15_scaled(accum, out_left, num_samples, master_amp);
    
    if (out_right && out_right != out_left) {
#if SPECTRAL_USE_CMSIS
        arm_copy_q15(out_left, out_right, num_samples);
#else
        memcpy(out_right, out_left, num_samples * sizeof(q15_t));
#endif
    }
    spectral_perf_amplitude_end(amp_start);
    
    ctx->output_position = out_end;
    spectral_perf_process_end(perf_start, num_samples, ctx->num_active);
    return num_samples;
}

/* Interleaved Stereo Output */

uint32_t spectral_arm32_process_interleaved(SpectralArm32Ctx* ctx,
                                               q15_t* out_interleaved,
                                               uint32_t num_samples) {
    if (!ctx || !out_interleaved) return 0;
    
    /* Cache-aligned temp buffer for efficient SIMD copy to interleaved output */
#if defined(__GNUC__) || defined(__clang__)
    q15_t temp[256] __attribute__((aligned(SPECTRAL_CACHE_LINE)));
#else
    q15_t temp[256];
#endif
    if (num_samples > 256) num_samples = 256;
    
    uint32_t written = spectral_arm32_process(ctx, temp, NULL, num_samples);
    
    /* Use shared stereo interleaving function */
    spectral_mono_to_stereo_q15(temp, out_interleaved, written);
    
    return written;
}

/* Restricted Mode Profiling */

/* Cycle-count profiling for optimization tuning */
#if SPECTRAL_RESTRICTED_PROFILE

void arm32_synth_profile_start(void) {
    s_last_cycle_count = get_cycles();
}

uint32_t arm32_synth_profile_end(void) {
    uint32_t elapsed = get_cycles() - s_last_cycle_count;
    if (elapsed > s_peak_cycle_count) s_peak_cycle_count = elapsed;
    return elapsed;
}

uint32_t arm32_synth_get_peak_cycles(void) {
    return s_peak_cycle_count;
}

void arm32_synth_reset_profile(void) {
    s_peak_cycle_count = 0;
    memset(&s_perf_stats, 0, sizeof(s_perf_stats));
    spectral_perf_counters_reset(&s_perf_op_counts);
}

/* TODO: DMA transfer whole struct or pack for UART/SWO */
uint32_t arm32_synth_get_total_cycles(void) {
    return s_perf_stats.process.total_cycles;
}

uint32_t arm32_synth_get_call_count(void) {
    return s_perf_stats.process.call_count;
}

uint32_t arm32_synth_get_avg_cycles(void) {
    if (s_perf_stats.process.call_count == 0) return 0;
    return s_perf_stats.process.total_cycles / s_perf_stats.process.call_count;
}

uint32_t arm32_synth_get_min_cycles(void) {
    return s_perf_stats.process.min_cycles;
}

uint32_t arm32_synth_get_max_cycles(void) {
    return s_perf_stats.process.max_cycles;
}

uint32_t arm32_synth_get_samples_processed(void) {
    return s_perf_stats.samples_processed;
}

void arm32_synth_get_op_counts(SpectralPerfCounters* out) {
    if (!out) return;
    *out = s_perf_op_counts;
}
#endif /* SPECTRAL_RESTRICTED_PROFILE */

#endif /* SPECTRAL_EMBEDDED */
