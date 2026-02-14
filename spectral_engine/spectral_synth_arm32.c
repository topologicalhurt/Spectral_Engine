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

#if SPECTRAL_EMBEDDED

#include <string.h>

/* SPECTRAL_DBG fallback — full definition is in spectral_utils.h (desktop) */
#ifndef SPECTRAL_DBG
#define SPECTRAL_DBG(...)       ((void)0)
#endif

#if defined(SPECTRAL_RESTRICTED_MODE) && SPECTRAL_ARM_M7
#define SPECTRAL_HAS_SDRAM      1
#define SPECTRAL_PREFETCH_DIST  4
#endif

/*
 * DMA prefetch from SDRAM to DTCM buffer.
 * User must provide dma_start_transfer() via HAL integration.
 */
#if SPECTRAL_HAS_DMA && SPECTRAL_HAS_SDRAM
extern void dma_start_transfer(const void* src, void* dst, size_t bytes);
extern int  dma_transfer_complete(void);

static SpectralSegmentQ15 dma_seg_buf[SPECTRAL_DMA_BATCH] SPECTRAL_DTCM;
static uint32_t dma_prefetch_start = 0;
static uint32_t dma_prefetch_count = 0;

static void spectral_arm32_dma_prefetch(SpectralArm32Ctx* ctx) {
    uint32_t next = ctx->next_seg_idx;
    uint32_t batch = ctx->num_segments - next;
    if (batch > SPECTRAL_DMA_BATCH) batch = SPECTRAL_DMA_BATCH;
    if (batch > 0) {
        dma_prefetch_start = next;
        dma_prefetch_count = batch;
        dma_start_transfer(&ctx->segments[next], dma_seg_buf,
                           batch * sizeof(SpectralSegmentQ15));
    }
}

/* Get segment pointer: use DMA buffer if segment was prefetched, else SDRAM */
static inline const SpectralSegmentQ15* get_segment(
    const SpectralArm32Ctx* ctx, uint32_t idx)
{
    if (idx >= dma_prefetch_start &&
        idx < dma_prefetch_start + dma_prefetch_count &&
        dma_transfer_complete()) {
        return &dma_seg_buf[idx - dma_prefetch_start];
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

#if defined(ARM_MATH_CM7) || defined(ARM_MATH_CM4)
#define SPECTRAL_USE_CMSIS      1
#include "arm_math.h"

#define CMSIS_FILL_Q15(val, buf, len)     arm_fill_q15((val), (buf), (len))
#define CMSIS_FILL_Q31(buf, val, len)     arm_fill_q31((val), (buf), (len))
#define CMSIS_Q31_TO_Q15(src, dst, len)   arm_q31_to_q15((src), (dst), (len))
#define CMSIS_SCALE_Q15(src, s, sh, dst, len)  arm_scale_q15((src), (s), (sh), (dst), (len))
#define CMSIS_COPY_Q15(src, dst, len)     arm_copy_q15((src), (dst), (len))
#define CMSIS_ABSMAX_Q15(src, len, max, idx)  arm_absmax_q15((src), (len), (max), (idx))
#define CMSIS_SHIFT_Q15(src, sh, dst, len)    arm_shift_q15((src), (sh), (dst), (len))
#else
#define SPECTRAL_USE_CMSIS      0
#define CMSIS_FILL_Q15(val, buf, len)     memset((buf), 0, (len) * sizeof(q15_t))
#define CMSIS_FILL_Q31(buf, val, len)     do { for(uint32_t _i=0; _i<(len); _i++) (buf)[_i]=(val); } while(0)
#define CMSIS_Q31_TO_Q15(src, dst, len)   /* fallback below */
#define CMSIS_SCALE_Q15(src, s, sh, dst, len) /* fallback below */
#define CMSIS_COPY_Q15(src, dst, len)     memcpy((dst), (src), (len) * sizeof(q15_t))
#define CMSIS_ABSMAX_Q15(src, len, max, idx)  /* fallback below */
#define CMSIS_SHIFT_Q15(src, sh, dst, len)    /* fallback below */
#endif

/* Performance tracking stubs (real impl at bottom of file when enabled) */
#if !(defined(SPECTRAL_RESTRICTED_MODE) && defined(SPECTRAL_DEBUG_RESTRICTED) && SPECTRAL_DEBUG_RESTRICTED)
#define get_cycles()            0
#define profile_update(p, c)    ((void)0)
#define PERF_PROCESS_START()
#define PERF_PROCESS_END(n)
#define PERF_TRACK_ACTIVE(n)
#endif

/* Alias centralized macros for conciseness */
#define UNROLL_HINT         SPECTRAL_UNROLL_4
#define UNROLL_2            SPECTRAL_UNROLL_2
#define LIKELY(x)           SPECTRAL_LIKELY(x)
#define UNLIKELY(x)         SPECTRAL_UNLIKELY(x)
#define RESTRICT            __restrict__

/* Optimization level: 0=safe, 1=balanced (default), 2=aggressive, 3=reserved */
#ifndef SPECTRAL_OPT_LEVEL
#define SPECTRAL_OPT_LEVEL 1
#endif

#define SPECTRAL_UNSAFE_MATH    (SPECTRAL_OPT_LEVEL >= 1)
#define SPECTRAL_NO_LUT_INTERP  (SPECTRAL_OPT_LEVEL >= 2)

/* Phase computation for 4-sample unrolling.
 * Independent outputs allow dual-issue on ARM M7. */
#define SPECTRAL_PHASES_4(phase, freq_inc, p0, p1, p2, p3) do { \
    const q31_t _base = (phase); \
    const q31_t _inc = (freq_inc); \
    (p0) = _base; \
    (p1) = _base + _inc; \
    (p2) = _base + (_inc << 1); \
    (p3) = _base + _inc + (_inc << 1); \
} while (0)

/* Amplitude computation. OPT_LEVEL>=1 uses non-saturating parallel math. */
#if SPECTRAL_UNSAFE_MATH
#define SPECTRAL_AMPS_4(amp, amp_delta, a0, a1, a2, a3) do { \
    const q15_t _a = (amp); \
    const q15_t _d = (amp_delta); \
    (a0) = _a; \
    (a1) = (q15_t)(_a + _d); \
    (a2) = (q15_t)(_a + (_d << 1)); \
    (a3) = (q15_t)(_a + _d + (_d << 1)); \
} while (0)
#else
#define SPECTRAL_AMPS_4(amp, amp_delta, a0, a1, a2, a3) do { \
    (a0) = (amp); \
    (a1) = spectral_qadd16((amp), (amp_delta)); \
    (a2) = spectral_qadd16((a1), (amp_delta)); \
    (a3) = spectral_qadd16((a2), (amp_delta)); \
} while (0)
#endif

#define SPECTRAL_ACCUM_4(accum, j, samples, a0, a1, a2, a3) do { \
    (accum)[(j)]     = spectral_mac_q15((accum)[(j)],     (samples)[0], (a0)); \
    (accum)[(j) + 1] = spectral_mac_q15((accum)[(j) + 1], (samples)[1], (a1)); \
    (accum)[(j) + 2] = spectral_mac_q15((accum)[(j) + 2], (samples)[2], (a2)); \
    (accum)[(j) + 3] = spectral_mac_q15((accum)[(j) + 3], (samples)[3], (a3)); \
} while (0)

/* ARM M7 Prefetch & Cache */

#if SPECTRAL_ARM_M7 && defined(__GNUC__)

#define PREFETCH_READ(addr)     __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr)    __builtin_prefetch((addr), 1, 3)

static inline void prefetch_segment(const SpectralSegmentQ15* seg) {
    PREFETCH_READ(seg);
}

/* Prefetch active segment state */
static inline void prefetch_active(const SpectralActiveSegment* act) {
    PREFETCH_READ(act);
}

/* Memory barrier for cache coherency */
#define DMB()                   __asm__ volatile ("dmb" ::: "memory")
#define DSB()                   __asm__ volatile ("dsb" ::: "memory")
#define ISB()                   __asm__ volatile ("isb" ::: "memory")

#else
/* Use definitions from spectral_common.h if available */
#ifndef PREFETCH_READ
#define PREFETCH_READ(addr)     ((void)(addr))
#endif
#ifndef PREFETCH_WRITE
#define PREFETCH_WRITE(addr)    ((void)(addr))
#endif
#define prefetch_segment(seg)   ((void)(seg))
#define prefetch_active(act)    ((void)(act))
#define DMB()
#define DSB()
#define ISB()
#endif

/* ARM M7 Dual MAC */

#if SPECTRAL_ARM_M7 && defined(__ARM_FEATURE_DSP)

/* Pack two Q15 values for dual MAC */
static inline q31_t pack_q15(q15_t lo, q15_t hi) {
    return ((q31_t)hi << 16) | ((q31_t)lo & 0xFFFF);
}

/* Unpack Q31 to two Q15 values */
static inline void unpack_q15(q31_t packed, q15_t* lo, q15_t* hi) {
    *lo = (q15_t)(packed & 0xFFFF);
    *hi = (q15_t)(packed >> 16);
}

/* Dual signed MAC: acc += (a0*b0) + (a1*b1)
 * Uses SMLAD instruction on ARM */
static inline q31_t dual_mac_q15(q31_t acc, q31_t a_packed, q31_t b_packed) {
#if defined(__GNUC__) && defined(__arm__)
    q31_t result;
    __asm__ volatile (
        "smlad %0, %2, %3, %1"
        : "=r" (result)
        : "r" (acc), "r" (a_packed), "r" (b_packed)
    );
    return result;
#else
    /* Fallback - extract and MAC separately */
    q15_t a0 = (q15_t)(a_packed & 0xFFFF);
    q15_t a1 = (q15_t)(a_packed >> 16);
    q15_t b0 = (q15_t)(b_packed & 0xFFFF);
    q15_t b1 = (q15_t)(b_packed >> 16);
    return acc + ((q31_t)a0 * b0) + ((q31_t)a1 * b1);
#endif
}

#define SPECTRAL_HAS_DUAL_MAC   1

#else
#define SPECTRAL_HAS_DUAL_MAC   0
#endif

/* Optimized LUT Lookup */

#if SPECTRAL_ARM_M7

#define OSC_FRAC_BITS   (16 - SPECTRAL_OSC_LUT_BITS)
#define OSC_FRAC_MASK   ((1u << OSC_FRAC_BITS) - 1)

static inline void osc_lut_lookup_2(
    const q15_t* RESTRICT lut,
    uq16_t idx0, uq16_t idx1,
    q15_t* RESTRICT out0, q15_t* RESTRICT out1
) {
    /* Extract table indices (upper 10 bits for 1024-entry LUT) */
    uint32_t t0 = idx0 >> OSC_FRAC_BITS;
    uint32_t t1 = idx1 >> OSC_FRAC_BITS;
    
    /* Always prefetch — redundant prefetches are free (ignored by HW) */
    PREFETCH_READ(&lut[t1]);
    
    /* Extract fractional part and scale to 0-255 for 8-bit interpolation */
    uint32_t f0 = ((idx0 & OSC_FRAC_MASK) * 256) >> OSC_FRAC_BITS;
    uint32_t f1 = ((idx1 & OSC_FRAC_MASK) * 256) >> OSC_FRAC_BITS;
    
    q15_t s0_a = lut[t0];
    q15_t s0_b = lut[t0 + 1];
    q15_t s1_a = lut[t1];
    q15_t s1_b = lut[t1 + 1];
    
    /* Interpolate: out = a + (b-a)*f/256 */
    *out0 = s0_a + (q15_t)(((q31_t)(s0_b - s0_a) * (int32_t)f0) >> 8);
    *out1 = s1_a + (q15_t)(((q31_t)(s1_b - s1_a) * (int32_t)f1) >> 8);
}

/* LUT interpolation: OPT_LEVEL>=2 uses nearest-neighbor (faster, ~0.1% THD) */
#if SPECTRAL_NO_LUT_INTERP
#define OSC_LOOKUP(lut, idx, out_ptr) do { \
    *(out_ptr) = (lut)[(idx) >> OSC_FRAC_BITS]; \
} while (0)
#else
#define OSC_LOOKUP(lut, idx, out_ptr) do { \
    uint32_t _t = (idx) >> OSC_FRAC_BITS; \
    uint32_t _f = (((idx) & OSC_FRAC_MASK) * 256) >> OSC_FRAC_BITS; \
    q15_t _a = (lut)[_t]; \
    q15_t _b = (lut)[_t + 1]; \
    *(out_ptr) = _a + (q15_t)(((q31_t)(_b - _a) * (int32_t)_f) >> 8); \
} while (0)
#endif

static inline void osc_lut_lookup_4(
    const q15_t* RESTRICT lut,
    const uq16_t* RESTRICT idx,
    q15_t* RESTRICT out
) {
    OSC_LOOKUP(lut, idx[0], &out[0]);
    OSC_LOOKUP(lut, idx[1], &out[1]);
    OSC_LOOKUP(lut, idx[2], &out[2]);
    OSC_LOOKUP(lut, idx[3], &out[3]);
}

#endif /* SPECTRAL_ARM_M7 */

/* Linear fade length for embedded paths (shorter than desktop's 64-sample Hann) */
#define SPECTRAL_FADE_EMBEDDED  32
#define FADE_STEP_Q15           (Q15_ONE / SPECTRAL_FADE_EMBEDDED)

#define Q214_UNITY  16384

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
    ctx->freq_inc_scale_q24 = (sample_rate > 0)
        ? (uint32_t)(((1u << 24) + (sample_rate / 2u)) / sample_rate)
        : 0;
    ctx->amplitude_q15 = Q15_ONE;
    ctx->stretch_q214 = Q214_UNITY;
    
    /* Ensure caches are coherent after init */
    DSB();
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
    
    ctx->num_active = 0;
    ctx->output_position = sample_pos;
    ctx->next_seg_idx = 0;
    
    /* Binary search would be faster for large segment counts,
     * but linear is fine for typical use cases */
    while (ctx->next_seg_idx < ctx->num_segments) {
        const SpectralSegmentQ15* seg = &ctx->segments[ctx->next_seg_idx];
        if (seg->start + seg->length > sample_pos) break;
        ctx->next_seg_idx++;
    }
}

/* Loading */

SpectralError spectral_arm32_load(SpectralArm32Ctx* ctx,
                                     const SpectralSegmentQ15* data,
                                     uint32_t num_segments,
                                     uint32_t output_len) {
    if (!ctx || !data) return SPECTRAL_ERR_PARAM;
    if (num_segments > ctx->segments_capacity) return SPECTRAL_ERR_OVERFLOW;
    
    memcpy((void*)ctx->segments, data, num_segments * sizeof(SpectralSegmentQ15));
    ctx->num_segments = num_segments;
    ctx->output_length = output_len;
    
    /* Ensure SDRAM writes are complete before synthesis */
    DSB();
    
    spectral_arm32_reset(ctx);
    return SPECTRAL_OK;
}

/* Parameters */

void spectral_arm32_set_amplitude(SpectralArm32Ctx* ctx, float amplitude) {
    if (!ctx) return;
    amplitude = CLAMP(amplitude, 0.0f, 1.0f);
    ctx->amplitude_q15 = (q15_t)(amplitude * Q15_ONE);
}

void spectral_arm32_set_stretch(SpectralArm32Ctx* ctx, float stretch) {
    if (!ctx) return;
    stretch = CLAMP(stretch, 0.1f, 10.0f);
    ctx->stretch_q214 = (q15_t)(stretch * Q214_UNITY);
}

/* Inner Synthesis Loop - ARM M7 Optimized */

#if SPECTRAL_ARM_M7

/* Core unrolled synthesis: accumulates samples from blk_start to blk_end.
 * No fade applied — used for sustain region and as building block for fade regions. */
SPECTRAL_ITCM
static inline void synth_core_m7(
    q31_t* RESTRICT accum,
    const q15_t* RESTRICT osc_lut,
    uint32_t blk_start,
    uint32_t blk_end,
    q31_t* RESTRICT phase,
    q15_t* RESTRICT amp,
    q31_t freq_inc,
    q15_t amp_delta
) {
    uint32_t j = blk_start;
    uint32_t len = blk_end - blk_start;
    uint32_t len4 = len & ~3U;
    uint32_t end4 = blk_start + len4;

    SPECTRAL_UNROLL_4
    for (; j < end4; j += 4) {
        PREFETCH_WRITE(&accum[j + 8]);

        q31_t p0, p1, p2, p3;
        SPECTRAL_PHASES_4(*phase, freq_inc, p0, p1, p2, p3);

        q15_t a0, a1, a2, a3;
        SPECTRAL_AMPS_4(*amp, amp_delta, a0, a1, a2, a3);

        uq16_t idx[4] = {
            (uq16_t)(p0 >> 16), (uq16_t)(p1 >> 16),
            (uq16_t)(p2 >> 16), (uq16_t)(p3 >> 16)
        };
        q15_t samples[4];
        osc_lut_lookup_4(osc_lut, idx, samples);

        SPECTRAL_ACCUM_4(accum, j, samples, a0, a1, a2, a3);

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
 * fade_val/fade_step are Q15 ramp state (0→Q15_ONE for fade-in, Q15_ONE→0 for fade-out). */
SPECTRAL_ITCM
static inline void synth_fade_m7(
    q31_t* RESTRICT accum,
    const q15_t* RESTRICT osc_lut,
    uint32_t blk_start,
    uint32_t blk_end,
    q31_t* RESTRICT phase,
    q15_t* RESTRICT amp,
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
 * seg_length: total segment length in samples */
SPECTRAL_ITCM
static inline void synth_segment_m7(
    q31_t* RESTRICT accum,
    const q15_t* RESTRICT osc_lut,
    uint32_t blk_start,
    uint32_t blk_end,
    q31_t phase_start,
    q31_t freq_inc,
    q15_t amp_start,
    q15_t amp_delta,
    q31_t* phase_out,
    q15_t* amp_out,
    uint32_t seg_offset,
    uint32_t seg_length
) {
    q31_t phase = phase_start;
    q15_t amp = amp_start;

    /* Compute fade boundaries within the segment */
    uint32_t fade_len = SPECTRAL_FADE_EMBEDDED;
    if (fade_len > seg_length / 2) fade_len = seg_length / 2;
    if (fade_len == 0) fade_len = 1;

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
        q15_t fade_val = (q15_t)((int32_t)seg_offset * FADE_STEP_Q15);
        synth_fade_m7(accum, osc_lut, orig_blk_start, fi_end,
                      &phase, &amp, freq_inc, amp_delta,
                      fade_val, FADE_STEP_Q15);
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
        q15_t fade_val = Q15_ONE - (q15_t)((int32_t)samples_into_fade * FADE_STEP_Q15);
        synth_fade_m7(accum, osc_lut, fo_start, blk_end,
                      &phase, &amp, freq_inc, amp_delta,
                      fade_val, -FADE_STEP_Q15);
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
    PERF_PROCESS_START();
    
    if (UNLIKELY(!ctx || !out_left)) { PERF_PROCESS_END(0); return 0; }
    if (UNLIKELY(ctx->num_segments == 0 || ctx->output_position >= ctx->output_length)) {
#if SPECTRAL_USE_CMSIS
        arm_fill_q15(0, out_left, num_samples);
        if (out_right) arm_fill_q15(0, out_right, num_samples);
#else
        memset(out_left, 0, num_samples * sizeof(q15_t));
        if (out_right) memset(out_right, 0, num_samples * sizeof(q15_t));
#endif
        PERF_PROCESS_END(0);
        return 0;
    }
    
    if (UNLIKELY(num_samples > 256)) {
        SPECTRAL_DBG("arm32: block size %u truncated to 256", (unsigned)num_samples);
        num_samples = 256;
    }

    const uint32_t out_pos = ctx->output_position;
    const uint32_t out_end = out_pos + num_samples;
    const q15_t* RESTRICT osc_lut = ctx->osc_lut;
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
#if SPECTRAL_HAS_DMA && SPECTRAL_HAS_SDRAM
    spectral_arm32_dma_prefetch(ctx);
#elif SPECTRAL_HAS_SDRAM
    if (ctx->next_seg_idx < ctx->num_segments) {
        prefetch_segment(&ctx->segments[ctx->next_seg_idx]);
        if (ctx->next_seg_idx + 1 < ctx->num_segments)
            prefetch_segment(&ctx->segments[ctx->next_seg_idx + 1]);
    }
#endif
    
    /* Activate new segments that start within this block */
    uint32_t _seg_scan_start = get_cycles();
    while (ctx->next_seg_idx < ctx->num_segments &&
           ctx->num_active < SPECTRAL_ARM32_MAX_ACTIVE) {
        const SpectralSegmentQ15* seg = get_segment(ctx, ctx->next_seg_idx);
        if (seg->start >= out_end) break;

        /* Prefetch next segment while processing this one */
#if SPECTRAL_HAS_SDRAM
        if (ctx->next_seg_idx + 2 < ctx->num_segments) {
            prefetch_segment(&ctx->segments[ctx->next_seg_idx + 2]);
        }
#endif

        uint32_t seg_end = seg->start + seg->length;
        if (seg_end > out_pos) {
            uint16_t slot = ctx->num_active;

            /* Calculate how many samples into the segment we are */
            uint32_t sample_offset = (out_pos > seg->start) ? (out_pos - seg->start) : 0;

            /* Convert Q8.8 frequency to phase increment */
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
#else
            SpectralActiveSegment* act = &ctx->active[slot];
            act->seg_idx = ctx->next_seg_idx;
            act->freq_inc = freq_inc;
            act->phase_acc = phase_acc;
            act->amp_current = amp_cur;
            act->amp_delta = seg->da_q15;
#endif
            ctx->num_active++;
        }
        ctx->next_seg_idx++;
    }
    profile_update(&s_perf_stats.segment_scan, get_cycles() - _seg_scan_start);
    
    /* Track peak polyphony */
    if (UNLIKELY(ctx->num_active > ctx->peak_active)) {
        ctx->peak_active = ctx->num_active;
    }
    
    /* Process all active segments */
    uint32_t _osc_start = get_cycles();
    uint16_t i = 0;
    while (i < ctx->num_active) {
#if SPECTRAL_SOA_ACTIVE
        const SpectralSegmentQ15* seg = get_segment(ctx, ctx->active_soa.seg_idx[i]);
#else
        SpectralActiveSegment* act = &ctx->active[i];
        const SpectralSegmentQ15* seg = get_segment(ctx, act->seg_idx);
#endif
        uint32_t seg_end = seg->start + seg->length;

        /* Remove expired segments */
        if (UNLIKELY(out_pos >= seg_end)) {
#if SPECTRAL_SOA_ACTIVE
            /* SoA removal: copy last element's fields into slot i */
            uint16_t last = --ctx->num_active;
            ctx->active_soa.phase_acc[i]   = ctx->active_soa.phase_acc[last];
            ctx->active_soa.freq_inc[i]    = ctx->active_soa.freq_inc[last];
            ctx->active_soa.amp_current[i] = ctx->active_soa.amp_current[last];
            ctx->active_soa.amp_delta[i]   = ctx->active_soa.amp_delta[last];
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
        uint32_t blk_start = (seg->start > out_pos) ? (seg->start - out_pos) : 0;
        uint32_t blk_end = (seg_end < out_end) ? (seg_end - out_pos) : num_samples;

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
            uint32_t seg_offset = (uint32_t)(out_pos + blk_start) - seg->start;
            synth_segment_m7(accum, osc_lut, blk_start, blk_end,
                             phase, freq_inc, amp, d_amp,
                             &phase, &amp, seg_offset, seg->length);
        }
#else
        /* Generic inner loop with linear fade envelope */
        {
            uint32_t seg_offset = (uint32_t)(out_pos + blk_start) - seg->start;
            uint32_t seg_len = seg->length;
            uint32_t fade_len = SPECTRAL_FADE_EMBEDDED;
            if (fade_len > seg_len / 2) fade_len = seg_len / 2;
            if (fade_len == 0) fade_len = 1;

            uint32_t seg_fo_start = seg_len - fade_len;
            uint32_t blk_len = blk_end - blk_start;

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
            q15_t fade_val = (q15_t)((int32_t)seg_offset * FADE_STEP_Q15);
            for (uint32_t j = blk_start; j < fi_end; j++) {
                q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                sample = spectral_mul_q15(sample, fade_val);
                accum[j] = spectral_mac_q15(accum[j], sample, amp);
                phase += freq_inc;
                amp = spectral_qadd16(amp, d_amp);
                fade_val = spectral_qadd16(fade_val, FADE_STEP_Q15);
            }

            /* Sustain - 4 samples at a time */
            uint32_t j = fi_end;
            uint32_t sustain_len = fo_start - fi_end;
            uint32_t sustain_len4 = sustain_len & ~3U;
            uint32_t end4 = fi_end + sustain_len4;

            SPECTRAL_UNROLL_4
            for (; j < end4; j += 4) {
                q31_t p0, p1, p2, p3;
                SPECTRAL_PHASES_4(phase, freq_inc, p0, p1, p2, p3);

                q15_t a0, a1, a2, a3;
                SPECTRAL_AMPS_4(amp, d_amp, a0, a1, a2, a3);

                q15_t samples[4];
                samples[0] = spectral_lut_sin((uq16_t)(p0 >> 16), osc_lut);
                samples[1] = spectral_lut_sin((uq16_t)(p1 >> 16), osc_lut);
                samples[2] = spectral_lut_sin((uq16_t)(p2 >> 16), osc_lut);
                samples[3] = spectral_lut_sin((uq16_t)(p3 >> 16), osc_lut);

                SPECTRAL_ACCUM_4(accum, j, samples, a0, a1, a2, a3);

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
                fade_val = Q15_ONE - (q15_t)((int32_t)into_fade * FADE_STEP_Q15);
                for (; j < blk_end; j++) {
                    q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                    sample = spectral_mul_q15(sample, fade_val);
                    accum[j] = spectral_mac_q15(accum[j], sample, amp);
                    phase += freq_inc;
                    amp = spectral_qadd16(amp, d_amp);
                    fade_val = spectral_qadd16(fade_val, -FADE_STEP_Q15);
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
    profile_update(&s_perf_stats.oscillator, get_cycles() - _osc_start);
    PERF_TRACK_ACTIVE(ctx->num_active);
    
    /* 
     * Convert Q31 accumulator to Q15 output with master gain
     */
    uint32_t _amp_start = get_cycles();
    spectral_q31_to_q15_scaled(accum, out_left, num_samples, master_amp);
    
    if (out_right && out_right != out_left) {
#if SPECTRAL_USE_CMSIS
        arm_copy_q15(out_left, out_right, num_samples);
#else
        memcpy(out_right, out_left, num_samples * sizeof(q15_t));
#endif
    }
    profile_update(&s_perf_stats.amplitude, get_cycles() - _amp_start);
    
    ctx->output_position = out_end;
    PERF_PROCESS_END(num_samples);
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

#if defined(SPECTRAL_RESTRICTED_MODE)

/* Cycle-count profiling for optimization tuning */
#if SPECTRAL_DEBUG_RESTRICTED
static uint32_t s_last_cycle_count = 0;
static uint32_t s_peak_cycle_count = 0;

/* Per-function cycle tracking */
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

/* Get current cycle count (Cortex-M7 DWT) */
static inline uint32_t get_cycles(void) {
#if defined(DWT) && defined(DWT_CYCCNT)
    return DWT->CYCCNT;
#else
    return 0;
#endif
}

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
}

/* Update function profile statistics */
static inline void profile_update(FunctionProfile* prof, uint32_t cycles) {
    prof->total_cycles += cycles;
    prof->call_count++;
    if (cycles < prof->min_cycles || prof->min_cycles == 0) prof->min_cycles = cycles;
    if (cycles > prof->max_cycles) prof->max_cycles = cycles;
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

/* Hook for tracking within process function */
#define PERF_PROCESS_START()    uint32_t _perf_start = get_cycles()
#define PERF_PROCESS_END(n)     do { \
    uint32_t _elapsed = get_cycles() - _perf_start; \
    profile_update(&s_perf_stats.process, _elapsed); \
    s_perf_stats.samples_processed += (n); \
} while(0)
#define PERF_TRACK_ACTIVE(n)    s_perf_stats.segments_active_total += (n)

#else /* !SPECTRAL_DEBUG_RESTRICTED */

#define PERF_PROCESS_START()
#define PERF_PROCESS_END(n)
#define PERF_TRACK_ACTIVE(n)

#endif /* SPECTRAL_DEBUG_RESTRICTED */

#endif /* SPECTRAL_RESTRICTED_MODE */

#endif /* SPECTRAL_EMBEDDED */
