/* spectral_synth_embedded.c - Q15 Fixed-Point Synthesis Implementation
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
 *   LUT in flash (read-only, cacheable)
 */

#include "spectral_synth_embedded.h"
#include "spectral_io.h"

#if SPECTRAL_EMBEDDED

#include <string.h>

/* Platform-specific configuration */

#if defined(SPECTRAL_RESTRICTED_MODE) && SPECTRAL_ARM_M7
    #define SPECTRAL_HAS_SDRAM      1
    #define SPECTRAL_CACHE_LINE     32
    #define SPECTRAL_PREFETCH_DIST  4
#endif

#if defined(ARM_MATH_CM7) || defined(ARM_MATH_CM4)
    #define SPECTRAL_USE_CMSIS      1
    #include "arm_math.h"
    
    #define CMSIS_FILL_Q31(buf, val, len)     arm_fill_q31((val), (buf), (len))
    #define CMSIS_Q31_TO_Q15(src, dst, len)   arm_q31_to_q15((src), (dst), (len))
    #define CMSIS_SCALE_Q15(src, s, sh, dst, len)  arm_scale_q15((src), (s), (sh), (dst), (len))
    #define CMSIS_COPY_Q15(src, dst, len)     arm_copy_q15((src), (dst), (len))
#else
    #define SPECTRAL_USE_CMSIS      0
    #define CMSIS_FILL_Q31(buf, val, len)     do { for(uint32_t _i=0; _i<(len); _i++) (buf)[_i]=(val); } while(0)
    #define CMSIS_Q31_TO_Q15(src, dst, len)   /* fallback below */
    #define CMSIS_SCALE_Q15(src, s, sh, dst, len) /* fallback below */
    #define CMSIS_COPY_Q15(src, dst, len)     memcpy((dst), (src), (len) * sizeof(q15_t))
#endif

/* Performance tracking stubs (real impl at bottom of file when enabled) */
#if !(defined(SPECTRAL_RESTRICTED_MODE) && defined(SPECTRAL_DEBUG_RESTRICTED) && SPECTRAL_DEBUG_RESTRICTED)
    #define get_cycles()            0
    #define profile_update(p, c)    ((void)0)
    #define PERF_PROCESS_START()
    #define PERF_PROCESS_END()
    #define PERF_TRACK_ACTIVE(n)
#endif

/* Alias centralized macros for conciseness */
#define UNROLL_HINT         SPECTRAL_UNROLL_4
#define UNROLL_2            SPECTRAL_UNROLL_2
#define LIKELY(x)           SPECTRAL_LIKELY(x)
#define UNLIKELY(x)         SPECTRAL_UNLIKELY(x)
#define RESTRICT            __restrict__

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
    
    /* Prefetch next cache line if indices are far apart */
    if ((t1 - t0) > 16) {
        PREFETCH_READ(&lut[t1]);
    }
    
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

/* Batch oscillator lookup - processes 4 samples at once */
static inline void osc_lut_lookup_4(
    const q15_t* RESTRICT lut,
    const uq16_t* RESTRICT idx,
    q15_t* RESTRICT out
) {
    /* Prefetch upcoming LUT entries */
    PREFETCH_READ(&lut[(idx[2] >> OSC_FRAC_BITS)]);
    PREFETCH_READ(&lut[(idx[3] >> OSC_FRAC_BITS)]);
    
    for (int i = 0; i < 4; i++) {
        uint32_t t = idx[i] >> OSC_FRAC_BITS;
        uint32_t f = ((idx[i] & OSC_FRAC_MASK) * 256) >> OSC_FRAC_BITS;
        q15_t a = lut[t];
        q15_t b = lut[t + 1];
        out[i] = a + (q15_t)(((q31_t)(b - a) * (int32_t)f) >> 8);
    }
}

#endif /* SPECTRAL_ARM_M7 */

#define Q214_UNITY  16384

/* Initialization */

void spectral_embedded_init(SpectralEmbeddedCtx* ctx,
                            SpectralSegmentQ15* segments,
                            uint32_t capacity,
                            const q15_t* osc_lut,
                            uint32_t sample_rate) {
    if (!ctx) return;
    
    memset(ctx, 0, sizeof(SpectralEmbeddedCtx));
    ctx->segments = segments;
    ctx->segments_capacity = capacity;
    ctx->osc_lut = osc_lut;
    ctx->sample_rate = sample_rate;
    ctx->amplitude_q15 = Q15_ONE;
    ctx->stretch_q214 = Q214_UNITY;
    
    /* Ensure caches are coherent after init */
    DSB();
}

void spectral_embedded_reset(SpectralEmbeddedCtx* ctx) {
    if (!ctx) return;
    ctx->output_position = 0;
    ctx->num_active = 0;
    ctx->next_seg_idx = 0;
    ctx->peak_active = 0;
}

void spectral_embedded_seek(SpectralEmbeddedCtx* ctx, uint32_t sample_pos) {
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

int spectral_embedded_load(SpectralEmbeddedCtx* ctx,
                           const SpectralSegmentQ15* data,
                           uint32_t num_segments,
                           uint32_t output_len) {
    if (!ctx || !data) return -1;
    if (num_segments > ctx->segments_capacity) return -2;
    
    memcpy((void*)ctx->segments, data, num_segments * sizeof(SpectralSegmentQ15));
    ctx->num_segments = num_segments;
    ctx->output_length = output_len;
    
    /* Ensure SDRAM writes are complete before synthesis */
    DSB();
    
    spectral_embedded_reset(ctx);
    return 0;
}

/* Parameters */

void spectral_embedded_set_amplitude(SpectralEmbeddedCtx* ctx, float amplitude) {
    if (!ctx) return;
    amplitude = CLAMP(amplitude, 0.0f, 1.0f);
    ctx->amplitude_q15 = (q15_t)(amplitude * Q15_ONE);
}

void spectral_embedded_set_stretch(SpectralEmbeddedCtx* ctx, float stretch) {
    if (!ctx) return;
    stretch = CLAMP(stretch, 0.1f, 10.0f);
    ctx->stretch_q214 = (q15_t)(stretch * Q214_UNITY);
}

/* Inner Synthesis Loop - ARM M7 Optimized */

#if SPECTRAL_ARM_M7

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
    q15_t* amp_out
) {
    q31_t phase = phase_start;
    q15_t amp = amp_start;
    
    uint32_t j = blk_start;
    uint32_t len = blk_end - blk_start;
    
    /* Process groups of 4 samples */
    uint32_t len4 = len & ~3U;
    uint32_t end4 = blk_start + len4;
    
    while (j < end4) {
        /* Prefetch accumulator line */
        PREFETCH_WRITE(&accum[j + 8]);
        
        /* Compute 4 phases */
        q31_t p0 = phase;
        q31_t p1 = phase + freq_inc;
        q31_t p2 = phase + (freq_inc << 1);
        q31_t p3 = phase + freq_inc * 3;
        
        /* Compute 4 amplitudes */
        q15_t a0 = amp;
        q15_t a1 = spectral_qadd16(amp, amp_delta);
        q15_t a2 = spectral_qadd16(a1, amp_delta);
        q15_t a3 = spectral_qadd16(a2, amp_delta);
        
        /* Batch LUT lookup */
        uq16_t idx[4] = {
            (uq16_t)(p0 >> 16),
            (uq16_t)(p1 >> 16),
            (uq16_t)(p2 >> 16),
            (uq16_t)(p3 >> 16)
        };
        q15_t samples[4];
        osc_lut_lookup_4(osc_lut, idx, samples);
        
        /* Accumulate: accum += sample * amp
         * Uses explicit SMULBB + QADD intrinsics on ARM */
        accum[j]   = spectral_mac_q15(accum[j],   samples[0], a0);
        accum[j+1] = spectral_mac_q15(accum[j+1], samples[1], a1);
        accum[j+2] = spectral_mac_q15(accum[j+2], samples[2], a2);
        accum[j+3] = spectral_mac_q15(accum[j+3], samples[3], a3);
        
        /* Advance state */
        phase += freq_inc << 2;
        amp = spectral_qadd16(a3, amp_delta);
        j += 4;
    }
    
    /* Process remaining samples */
    while (j < blk_end) {
        uq16_t lut_idx = (uq16_t)(phase >> 16);
        q15_t sample = spectral_osc_lut_lookup(lut_idx, osc_lut);
        accum[j] = spectral_mac_q15(accum[j], sample, amp);
        
        phase += freq_inc;
        amp = spectral_qadd16(amp, amp_delta);
        j++;
    }
    
    *phase_out = phase;
    *amp_out = amp;
}

#endif /* SPECTRAL_ARM_M7 */

/*
 * Real-time Processing
 * 
 * Design considerations for hard real-time audio:
 *   - Fixed stack allocation (no malloc in audio callback)
 *   - Bounded loop iterations (max_active segments)
 *   - Deterministic segment activation (sorted by start time)
 *   - CMSIS-DSP vectorized operations where possible
 *   - Cache-friendly memory access patterns
 */

uint32_t spectral_embedded_process(SpectralEmbeddedCtx* ctx,
                                   q15_t* out_left,
                                   q15_t* out_right,
                                   uint32_t num_samples) {
    PERF_PROCESS_START();
    
    if (UNLIKELY(!ctx || !out_left)) { PERF_PROCESS_END(); return 0; }
    if (UNLIKELY(ctx->num_segments == 0 || ctx->output_position >= ctx->output_length)) {
        /* Fast zero-fill */
        memset(out_left, 0, num_samples * sizeof(q15_t));
        if (out_right) memset(out_right, 0, num_samples * sizeof(q15_t));
        PERF_PROCESS_END();
        return 0;
    }
    
    const uint32_t out_pos = ctx->output_position;
    const uint32_t out_end = out_pos + num_samples;
    const q15_t* RESTRICT osc_lut = ctx->osc_lut;
    const q15_t master_amp = ctx->amplitude_q15;
    
    /* Q31 accumulator - fixed stack allocation for deterministic timing */
    q31_t accum[256];
    if (num_samples > 256) num_samples = 256;
    
    /* Zero accumulator - CMSIS arm_fill_q31 uses SIMD on M7 */
    #if SPECTRAL_USE_CMSIS
    arm_fill_q31(0, accum, num_samples);
    #else
    memset(accum, 0, num_samples * sizeof(q31_t));
    #endif
    
    /* Prefetch first few segments from SDRAM */
    #if SPECTRAL_HAS_SDRAM
    if (ctx->next_seg_idx < ctx->num_segments) {
        prefetch_segment(&ctx->segments[ctx->next_seg_idx]);
        if (ctx->next_seg_idx + 1 < ctx->num_segments)
            prefetch_segment(&ctx->segments[ctx->next_seg_idx + 1]);
    }
    #endif
    
    /* Activate new segments that start within this block */
    uint32_t _seg_scan_start = get_cycles();
    while (ctx->next_seg_idx < ctx->num_segments && 
           ctx->num_active < SPECTRAL_EMBEDDED_MAX_ACTIVE) {
        const SpectralSegmentQ15* seg = &ctx->segments[ctx->next_seg_idx];
        if (seg->start >= out_end) break;
        
        /* Prefetch next segment while processing this one */
        #if SPECTRAL_HAS_SDRAM
        if (ctx->next_seg_idx + 2 < ctx->num_segments) {
            prefetch_segment(&ctx->segments[ctx->next_seg_idx + 2]);
        }
        #endif
        
        uint32_t seg_end = seg->start + seg->length;
        if (seg_end > out_pos) {
            SpectralActiveSegment* act = &ctx->active[ctx->num_active];
            act->seg_idx = ctx->next_seg_idx;
            
            /* Calculate how many samples into the segment we are */
            uint32_t sample_offset = (out_pos > seg->start) ? (out_pos - seg->start) : 0;
            
            /* Convert Q8.8 frequency to phase increment
             * freq_q88 is Hz * 256, need phase increment per sample in 32-bit accumulator
             * freq_inc = (freq_hz * 2^32) / sample_rate = (freq_q88 * 2^24) / sample_rate
             */
            uint32_t freq_scaled = (uint32_t)seg->freq_q88 * 256;
            act->freq_inc = (q31_t)((freq_scaled << 16) / ctx->sample_rate);
            
            /* Initialize phase: convert Q15 [-32768,32767] to upper 16 bits of 32-bit accumulator
             * Adding 32768 converts to unsigned [0,65535], then shift left 16 to fill upper half
             */
            act->phase_acc = ((q31_t)seg->phase_q15 + 32768) << 16;
            
            /* Advance phase and amplitude if segment started before current position */
            if (sample_offset > 0) {
                act->phase_acc += sample_offset * act->freq_inc;
                /* Advance amplitude using saturating add to prevent overflow */
                q31_t amp_advance = (q31_t)seg->da_q15 * (q31_t)sample_offset;
                act->amp_current = spectral_ssat16((q31_t)seg->amp_q15 + amp_advance);
            } else {
                act->amp_current = seg->amp_q15;
            }
            act->amp_delta = seg->da_q15;
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
        SpectralActiveSegment* act = &ctx->active[i];
        const SpectralSegmentQ15* seg = &ctx->segments[act->seg_idx];
        uint32_t seg_end = seg->start + seg->length;
        
        /* Remove expired segments */
        if (UNLIKELY(out_pos >= seg_end)) {
            *act = ctx->active[--ctx->num_active];
            continue;
        }
        
        /* Compute block range */
        uint32_t blk_start = (seg->start > out_pos) ? (seg->start - out_pos) : 0;
        uint32_t blk_end = (seg_end < out_end) ? (seg_end - out_pos) : num_samples;
        
        #if SPECTRAL_ARM_M7
        /* Use ARM M7 optimized inner loop */
        synth_segment_m7(accum, osc_lut, blk_start, blk_end,
                         act->phase_acc, act->freq_inc,
                         act->amp_current, act->amp_delta,
                         &act->phase_acc, &act->amp_current);
        #else
        /* Generic inner loop - 4 samples at a time with explicit computation */
        q31_t phase = act->phase_acc;
        q31_t freq_inc = act->freq_inc;
        q15_t amp = act->amp_current;
        q15_t d_amp = act->amp_delta;
        
        uint32_t j = blk_start;
        uint32_t len = blk_end - blk_start;
        uint32_t len4 = len & ~3U;
        uint32_t end4 = blk_start + len4;
        
        /* Process 4 samples per iteration */
        while (j < end4) {
            /* Pre-compute all 4 phases */
            q31_t p0 = phase;
            q31_t p1 = phase + freq_inc;
            q31_t p2 = phase + (freq_inc << 1);
            q31_t p3 = phase + freq_inc * 3;
            
            /* Pre-compute all 4 amplitudes */
            q15_t a0 = amp;
            q15_t a1 = spectral_qadd16(amp, d_amp);
            q15_t a2 = spectral_qadd16(a1, d_amp);
            q15_t a3 = spectral_qadd16(a2, d_amp);
            
            /* LUT lookups */
            q15_t s0 = spectral_osc_lut_lookup((uq16_t)(p0 >> 16), osc_lut);
            q15_t s1 = spectral_osc_lut_lookup((uq16_t)(p1 >> 16), osc_lut);
            q15_t s2 = spectral_osc_lut_lookup((uq16_t)(p2 >> 16), osc_lut);
            q15_t s3 = spectral_osc_lut_lookup((uq16_t)(p3 >> 16), osc_lut);
            
            /* Accumulate - independent operations */
            accum[j]     = spectral_mac_q15(accum[j],     s0, a0);
            accum[j + 1] = spectral_mac_q15(accum[j + 1], s1, a1);
            accum[j + 2] = spectral_mac_q15(accum[j + 2], s2, a2);
            accum[j + 3] = spectral_mac_q15(accum[j + 3], s3, a3);
            
            /* Advance state by 4 */
            phase += freq_inc << 2;
            amp = spectral_qadd16(a3, d_amp);
            j += 4;
        }
        
        /* Process remaining 0-3 samples */
        while (j < blk_end) {
            uq16_t lut_idx = (uq16_t)(phase >> 16);
            q15_t sample = spectral_osc_lut_lookup(lut_idx, osc_lut);
            accum[j] = spectral_mac_q15(accum[j], sample, amp);
            phase += freq_inc;
            amp = spectral_qadd16(amp, d_amp);
            j++;
        }
        
        act->phase_acc = phase;
        act->amp_current = amp;
        #endif
        
        i++;
    }
    profile_update(&s_perf_stats.oscillator, get_cycles() - _osc_start);
    PERF_TRACK_ACTIVE(ctx->num_active);
    
    /* 
     * Convert Q31 accumulator to Q15 output with master gain
     * CMSIS-DSP uses SIMD instructions: 
     *   arm_q31_to_q15 - saturating shift (SSAT)
     *   arm_scale_q15 - vectorized multiply with saturation
     */
    uint32_t _amp_start = get_cycles();
    #if SPECTRAL_USE_CMSIS
    arm_q31_to_q15(accum, out_left, num_samples);
    arm_scale_q15(out_left, master_amp, 0, out_left, num_samples);
    #else
    UNROLL_HINT
    for (uint32_t j = 0; j < num_samples; j++) {
        q15_t sample = spectral_q31_to_q15_sat(accum[j]);
        out_left[j] = spectral_scale_q15(sample, master_amp);
    }
    #endif
    
    /* Copy to right channel if separate - use CMSIS copy for SIMD */
    if (out_right && out_right != out_left) {
        #if SPECTRAL_USE_CMSIS
        arm_copy_q15(out_left, out_right, num_samples);
        #else
        memcpy(out_right, out_left, num_samples * sizeof(q15_t));
        #endif
    }
    profile_update(&s_perf_stats.amplitude, get_cycles() - _amp_start);
    
    ctx->output_position = out_end;
    PERF_PROCESS_END();
    return num_samples;
}

/* Interleaved Stereo Output */

uint32_t spectral_embedded_process_interleaved(SpectralEmbeddedCtx* ctx,
                                               q15_t* out_interleaved,
                                               uint32_t num_samples) {
    if (!ctx || !out_interleaved) return 0;
    
    q15_t temp[256];
    if (num_samples > 256) num_samples = 256;
    
    uint32_t written = spectral_embedded_process(ctx, temp, NULL, num_samples);
    
    /* Use shared stereo interleaving function */
    spectral_mono_to_stereo_q15(temp, out_interleaved, written);
    
    return written;
}

/* Restricted Mode Profiling */

#if defined(SPECTRAL_RESTRICTED_MODE)

/* DMA-accelerated segment prefetch (future enhancement)
 * For very large segment files, we could use DMA to prefetch segments
 * from SDRAM to SRAM while the CPU processes the current batch. */

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
    FunctionProfile process;        /* spectral_embedded_process() */
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

void restricted_synth_profile_start(void) {
    s_last_cycle_count = get_cycles();
}

uint32_t restricted_synth_profile_end(void) {
    uint32_t elapsed = get_cycles() - s_last_cycle_count;
    if (elapsed > s_peak_cycle_count) s_peak_cycle_count = elapsed;
    return elapsed;
}

uint32_t restricted_synth_get_peak_cycles(void) {
    return s_peak_cycle_count;
}

void restricted_synth_reset_profile(void) {
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

/* Get performance report (fills buffer, returns bytes written) */
int restricted_synth_get_perf_report(char* buf, int buf_len) {
    if (!buf || buf_len < 256) return 0;
    
    int n = 0;
    n += snprintf(buf + n, buf_len - n, "=== Q15 Synth Performance ===\n");
    
    if (s_perf_stats.process.call_count > 0) {
        uint32_t avg = s_perf_stats.process.total_cycles / s_perf_stats.process.call_count;
        n += snprintf(buf + n, buf_len - n, 
            "process(): calls=%lu avg=%lu min=%lu max=%lu cycles\n",
            (unsigned long)s_perf_stats.process.call_count,
            (unsigned long)avg,
            (unsigned long)s_perf_stats.process.min_cycles,
            (unsigned long)s_perf_stats.process.max_cycles);
    }
    
    if (s_perf_stats.samples_processed > 0) {
        float cycles_per_sample = (float)s_perf_stats.process.total_cycles / s_perf_stats.samples_processed;
        float avg_active = (float)s_perf_stats.segments_active_total / s_perf_stats.process.call_count;
        n += snprintf(buf + n, buf_len - n,
            "samples=%lu cycles/sample=%.1f avg_active=%.1f\n",
            (unsigned long)s_perf_stats.samples_processed,
            cycles_per_sample, avg_active);
    }
    
    n += snprintf(buf + n, buf_len - n, "peak_cycles=%lu\n", (unsigned long)s_peak_cycle_count);
    
    return n;
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
