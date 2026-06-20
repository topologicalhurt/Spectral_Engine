/* spectral_synth_arm32.c - Q15 Fixed-Point Synthesis for ARM Cortex-M7.
 *
 * Targets the Daisy Seed / STM32H7. Correct and host-verifiable (the real path
 * runs on the host via spectral_q.h portable fallbacks; tests/arm_core asserts
 * its audio). On Cortex-M7, -mcpu=cortex-m7 defines __ARM_FEATURE_DSP, so the
 * single-lane saturating DSP intrinsics (__smulbb, __qadd16, __qadd32) are used.
 *
 * What is actually true today:
 *   - Phase is a UQ0.32 accumulator (full circle == 2^32); the per-sample
 *     oscillator is the coupled-form Q31 rotation (spectral_coupled_step), output
 *     taken from the Q31 sine >>16 to Q15.
 *   - An active-segment list avoids scanning inactive segments.
 *   - The hot loop (synth_core_m7) is a SCALAR Q31 recurrence + Q15 MAC, unrolled
 *     by 4, segment-major.
 *   - Dual 16-bit MAC IS realized: synth_core_pair_m7 folds two full-sustain voices'
 *     (sine,amp) products into the q63 accumulator with one SMLALD (gated on the
 *     SPECTRAL_HAS_DUAL_MAC capability, codegen-confirmed).
 *
 * Resource model -- FPU vs ALU (see docs/core_audit/EMBEDDED_RESOURCE_SEPARATION_PLAN.md):
 *   - The per-sample recurrence is PURE Q31/Q15 integer (ALU/DSP). No float.
 *   - The ONLY floating-point work is the per-voice oscillator seed
 *     (spectral_coupled_freq_constants + _seed_state, double precision), paid ONCE at
 *     voice ACTIVATION (fpu-03/fpu-01). The invariant (cos_w,sin_w) are stored in the
 *     active record; the (c,s) state CARRIES across blocks, bounded by a cheap ALU
 *     renorm -- so the audio steady state touches no FPU at all. Clean temporal
 *     ownership; no FPU/ALU overlap (the recurrence is serial -- see the plan section 0).
 *
 * Memory placement -- resolved by the BSP, objdump-checkable, NOT inert:
 *   - With no BSP (host / simulation / unbound target) SPECTRAL_MEM_FAST / _BULK /
 *     _FAST_CODE are portable no-ops (default memory).
 *   - On the Daisy build SPECTRAL_BSP_MEM_HEADER binds SPECTRAL_MEM_FAST ->
 *     .dtcmram_bss, so the q63 accumulator (the one SPECTRAL_MEM_FAST static here) lands
 *     in DTCM (zero wait-state). The active-voice CONTEXT (ctx_/synth state, incl. the
 *     fpu-01 osc_c/osc_s carries) is NOT tagged -> default .bss/SRAM; pinning it (or just
 *     the hot SoA carries) to DTCM is a tracked on-target follow-up (dtcm-ctx in the plan).
 *     SPECTRAL_MEM_BULK -> .sdram_bss. SPECTRAL_MEM_FAST_CODE stays a no-op (libDaisy maps
 *     no ITCM section). Verify after a firmware link with the objdump recipe in
 *     api/daisy_seed/daisy_seed_mem.h.
 *   - LUT residency: ONLY the generic (non-M7) fallback gathers the sine LUT. The M7
 *     path uses the gather-free coupled oscillator and never reads the LUT during
 *     synthesis, so "cache misses" do not apply on M7 (the LUT is still required
 *     non-null as a precondition -- tracked in the plan).
 *   - Cycle/WCET numbers come from the validated M7 measurement stack
 *     (m7-census / m7-stalls / m7-wcet); the old in-C cost model was retired
 *     (uncalibrated, priced an obsolete kernel shape).
 */

#include "spectral_synth_arm32.h"
#include "spectral_config.h"
#include "spectral_io.h"
#include "spectral_lut.h"
#include "spectral_osc_q31.h"
#include "spectral_perf_accounting.h"
#include "spectral_utils.h"

#if SPECTRAL_EMBEDDED

#include <string.h>

/* Full data-synchronisation barrier. Defined here (ahead of the DMA RX path,
 * which calls it) and reused by the live synth path below. On a real Cortex-M
 * this is a `dsb sy` (== `dsb 0xF`, the same encoding CMSIS __DSB() emits); on
 * a forced-M7 host-sim it degrades to a seq-cst thread fence (ordering only).
 * Single source of truth for the barrier so the dormant DMA path and the live
 * path cannot drift. Guard kept identical to the prefetch helpers' guard. */
#if SPECTRAL_ARM_M7 && defined(__GNUC__)
static inline void spectral_data_sync_barrier(void) {
#if defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7M__)
    __asm__ volatile ("dsb" ::: "memory");   /* real Cortex-M: full data sync barrier */
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST); /* host (incl. forced-M7 host-sim): ordering only */
#endif
}
#else
static inline void spectral_data_sync_barrier(void) {
}
#endif

/* Maximum samples this kernel renders per spectral_arm32_process() call. The
 * static DTCM accumulator (accum[]) and the interleaved scratch (temp[]) are
 * sized to this, and a larger num_samples is truncated to it. This is the
 * engine's canonical embedded block size (SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE,
 * spectral_config.h) -- the same block the M7 perf model / WCET budget assume
 * (daisy_seed_config.h: "128 active voices at 256-sample blocks"). accum[]/temp[]
 * are declared [SPECTRAL_ARM32_MAX_BLOCK], so they can never desync from the cap; the
 * assert below just bounds it to a sane range. The Daisy firmware overrides
 * SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE down to 64 (>= its 48-sample codec block, with
 * headroom) via daisy-config.cmake -- reclaiming the DTCM the 256 cap over-allocated for
 * accum[]/temp[] (SPECTRAL_MEM_FAST -> .dtcmram_bss). Host/sim/test builds keep 256 (they
 * exercise 256-sample blocks: test_embedded_perf, test_proc_mask_honesty). */
#define SPECTRAL_ARM32_MAX_BLOCK SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE
_Static_assert(SPECTRAL_ARM32_MAX_BLOCK >= 48u && SPECTRAL_ARM32_MAX_BLOCK <= 256u,
               "arm32 block cap must hold the real codec block (>= 48) and stay within the "
               "256-sample design ceiling; set via SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE");

/*
 * DMA prefetch from SDRAM to DTCM buffer.
 * User must provide dma_start_transfer() via HAL integration.
 */
#if SPECTRAL_HAS_DMA && SPECTRAL_ARM_M7
/* DORMANT CONFIGURATION: no build target sets SPECTRAL_HAS_DMA — the two
 * transfer hooks below are a BSP port contract and verifying them needs a
 * board. Kept type-correct by test_dormant_dma_branch_still_compiles (DTCM
 * buffer placement); the cacheable-buffer configuration additionally needs
 * the CMSIS device header (SCB cache maintenance) and so only compiles in a
 * firmware build. Fate decided in the on-target campaign. */
extern void dma_start_transfer(const void* src, void* dst, size_t bytes);
extern int  dma_transfer_complete(void);

/* CMSIS-core provides __DSB in any firmware build; this fallback (the same
 * full-system barrier encoding) keeps the TU checkable standalone. */
#ifndef __DSB
#define __DSB() __asm volatile("dsb 0xF" ::: "memory")
#endif

#ifndef SPECTRAL_ARM32_DMA_BUFFER_DTCM
#define SPECTRAL_ARM32_DMA_BUFFER_DTCM 0
#endif
/* D-cache maintenance after a DMA RX is REQUIRED whenever the buffer is cacheable, which
 * (absent an MPU non-cacheable region) is exactly when it is NOT in tightly-coupled DTCM.
 * DERIVE it from placement so the two can never be set to the silently-incoherent combination
 * the old independent default produced: a cacheable buffer (DTCM=0) with invalidation off
 * (CACHEABLE=0) -> the CPU reads stale cache after every DMA -> corrupt segment data. A target
 * that maps the buffer's SRAM region non-cacheable via the MPU may override this to 0. */
#ifndef SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE
#define SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE (!SPECTRAL_ARM32_DMA_BUFFER_DTCM)
#endif

#if SPECTRAL_ARM32_DMA_BUFFER_DTCM
#define SPECTRAL_ARM32_DMA_BUFFER_ATTR SPECTRAL_MEM_FAST
#else
#define SPECTRAL_ARM32_DMA_BUFFER_ATTR
#endif

#if SPECTRAL_USE_CMSIS
#include "arm_math.h"
#endif

/* DMA RX target: MUST be cache-line aligned and a whole number of lines long.
 * spectral_arm32_dma_rx_sync() rounds the invalidate range out to cache-line
 * boundaries; if this buffer shared a line with another (dirty) static, the
 * rounded-out SCB_InvalidateDCache_by_Addr would discard that neighbor's
 * un-written-back data. The static asserts below are the permanent guard:
 * they fail the build (arm_core_test on host + every embedded build) if the
 * alignment attribute is dropped or the size stops being a line multiple. */
static SpectralSegmentQ15 dma_seg_buf[SPECTRAL_DMA_BATCH]
    __attribute__((aligned(SPECTRAL_CACHE_LINE))) SPECTRAL_ARM32_DMA_BUFFER_ATTR;
_Static_assert(__alignof__(dma_seg_buf) >= SPECTRAL_CACHE_LINE,
               "dma_seg_buf must be cache-line aligned (DMA invalidate-by-addr coherency)");
_Static_assert(sizeof(dma_seg_buf) % SPECTRAL_CACHE_LINE == 0,
               "dma_seg_buf must span a whole number of cache lines");
static uint32_t dma_prefetch_start = 0;
static uint32_t dma_prefetch_count = 0;
static int dma_prefetch_coherent = 0;

static void spectral_arm32_dma_rx_sync(const void* ptr, size_t bytes) {
#if SPECTRAL_ARM32_DMA_BUFFER_CACHEABLE && SPECTRAL_USE_CMSIS && defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
    /* Range-rounding + overflow guards live in the host-testable helper
     * spectral_cache_invalidate_range (spectral_mem.h, pinned by cache_invalidate_range
     * in arm_core_test); only the SCB call stays firmware-only. */
    uintptr_t inv_begin;
    int32_t inv_len;
    if (spectral_cache_invalidate_range((uintptr_t)ptr, bytes, (uintptr_t)SPECTRAL_CACHE_LINE,
                                        &inv_begin, &inv_len)) {
        SCB_InvalidateDCache_by_Addr((uint32_t*)inv_begin, inv_len);
    }
#else
    (void)ptr;
    (void)bytes;
#endif
    /* Same full data-sync barrier the live path uses; see
     * spectral_data_sync_barrier() above. On real Cortex-M this is `dsb sy`
     * (identical encoding to the former inline __DSB()). */
    spectral_data_sync_barrier();
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

SPECTRAL_MAYBE_UNUSED static inline void spectral_perf_paired_voice_samples(uint32_t segment_samples) {
#if SPECTRAL_RESTRICTED_PROFILE
    spectral_perf_count_paired_voice(&s_perf_op_counts, segment_samples);
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
/* LUT-path helpers: used by the generic (non-M7) fallback; the M7 path uses the
 * gather-free coupled oscillator. MAYBE_UNUSED so the M7-only build stays warning-clean. */
SPECTRAL_MAYBE_UNUSED static inline void spectral_phase_batch4(uint32_t phase,
                                         uint32_t phase_inc,
                                         uint32_t* p0,
                                         uint32_t* p1,
                                         uint32_t* p2,
                                         uint32_t* p3) {
    const uint32_t inc2 = phase_inc << 1;
    *p0 = phase;
    *p1 = phase + phase_inc;
    *p2 = phase + inc2;
    *p3 = phase + phase_inc + inc2;
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

SPECTRAL_MAYBE_UNUSED static inline void spectral_accum_batch4(q63_t* accum,
                                         uint32_t j,
                                         const q15_t* samples,
                                         q15_t a0,
                                         q15_t a1,
                                         q15_t a2,
                                         q15_t a3) {
    accum[j] = spectral_mac_q15_64(accum[j], samples[0], a0);
    accum[j + 1] = spectral_mac_q15_64(accum[j + 1], samples[1], a1);
    accum[j + 2] = spectral_mac_q15_64(accum[j + 2], samples[2], a2);
    accum[j + 3] = spectral_mac_q15_64(accum[j + 3], samples[3], a3);
}

/* ARM M7 Prefetch & Cache */

#if SPECTRAL_ARM_M7 && defined(__GNUC__)

static SPECTRAL_MAYBE_UNUSED inline void prefetch_segment(const SpectralSegmentQ15* seg) {
    SPECTRAL_PREFETCH_READ(seg);
}

/* spectral_data_sync_barrier() is defined near the top of the TU (ahead of the
 * dormant DMA RX path that also calls it). */

#else
static SPECTRAL_MAYBE_UNUSED inline void prefetch_segment(const SpectralSegmentQ15* seg) {
    (void)seg;
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
     * backend and tests/arm_core both render the nominal frequency).
     *
     * NOTE: the activation-time product `freq_q88 * phase_inc_scale_q24` (uint32) wraps
     * mod 2^32 for omega > 2*pi (super-2x-Nyquist input). This is CORRECT, not a bug to
     * "fix" with a wider int: phase_inc is a uint32 phase increment for a uint32 phase
     * accumulator, so mod-2^32 IS the intended aliasing of an out-of-band partial. A 64-bit
     * intermediate would be truncated back to the same value; widening phase_inc would change
     * nothing the accumulator does. Valid partials (omega <= pi) never reach the wrap. */
    ctx->phase_inc_scale_q24 = (uint32_t)((double)(1u << 24) / SPECTRAL_TWO_PI + 0.5);
    ctx->amplitude_q15 = Q15_MAX;
    /* No barrier here: init only does same-core CPU writes to ctx (memset + the scalar fields
     * above), publishing nothing to a DMA/peripheral/second master, so a dsb would order nothing
     * a later same-core access to normal memory does not already see. The barriers that DO earn
     * their place are in load/load_in_place, which fence the segment-pool writes (incl. an
     * external SD/in-place fill) before the pool is exposed to synthesis. */
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

    /* Load is the segment boundary: validate payload overflow, monotonic
     * ordering, output_len bound, simultaneous-active bound, and chirp support
     * before copying (KERNEL_PATCHING_GUIDELINES: validate at file/cache load). */
    {
        SpectralError verr = spectral_arm32_validate_segment_data(data, num_segments, output_len);
        if (verr != SPECTRAL_OK) return verr;
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

/* Same segment boundary as spectral_arm32_load, but for a payload already resident
 * in ctx->segments — e.g. read in place off an SD card to avoid a second
 * pool-sized staging buffer. Validates the loaded pool, then commits (set state,
 * SDRAM barrier, reset). An untrusted-file loader MUST route through here so a
 * malformed payload (non-monotonic / >MAX_ACTIVE / out-of-bounds) is rejected
 * before synthesis, rather than overrunning the fixed-size active[] array at
 * render. The commit tail mirrors spectral_arm32_load's (a 4-line parallel kept
 * inline so that function's m7-pinned codegen is byte-for-byte untouched). */
SpectralError spectral_arm32_load_in_place(SpectralArm32Ctx* ctx,
                                           uint32_t num_segments,
                                           uint32_t output_len) {
    if (!ctx) return SPECTRAL_ERR_PARAM;
    if (num_segments > ctx->segments_capacity) return SPECTRAL_ERR_OVERFLOW;
    if (num_segments > 0u && !ctx->segments) return SPECTRAL_ERR_PARAM;

    SpectralError verr = spectral_arm32_validate_segment_data(ctx->segments, num_segments, output_len);
    if (verr != SPECTRAL_OK) return verr;

    ctx->num_segments = num_segments;
    ctx->output_length = output_len;
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

/* Stretch is a SYNTH-TIME parameter baked into freq_q88 at segment conversion, NOT a
 * runtime control: the embedded path has no re-render, so a stretch applied after load
 * cannot take effect. This setter is intentionally inert. The public Daisy stack still
 * forwards a pot/UART stretch and replies OK -- making that path fail loud (or marking it
 * unimplemented at its surface) is the honesty item tracked in
 * docs/core_audit/EMBEDDED_RESOURCE_SEPARATION_PLAN.md. Do not read this no-op as a knob. */
void spectral_arm32_set_stretch(SpectralArm32Ctx* ctx, float stretch) {
    if (!ctx) return;
    (void)stretch;
}

/* Inner Synthesis Loop - ARM M7 Optimized */

#if SPECTRAL_ARM_M7

/* Core unrolled synthesis: accumulates samples from blk_start to blk_end.
 * No fade applied — used for sustain region and as building block for fade regions. */
/* uint32 phase-accumulator units -> radians: phase/2^32 * 2*pi. DOUBLE precision (a float
 * 2*pi would mis-set the recurrence frequency and re-introduce the init drift the module
 * characterization caught). Used to seed the coupled oscillator from the canonical phase. */
#define SPECTRAL_PHASE_U32_TO_RAD (6.283185307179586232 / 4294967296.0)

SPECTRAL_MEM_FAST_CODE
static inline void synth_core_m7(
    q63_t* restrict accum,
    SpectralCoupledOsc* restrict osc,
    q31_t cos_w,
    q31_t sin_w,
    uint32_t blk_start,
    uint32_t blk_end,
    q15_t* restrict amp,
    q15_t amp_delta
) {
    // SPECTRAL_Q_DOMAIN BEGIN  -- per-sample Q31 recurrence + Q15 MAC; no float (enforced)
    SpectralCoupledOsc o = *osc;   /* recurrence runs in registers */
    q15_t am = *amp;
    uint32_t j = blk_start;
    uint32_t len = blk_end - blk_start;
    uint32_t end4 = blk_start + (len & ~3U);

    SPECTRAL_UNROLL_4
    for (; j < end4; j += 4) {
        SPECTRAL_PREFETCH_WRITE(&accum[j + 8]);

        q15_t a0, a1, a2, a3;
        spectral_amp_batch4(am, amp_delta, &a0, &a1, &a2, &a3);

        /* output-then-step == the LUT's lookup-then-increment: o.s is sin(phase) for this
         * sample, then the recurrence rotates to the next (serial; no cross-sample ILP). */
        q15_t s0 = (q15_t)(o.s >> 16); spectral_coupled_step(&o, cos_w, sin_w);
        q15_t s1 = (q15_t)(o.s >> 16); spectral_coupled_step(&o, cos_w, sin_w);
        q15_t s2 = (q15_t)(o.s >> 16); spectral_coupled_step(&o, cos_w, sin_w);
        q15_t s3 = (q15_t)(o.s >> 16); spectral_coupled_step(&o, cos_w, sin_w);

        accum[j]     = spectral_mac_q15_64(accum[j],     s0, a0);
        accum[j + 1] = spectral_mac_q15_64(accum[j + 1], s1, a1);
        accum[j + 2] = spectral_mac_q15_64(accum[j + 2], s2, a2);
        accum[j + 3] = spectral_mac_q15_64(accum[j + 3], s3, a3);

        am = spectral_qadd16(a3, amp_delta);
    }

    for (; j < blk_end; j++) {
        q15_t sample = (q15_t)(o.s >> 16);
        spectral_coupled_step(&o, cos_w, sin_w);
        accum[j] = spectral_mac_q15_64(accum[j], sample, am);
        am = spectral_qadd16(am, amp_delta);
    }
    *osc = o;
    *amp = am;
    // SPECTRAL_Q_DOMAIN END
}

#if SPECTRAL_HAS_DUAL_MAC
/* Two-voice sustain accumulation via the M7 dual 16-bit MAC (SMLALD): both voices'
 * sine*amp products fold into the q63 accumulator in one instruction. Because the q63
 * accumulate is exact (addition is associative with no overflow), this is bit-identical
 * to two separate synth_core_m7 voices -- it just halves the MAC instructions and the
 * accumulator read-modify-writes for the paired voices. */
SPECTRAL_MEM_FAST_CODE
static inline void synth_core_pair_m7(
    q63_t* restrict accum,
    uint32_t blk_start,
    uint32_t blk_end,
    uint32_t* restrict phaseA, q15_t* restrict ampA, uint32_t phase_incA, q15_t amp_deltaA,
    q31_t cwA, q31_t swA, SpectralCoupledOsc* restrict oscA,
    uint32_t* restrict phaseB, q15_t* restrict ampB, uint32_t phase_incB, q15_t amp_deltaB,
    q31_t cwB, q31_t swB, SpectralCoupledOsc* restrict oscB
) {
    /* Rotation constants cwA/swA/cwB/swB are per-voice activation constants; the (c,s) state is
     * CARRIED in oscA/oscB across blocks (no per-block f64 seed), one renorm per block each. */
    // SPECTRAL_Q_DOMAIN BEGIN  -- dual-MAC per-sample Q31 recurrence; no float (enforced)
    SpectralCoupledOsc oA = *oscA, oB = *oscB;
    /* Carry both voices' amps PACKED in one register (low lane = A, high = B): the per-sample
     * ramp is ONE QADD16 for the pair, and SMLALD consumes the packed amps directly -- saving
     * the second QADD16 and the per-call repack. Bit-identical (each 16-bit lane is independent). */
    uint32_t amps   = ((uint32_t)(uint16_t)*ampB << 16) | (uint16_t)*ampA;
    uint32_t deltas = ((uint32_t)(uint16_t)amp_deltaB << 16) | (uint16_t)amp_deltaA;
    for (uint32_t j = blk_start; j < blk_end; j++) {
        SPECTRAL_PREFETCH_WRITE(&accum[j + 8]);
        q15_t sA = (q15_t)(oA.s >> 16); spectral_coupled_step(&oA, cwA, swA);
        q15_t sB = (q15_t)(oB.s >> 16); spectral_coupled_step(&oB, cwB, swB);
        uint32_t samples = ((uint32_t)(uint16_t)sB << 16) | (uint16_t)sA;
        accum[j] = spectral_smlald_packed(accum[j], samples, amps);
        amps = spectral_qadd16x2(amps, deltas);
    }
    spectral_coupled_renorm(&oA);  *oscA = oA;
    spectral_coupled_renorm(&oB);  *oscB = oB;
    uint32_t blk_len = blk_end - blk_start;
    *phaseA += phase_incA * blk_len;  *ampA = (q15_t)(int16_t)(amps & 0xFFFFu);
    *phaseB += phase_incB * blk_len;  *ampB = (q15_t)(int16_t)(amps >> 16);
    // SPECTRAL_Q_DOMAIN END
}

#endif /* SPECTRAL_HAS_DUAL_MAC */

/* Fade region synthesis: applies linear Q15 fade ramp to each sample.
 * fade_val/fade_step are Q15 ramp state (0->Q15_MAX for fade-in, Q15_MAX->0 for fade-out).
 * Backend parity note: envelope shape matches GPU/desktop fade semantics but uses
 * fixed-point arithmetic to preserve embedded determinism. */
SPECTRAL_MEM_FAST_CODE
static inline void synth_fade_m7(
    q63_t* restrict accum,
    SpectralCoupledOsc* restrict osc,
    q31_t cos_w,
    q31_t sin_w,
    uint32_t blk_start,
    uint32_t blk_end,
    q15_t* restrict amp,
    q15_t amp_delta,
    q15_t fade_val,
    q15_t fade_step
) {
    // SPECTRAL_Q_DOMAIN BEGIN  -- per-sample Q31 recurrence + Q15 fade ramp; no float (enforced)
    SpectralCoupledOsc o = *osc;
    q15_t am = *amp;
    for (uint32_t j = blk_start; j < blk_end; j++) {
        q15_t sample = (q15_t)(o.s >> 16);
        spectral_coupled_step(&o, cos_w, sin_w);
        q15_t faded = spectral_mul_q15(sample, fade_val);
        accum[j] = spectral_mac_q15_64(accum[j], faded, am);
        am = spectral_qadd16(am, amp_delta);
        fade_val = spectral_qadd16(fade_val, fade_step);
    }
    *osc = o;
    *amp = am;
    // SPECTRAL_Q_DOMAIN END
}

/* Full segment synthesis with linear fade envelope.
 * seg_offset: position within segment at blk_start
 * seg_length: total segment length in samples
 * Backend parity note: this path preserves the same three-region fade partition
 * semantics used in desktop/GPU backends, but executes fully in Q15/Q31. */
SPECTRAL_MEM_FAST_CODE
static inline void synth_segment_m7(
    q63_t* restrict accum,
    uint32_t blk_start,
    uint32_t blk_end,
    uint32_t phase_start,
    uint32_t phase_inc,
    q31_t cos_w,
    q31_t sin_w,
    SpectralCoupledOsc* restrict osc_io,
    q15_t amp_start,
    q15_t amp_delta,
    uint32_t* phase_out,
    q15_t* amp_out,
    uint32_t seg_offset,
    uint32_t seg_length,
    uint32_t fade_len
) {
    q15_t amp = amp_start;

    /* Continue the rotating state (c,s) from where the previous block left off -- carried in
     * osc_io, not re-seeded from phase (no per-block f64 sine). cos_w/sin_w are the per-voice
     * constants. One renorm at the end of the block bounds the slow Q31 drift. */
    SpectralCoupledOsc osc = *osc_io;

    uint32_t seg_fade_out_start = seg_length - fade_len;
    uint32_t blk_len = blk_end - blk_start;
    uint32_t seg_end_in_blk = seg_offset + blk_len;

    /* Per-segment fade step: the ramp must span [0, Q15_MAX] over the ACTUAL
     * fade_len, which the activator clamps to seg_length/2 (down to 1) for short
     * segments. A fixed step (Q15_MAX/SPECTRAL_FADE_SAMPLES_EMBEDDED) would leave
     * the ramp short of full scale on segments < 2*SPECTRAL_FADE_SAMPLES_EMBEDDED,
     * producing an amplitude discontinuity at the fade/sustain boundary. fade_len
     * is guaranteed >= 1 at activation, so the divide is safe. */
    q15_t fade_step = (q15_t)(Q15_MAX / fade_len);

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
        q15_t fade_val = (q15_t)((int32_t)seg_offset * fade_step);
        synth_fade_m7(accum, &osc, cos_w, sin_w, orig_blk_start, fi_end,
                      &amp, amp_delta, fade_val, fade_step);
    }

    /* Sustain region: no fade, full unrolled path */
    if (fi_end < fo_start) {
        synth_core_m7(accum, &osc, cos_w, sin_w, fi_end, fo_start, &amp, amp_delta);
    }

    /* Fade-out region */
    if (fo_start < blk_end) {
        uint32_t fo_seg_pos = seg_offset + (fo_start - orig_blk_start);
        uint32_t samples_into_fade = fo_seg_pos - seg_fade_out_start;
        q15_t fade_val = Q15_MAX - (q15_t)((int32_t)samples_into_fade * fade_step);
        synth_fade_m7(accum, &osc, cos_w, sin_w, fo_start, blk_end,
                      &amp, amp_delta, fade_val, (q15_t)-fade_step);
    }

    /* One ALU renorm per block bounds the slow Q31 magnitude drift of the carried recurrence
     * (validated to the SNR budget by test_osc_recursive at this cadence), then hand the state
     * to the next block. */
    spectral_coupled_renorm(&osc);
    *osc_io = osc;

    /* Advance the canonical phase by the whole block (kept for seek/activation consistency; the
     * oscillator no longer re-seeds from it -- it carries (c,s) across blocks). */
    *phase_out = phase_start + phase_inc * blk_len;
    *amp_out = amp;
}

#endif /* SPECTRAL_ARM_M7 */

#if SPECTRAL_HAS_DUAL_MAC
/* True if active voice k renders the WHOLE block [out_pos, out_end) entirely within its
 * sustain region (no fade-in/out this block), so it can be dual-MAC paired with another
 * such voice. fade_len is clamped to <= seg_length/2 at activation, so seg_length -
 * fade_len does not underflow. */
static inline int arm32_voice_full_sustain(const SpectralArm32Ctx* ctx, uint16_t k,
                                           uint32_t out_pos, uint32_t out_end,
                                           uint32_t num_samples) {
#if SPECTRAL_SOA_ACTIVE
    uint32_t seg_start = ctx->active_soa.seg_start[k];
    uint32_t seg_end   = ctx->active_soa.seg_end[k];
    uint32_t seg_len   = ctx->active_soa.seg_length[k];
    uint32_t fade_len  = ctx->active_soa.fade_len[k];
#else
    const SpectralActiveSegment* a = &ctx->active[k];
    uint32_t seg_start = a->seg_start;
    uint32_t seg_end   = a->seg_end;
    uint32_t seg_len   = a->seg_length;
    uint32_t fade_len  = a->fade_len;
#endif
    if (seg_start > out_pos || seg_end < out_end) return 0;          /* not full-block  */
    uint32_t seg_offset = out_pos - seg_start;                       /* blk_start == 0  */
    if (seg_offset < fade_len) return 0;                             /* still fading in */
    if (seg_offset + num_samples > seg_len - fade_len) return 0;     /* enters fade-out */
    return 1;
}

#endif /* SPECTRAL_HAS_DUAL_MAC */

/* Drop active segments that have ended at or before out_pos.
 *
 * This MUST run before the activation scan: the loader bounds simultaneous-active
 * segments with a half-open overlap model (spectral_arm32_validate_segment_data
 * counts only segments whose end is strictly past a new segment's start), so a
 * segment ending exactly at out_pos does not count toward SPECTRAL_ARM32_MAX_ACTIVE.
 * If its slot were still occupied when the activation loop runs, a new segment
 * starting at the same out_pos could be refused once num_active hits the cap and
 * then be skipped entirely — a dropped partial / lost onset the validator allowed.
 * Pruning first keeps runtime occupancy consistent with that validated model. */
static inline void spectral_arm32_prune_expired_active(SpectralArm32Ctx* ctx, uint32_t out_pos) {
    uint16_t i = 0;
    while (i < ctx->num_active) {
#if SPECTRAL_SOA_ACTIVE
        if (out_pos >= ctx->active_soa.seg_end[i]) {
            uint16_t last = --ctx->num_active;
            ctx->active_soa.phase_acc[i]   = ctx->active_soa.phase_acc[last];
            ctx->active_soa.phase_inc[i]    = ctx->active_soa.phase_inc[last];
#if SPECTRAL_ARM_M7
            ctx->active_soa.cos_w[i]       = ctx->active_soa.cos_w[last];
            ctx->active_soa.sin_w[i]       = ctx->active_soa.sin_w[last];
            ctx->active_soa.osc_c[i]       = ctx->active_soa.osc_c[last];
            ctx->active_soa.osc_s[i]       = ctx->active_soa.osc_s[last];
#endif
            ctx->active_soa.amp_current[i] = ctx->active_soa.amp_current[last];
            ctx->active_soa.amp_delta[i]   = ctx->active_soa.amp_delta[last];
            ctx->active_soa.seg_start[i]   = ctx->active_soa.seg_start[last];
            ctx->active_soa.seg_end[i]     = ctx->active_soa.seg_end[last];
            ctx->active_soa.seg_length[i]  = ctx->active_soa.seg_length[last];
            ctx->active_soa.fade_len[i]    = ctx->active_soa.fade_len[last];
#if SPECTRAL_HAS_CHIRP
            ctx->active_soa.freq_delta[i]  = ctx->active_soa.freq_delta[last];
#endif
            continue;
        }
#else
        if (out_pos >= ctx->active[i].seg_end) {
            ctx->active[i] = ctx->active[--ctx->num_active];
            continue;
        }
#endif
        i++;
    }
}

SPECTRAL_MEM_FAST_CODE
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

    if (SPECTRAL_UNLIKELY(num_samples > SPECTRAL_ARM32_MAX_BLOCK)) {
        SPECTRAL_DBG("arm32: block size %u truncated to %u",
                     (unsigned)num_samples, (unsigned)SPECTRAL_ARM32_MAX_BLOCK);
        num_samples = SPECTRAL_ARM32_MAX_BLOCK;
    }

    if (SPECTRAL_UNLIKELY(ctx->output_position >= ctx->output_length ||
                          ctx->num_segments == 0u || !ctx->segments
#if !SPECTRAL_ARM_M7
                          || !ctx->osc_lut   /* only the LUT-gather path needs the table; the M7
                                              * coupled oscillator never reads it (bandwidth-01) */
#endif
                          )) {
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
    SPECTRAL_MAYBE_UNUSED const q15_t* restrict osc_lut = ctx->osc_lut;  /* generic (non-M7) LUT path */
    const q15_t master_amp = ctx->amplitude_q15;

    /* Static q63 accumulator, cache-line aligned and tagged SPECTRAL_MEM_FAST. On the
     * Daisy build that resolves to .dtcmram_bss -> DTCM (zero wait-state); with no BSP
     * (host/sim) it is a portable no-op and the buffer lands in default memory (see the
     * placement note in the file header). 2 KB (256 q63), single-threaded audio callback,
     * no reentrancy. The width is q63 by contract: at the 128-voice budget a worst-case sum
     * is ~128 * 2^30 (Q2.30 product) = 2^37, which overflows int32, and SMLALD requires a
     * 64-bit destination -- the wide carrier lets the mix saturate exactly once at pack. */
#if defined(__GNUC__) || defined(__clang__)
    static q63_t accum[SPECTRAL_ARM32_MAX_BLOCK] __attribute__((aligned(SPECTRAL_CACHE_LINE))) SPECTRAL_MEM_FAST;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    static _Alignas(32) q63_t accum[SPECTRAL_ARM32_MAX_BLOCK];
#else
    static q63_t accum[SPECTRAL_ARM32_MAX_BLOCK];
#endif
    
    /* No per-block accum clear: spectral_q63_to_q15_scaled DRAINS (zeros) the accumulator as it
     * packs the output, and the static accum is zero-initialized, so each block's [0..num_samples]
     * is already clean (the synth only writes that range; the tail stays zero). Saves the per-block
     * zeroing pass entirely (it was a newlib memset, ~90 K dynamic insns on the counts fixture). */

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
    
    /* Free slots held by segments that ended at or before this block so the
     * activation scan sees the same occupancy the loader validated against. */
    spectral_arm32_prune_expired_active(ctx, out_pos);

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
            uint32_t phase_inc = (uint32_t)seg->freq_q88 * ctx->phase_inc_scale_q24;
            uint32_t phase_acc = ((uint32_t)((int32_t)seg->phase_q15 + 32768)) << 16;
            q15_t amp_cur;

            if (sample_offset > 0) {
                phase_acc += sample_offset * phase_inc;
                int64_t amp_target = (int64_t)seg->amp_q15 +
                                     (int64_t)seg->da_q15 * (int64_t)sample_offset;
                amp_cur = (amp_target > Q15_MAX) ? Q15_MAX
                        : (amp_target < Q15_MIN) ? Q15_MIN
                        : (q15_t)amp_target;
            } else {
                amp_cur = seg->amp_q15;
            }

#if SPECTRAL_ARM_M7
            /* Seed the per-voice oscillator ONCE here: the rotation constants depend only on
             * the constant phase_inc, and the (c,s) state is seeded from the activation phase.
             * The per-block render reuses the constants and CARRIES (c,s) across blocks (one
             * ALU renorm per block bounds drift), so no f64 sine is ever evaluated per block. */
            q31_t seg_cos_w, seg_sin_w;
            spectral_coupled_freq_constants((double)phase_inc * SPECTRAL_PHASE_U32_TO_RAD,
                                            &seg_cos_w, &seg_sin_w);
            SpectralCoupledOsc seg_osc;
            spectral_coupled_seed_state(&seg_osc, (double)phase_acc * SPECTRAL_PHASE_U32_TO_RAD);
#endif
#if SPECTRAL_SOA_ACTIVE
            ctx->active_soa.phase_inc[slot] = phase_inc;
            ctx->active_soa.phase_acc[slot] = phase_acc;
            ctx->active_soa.amp_current[slot] = amp_cur;
            ctx->active_soa.amp_delta[slot] = seg->da_q15;
            ctx->active_soa.seg_start[slot] = seg_start;
            ctx->active_soa.seg_end[slot] = seg_end;
            ctx->active_soa.seg_length[slot] = (uint16_t)seg_length;
            ctx->active_soa.fade_len[slot] = (uint16_t)fade_len;
#if SPECTRAL_ARM_M7
            ctx->active_soa.cos_w[slot] = seg_cos_w;
            ctx->active_soa.sin_w[slot] = seg_sin_w;
            ctx->active_soa.osc_c[slot] = seg_osc.c;
            ctx->active_soa.osc_s[slot] = seg_osc.s;
#endif
#else
            SpectralActiveSegment* act = &ctx->active[slot];
            act->phase_inc = phase_inc;
            act->phase_acc = phase_acc;
            act->amp_current = amp_cur;
            act->amp_delta = seg->da_q15;
            act->seg_start = seg_start;
            act->seg_end = seg_end;
            act->seg_length = (uint16_t)seg_length;
            act->fade_len = (uint16_t)fade_len;
#if SPECTRAL_ARM_M7
            act->cos_w = seg_cos_w;
            act->sin_w = seg_sin_w;
            act->osc_c = seg_osc.c;
            act->osc_s = seg_osc.s;
#endif
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

        /* Expired segments were pruned before activation, so every active here
         * satisfies seg_end > out_pos and renders a non-empty block range. */

        /* Compute block range */
        uint32_t blk_start = (seg_start > out_pos) ? (seg_start - out_pos) : 0;
        uint32_t blk_end = (seg_end < out_end) ? (seg_end - out_pos) : num_samples;
        uint32_t len = blk_end - blk_start;
        spectral_perf_segment_samples(len);

        /* Read current state */
#if SPECTRAL_SOA_ACTIVE
        uint32_t phase = ctx->active_soa.phase_acc[i];
        uint32_t phase_inc = (uint32_t)ctx->active_soa.phase_inc[i];
        q15_t amp = ctx->active_soa.amp_current[i];
        q15_t d_amp = ctx->active_soa.amp_delta[i];
#else
        uint32_t phase = act->phase_acc;
        uint32_t phase_inc = (uint32_t)act->phase_inc;
        q15_t amp = act->amp_current;
        q15_t d_amp = act->amp_delta;
#endif

        /* This #if SPECTRAL_ARM_M7 forks the whole OSCILLATOR ALGORITHM, not just DSP intrinsics:
         * the M7 arm runs the gather-free coupled-recurrence oscillator (deterministic latency, no
         * LUT), the #else arm the LUT-gather oscillator. That is a CAPABILITY ("fast tightly-coupled
         * recurrence / no-gather", config.h capability block) currently PROXIED by the CPU macro --
         * the engine's one embedded core. A second core (M4F, RISC-V) should extend a capability
         * map here, not the CPU id; the macro is kept (not a speculative new capability) per the
         * config.h "express by the CPU macro until measured" policy. */
#if SPECTRAL_ARM_M7
#if SPECTRAL_HAS_DUAL_MAC
        /* Dual-MAC fast path: when this voice and the next BOTH render the whole block
         * entirely in sustain, fold them into the q63 accumulator via SMLALD (one
         * dual-MAC, one accumulator slot for both). Bit-identical to two single-voice
         * renders -- exact q63 add is associative -- so any voice at a fade boundary or
         * a partial block falls through to the scalar path below with no loss. */
        if (i + 1u < ctx->num_active &&
            arm32_voice_full_sustain(ctx, i, out_pos, out_end, num_samples) &&
            arm32_voice_full_sustain(ctx, (uint16_t)(i + 1u), out_pos, out_end, num_samples)) {
#if SPECTRAL_SOA_ACTIVE
            uint32_t phaseB    = ctx->active_soa.phase_acc[i + 1];
            uint32_t phase_incB = (uint32_t)ctx->active_soa.phase_inc[i + 1];
            q15_t    ampB      = ctx->active_soa.amp_current[i + 1];
            q15_t    d_ampB    = ctx->active_soa.amp_delta[i + 1];
            q31_t    cwA = ctx->active_soa.cos_w[i],     swA = ctx->active_soa.sin_w[i];
            q31_t    cwB = ctx->active_soa.cos_w[i + 1], swB = ctx->active_soa.sin_w[i + 1];
            SpectralCoupledOsc oscA = { ctx->active_soa.osc_c[i],     ctx->active_soa.osc_s[i] };
            SpectralCoupledOsc oscB = { ctx->active_soa.osc_c[i + 1], ctx->active_soa.osc_s[i + 1] };
#else
            SpectralActiveSegment* actB = &ctx->active[i + 1];
            uint32_t phaseB    = actB->phase_acc;
            uint32_t phase_incB = (uint32_t)actB->phase_inc;
            q15_t    ampB      = actB->amp_current;
            q15_t    d_ampB    = actB->amp_delta;
            q31_t    cwA = act->cos_w,  swA = act->sin_w;
            q31_t    cwB = actB->cos_w, swB = actB->sin_w;
            SpectralCoupledOsc oscA = { act->osc_c,  act->osc_s };
            SpectralCoupledOsc oscB = { actB->osc_c, actB->osc_s };
#endif
            synth_core_pair_m7(accum, 0u, num_samples,
                               &phase, &amp, phase_inc, d_amp, cwA, swA, &oscA,
                               &phaseB, &ampB, phase_incB, d_ampB, cwB, swB, &oscB);
            spectral_perf_paired_voice_samples(num_samples);  /* voice B (A counted above) */
#if SPECTRAL_SOA_ACTIVE
            ctx->active_soa.phase_acc[i]       = phase;
            ctx->active_soa.amp_current[i]     = amp;
            ctx->active_soa.osc_c[i]           = oscA.c;
            ctx->active_soa.osc_s[i]           = oscA.s;
            ctx->active_soa.phase_acc[i + 1]   = phaseB;
            ctx->active_soa.amp_current[i + 1] = ampB;
            ctx->active_soa.osc_c[i + 1]       = oscB.c;
            ctx->active_soa.osc_s[i + 1]       = oscB.s;
#else
            act->phase_acc   = phase;   act->amp_current   = amp;
            act->osc_c = oscA.c;        act->osc_s = oscA.s;
            actB->phase_acc  = phaseB;  actB->amp_current  = ampB;
            actB->osc_c = oscB.c;       actB->osc_s = oscB.s;
#endif
            i += 2;
            continue;
        }
#endif /* SPECTRAL_HAS_DUAL_MAC */

        /* Use ARM M7 optimized inner loop */
        {
            uint32_t seg_offset = (uint32_t)(out_pos + blk_start) - seg_start;
#if SPECTRAL_SOA_ACTIVE
            q31_t cos_w = ctx->active_soa.cos_w[i];
            q31_t sin_w = ctx->active_soa.sin_w[i];
            SpectralCoupledOsc osc = { ctx->active_soa.osc_c[i], ctx->active_soa.osc_s[i] };
#else
            q31_t cos_w = act->cos_w;
            q31_t sin_w = act->sin_w;
            SpectralCoupledOsc osc = { act->osc_c, act->osc_s };
#endif
            synth_segment_m7(accum, blk_start, blk_end,
                             phase, phase_inc, cos_w, sin_w, &osc, amp, d_amp,
                             &phase, &amp, seg_offset, seg_length, fade_len);
#if SPECTRAL_SOA_ACTIVE
            ctx->active_soa.osc_c[i] = osc.c;
            ctx->active_soa.osc_s[i] = osc.s;
#else
            act->osc_c = osc.c;
            act->osc_s = osc.s;
#endif
        }
#else
        /* Generic inner loop with linear fade envelope */
        {
            uint32_t seg_offset = (uint32_t)(out_pos + blk_start) - seg_start;
            uint32_t seg_len = seg_length;

            uint32_t seg_fo_start = seg_len - fade_len;
            uint32_t blk_len = len;

            /* Per-segment fade step (see synth_segment_m7): ramp spans [0, Q15_MAX]
             * over the ACTUAL clamped fade_len, not a fixed Q15_MAX/32. */
            q15_t fade_step = (q15_t)(Q15_MAX / fade_len);

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
            q15_t fade_val = (q15_t)((int32_t)seg_offset * fade_step);
            for (uint32_t j = blk_start; j < fi_end; j++) {
                q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                sample = spectral_mul_q15(sample, fade_val);
                accum[j] = spectral_mac_q15_64(accum[j], sample, amp);
                phase += phase_inc;
                amp = spectral_qadd16(amp, d_amp);
                fade_val = spectral_qadd16(fade_val, fade_step);
            }

            /* Sustain - 4 samples at a time */
            uint32_t j = fi_end;
            uint32_t sustain_len = fo_start - fi_end;
            uint32_t sustain_len4 = sustain_len & ~3U;
            uint32_t end4 = fi_end + sustain_len4;

            SPECTRAL_UNROLL_4
            for (; j < end4; j += 4) {
                uint32_t p0, p1, p2, p3;
                spectral_phase_batch4(phase, phase_inc, &p0, &p1, &p2, &p3);

                q15_t a0, a1, a2, a3;
                spectral_amp_batch4(amp, d_amp, &a0, &a1, &a2, &a3);

                q15_t samples[4];
                samples[0] = spectral_lut_sin((uq16_t)(p0 >> 16), osc_lut);
                samples[1] = spectral_lut_sin((uq16_t)(p1 >> 16), osc_lut);
                samples[2] = spectral_lut_sin((uq16_t)(p2 >> 16), osc_lut);
                samples[3] = spectral_lut_sin((uq16_t)(p3 >> 16), osc_lut);

                spectral_accum_batch4(accum, j, samples, a0, a1, a2, a3);

                phase = p3 + phase_inc;
                amp = spectral_qadd16(a3, d_amp);
            }
            for (; j < fo_start; j++) {
                q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                accum[j] = spectral_mac_q15_64(accum[j], sample, amp);
                phase += phase_inc;
                amp = spectral_qadd16(amp, d_amp);
            }

            /* Fade-out */
            if (fo_start < blk_end) {
                uint32_t fo_seg_pos = seg_offset + (fo_start - blk_start);
                uint32_t into_fade = fo_seg_pos - seg_fo_start;
                fade_val = Q15_MAX - (q15_t)((int32_t)into_fade * fade_step);
                for (; j < blk_end; j++) {
                    q15_t sample = spectral_lut_sin((uq16_t)(phase >> 16), osc_lut);
                    sample = spectral_mul_q15(sample, fade_val);
                    accum[j] = spectral_mac_q15_64(accum[j], sample, amp);
                    phase += phase_inc;
                    amp = spectral_qadd16(amp, d_amp);
                    fade_val = spectral_qadd16(fade_val, (q15_t)-fade_step);
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
    
    /* The q63 accumulator holds the exact Q2.30 sum of the Q15*Q15 voice products
     * (64-bit carrier so the multi-voice sum cannot overflow). spectral_q63_to_q15_scaled
     * reduces Q2.30 -> Q15 with a >>15 shift, applies the master gain, and saturates
     * once at the output. */
    uint32_t amp_start = spectral_perf_amplitude_start();
    spectral_q63_to_q15_scaled(accum, out_left, num_samples, master_amp);
    
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
    q15_t temp[SPECTRAL_ARM32_MAX_BLOCK] __attribute__((aligned(SPECTRAL_CACHE_LINE)));
#else
    q15_t temp[SPECTRAL_ARM32_MAX_BLOCK];
#endif
    if (num_samples > SPECTRAL_ARM32_MAX_BLOCK) num_samples = SPECTRAL_ARM32_MAX_BLOCK;
    
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
