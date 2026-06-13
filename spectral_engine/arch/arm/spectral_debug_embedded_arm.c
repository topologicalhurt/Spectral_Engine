/* spectral_debug_embedded_arm.c - ARM DWT/ITM Debug Monitoring
 * 
 * Provides real-time performance monitoring on ARM Cortex-M targets
 * using the Data Watchpoint and Trace (DWT) unit and Instrumentation
 * Trace Macrocell (ITM).
 * 
 * Features:
 *   - Cycle-accurate timing measurement via DWT_CYCCNT
 *   - Memory access profiling (DWT_LSUCNT)
 *   - Exception monitoring (DWT_EXCCNT)
 *   - Printf-style output via ITM stimulus ports
 * 
 * Platform Support:
 *   - STM32H750
 *   - Generic Cortex-M
 * 
 * Requires debug probe (ST-Link, J-Link) with SWO trace enabled.
 */

#include "spectral_debug_embedded_arm.h"
#include "spectral_utils.h"

#ifdef SPECTRAL_DEBUG_ARM

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Cortex-M7 register map and bit contracts. */
enum {
    SPECTRAL_COREDEBUG_DEMCR_ADDR = 0xE000EDFCu,
    SPECTRAL_DWT_CTRL_ADDR = 0xE0001000u,
    SPECTRAL_DWT_CYCCNT_ADDR = 0xE0001004u,
    SPECTRAL_DWT_CPICNT_ADDR = 0xE0001008u,
    SPECTRAL_DWT_EXCCNT_ADDR = 0xE000100Cu,
    SPECTRAL_DWT_SLEEPCNT_ADDR = 0xE0001010u,
    SPECTRAL_DWT_LSUCNT_ADDR = 0xE0001014u,
    SPECTRAL_DWT_FOLDCNT_ADDR = 0xE0001018u,
    SPECTRAL_ITM_STIM_BASE_ADDR = 0xE0000000u,
    SPECTRAL_ITM_TER_ADDR = 0xE0000E00u,
    SPECTRAL_ITM_TCR_ADDR = 0xE0000E80u,
    SPECTRAL_ITM_LAR_ADDR = 0xE0000FB0u
};

enum {
    SPECTRAL_COREDEBUG_DEMCR_TRCENA_MASK = (1u << 24),
    SPECTRAL_DWT_CTRL_CYCCNTENA_MASK = (1u << 0),
    SPECTRAL_DWT_CTRL_CPIEVTENA_MASK = (1u << 17),
    SPECTRAL_DWT_CTRL_EXCEVTENA_MASK = (1u << 18),
    SPECTRAL_DWT_CTRL_SLEEPEVTENA_MASK = (1u << 19),
    SPECTRAL_DWT_CTRL_LSUEVTENA_MASK = (1u << 20),
    SPECTRAL_DWT_CTRL_FOLDEVTENA_MASK = (1u << 21),
    SPECTRAL_ITM_TCR_ITMENA_MASK = (1u << 0),
    SPECTRAL_ITM_LAR_UNLOCK_VALUE = 0xC5ACCE55u,
    SPECTRAL_DEBUG_ITM_MAX_SPINS = 1024u
};

static inline volatile uint32_t* spectral_reg_ptr(uint32_t addr) {
    return (volatile uint32_t*)(uintptr_t)addr;
}

static inline uint32_t spectral_reg_read(uint32_t addr) {
    return *spectral_reg_ptr(addr);
}

static inline void spectral_reg_write(uint32_t addr, uint32_t value) {
    *spectral_reg_ptr(addr) = value;
}

static inline void spectral_reg_or(uint32_t addr, uint32_t mask) {
    *spectral_reg_ptr(addr) |= mask;
}

static inline uint32_t spectral_itm_stim_addr(uint8_t port) {
    return SPECTRAL_ITM_STIM_BASE_ADDR + (4u * (uint32_t)port);
}

/* Time source */
#if defined(SPECTRAL_PLATFORM_DAISY)
extern uint32_t daisy_system_get_now_ms(void);
static inline uint32_t spectral_debug_now_ms(void) {
    return daisy_system_get_now_ms();
}
#else
static uint32_t s_debug_ms_counter = 0;
static inline uint32_t spectral_debug_now_ms(void) {
    return s_debug_ms_counter;
}
#endif

static inline uint32_t spectral_debug_cpu_freq_hz(void) {
#if defined(SPECTRAL_CPU_FREQ_HZ)
    return (uint32_t)SPECTRAL_CPU_FREQ_HZ;
#elif defined(SPECTRAL_PLATFORM_DAISY)
    return 480000000u;
#else
    return 168000000u;
#endif
}

static inline uint32_t spectral_debug_sdram_freq_hz(void) {
#if defined(SPECTRAL_SDRAM_FREQ_HZ)
    return (uint32_t)SPECTRAL_SDRAM_FREQ_HZ;
#else
    return 100000000u;
#endif
}

static inline uint32_t spectral_debug_sdram_bus_width_bits(void) {
#if defined(SPECTRAL_SDRAM_BUS_WIDTH)
    return (uint32_t)SPECTRAL_SDRAM_BUS_WIDTH;
#else
    return 32u;
#endif
}

static inline uint32_t spectral_debug_sdram_max_bandwidth_bytes_per_sec(void) {
    return (spectral_debug_sdram_freq_hz() * spectral_debug_sdram_bus_width_bits()) / 8u;
}

/* Initialization */

void spectral_debug_init(SpectralDebugCtx* ctx, uint32_t sample_rate, uint32_t block_size) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Clear context */
    memset(ctx, 0, sizeof(SpectralDebugCtx));
    
    /* Calculate timing budget */
    ctx->realtime.sample_rate = sample_rate;
    ctx->realtime.block_size = block_size;
    ctx->realtime.deadline_cycles = (spectral_debug_cpu_freq_hz() / sample_rate) * block_size;
    ctx->realtime.is_realtime = true;
    
    /* Initialize timing stats */
    ctx->timing.cycles_min = UINT32_MAX;
    ctx->timing.budget_cycles = ctx->realtime.deadline_cycles;
    
    /* Memory pool sizes (platform-specific) */
#if defined(SPECTRAL_PLATFORM_DAISY)
    ctx->memory.sram_total = (uint32_t)(128u * SPECTRAL_BYTES_PER_KIB); /* 128KB SRAM */
    ctx->memory.sdram_total = (uint32_t)(64u * SPECTRAL_BYTES_PER_MIB); /* 64MB SDRAM */
    ctx->memory.stack_total = (uint32_t)(8u * SPECTRAL_BYTES_PER_KIB);   /* 8KB stack */
#else
    ctx->memory.sram_total = (uint32_t)(64u * SPECTRAL_BYTES_PER_KIB);   /* Generic default */
    ctx->memory.sdram_total = 0;
    ctx->memory.stack_total = (uint32_t)(4u * SPECTRAL_BYTES_PER_KIB);
#endif
    
    /* Enable DWT cycle counter */
    spectral_reg_or(SPECTRAL_COREDEBUG_DEMCR_ADDR, SPECTRAL_COREDEBUG_DEMCR_TRCENA_MASK);
    spectral_reg_or(SPECTRAL_DWT_CTRL_ADDR, SPECTRAL_DWT_CTRL_CYCCNTENA_MASK);
    
    /* Enable DWT performance counters */
    spectral_reg_or(SPECTRAL_DWT_CTRL_ADDR,
                    SPECTRAL_DWT_CTRL_CPIEVTENA_MASK |
                    SPECTRAL_DWT_CTRL_EXCEVTENA_MASK |
                    SPECTRAL_DWT_CTRL_SLEEPEVTENA_MASK |
                    SPECTRAL_DWT_CTRL_LSUEVTENA_MASK |
                    SPECTRAL_DWT_CTRL_FOLDEVTENA_MASK);
    
    /* Enable ITM */
    spectral_reg_write(SPECTRAL_ITM_LAR_ADDR, SPECTRAL_ITM_LAR_UNLOCK_VALUE);
    spectral_reg_or(SPECTRAL_ITM_TCR_ADDR, SPECTRAL_ITM_TCR_ITMENA_MASK);
    spectral_reg_or(SPECTRAL_ITM_TER_ADDR,
                    (1u << ITM_PORT_TIMING) |
                    (1u << ITM_PORT_MEMORY) |
                    (1u << ITM_PORT_CACHE) |
                    (1u << ITM_PORT_SDRAM) |
                    (1u << ITM_PORT_SDCARD) |
                    (1u << ITM_PORT_REALTIME) |
                    (1u << ITM_PORT_PRINTF));
    
    ctx->initialized = true;
    ctx->last_update_ms = spectral_debug_now_ms();
    
    spectral_debug_printf("Spectral Debug: initialized @ %lu Hz, %lu samples\n",
                          (unsigned long)sample_rate, (unsigned long)block_size);
}

void spectral_debug_deinit(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx || !ctx->initialized);
    
    spectral_debug_printf("Spectral Debug: shutdown\n");
    spectral_debug_printf("  Peak cycles: %lu\n", (unsigned long)ctx->timing.cycles_peak);
    spectral_debug_printf("  Total xruns: %lu\n", (unsigned long)ctx->realtime.xruns);
    
    ctx->initialized = false;
}

void spectral_debug_reset(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Preserve settings, reset statistics */
    memset(&ctx->timing, 0, sizeof(ctx->timing));
    memset(&ctx->cache, 0, sizeof(ctx->cache));
    memset(&ctx->sdram, 0, sizeof(ctx->sdram));
    memset(&ctx->sdcard, 0, sizeof(ctx->sdcard));
    
    ctx->timing.cycles_min = UINT32_MAX;
    ctx->timing.budget_cycles = ctx->realtime.deadline_cycles;
    
    ctx->realtime.xruns = 0;
    ctx->realtime.is_realtime = true;
    
    /* Reset DWT counters */
    spectral_reg_write(SPECTRAL_DWT_CYCCNT_ADDR, 0);
    spectral_reg_write(SPECTRAL_DWT_CPICNT_ADDR, 0);
    spectral_reg_write(SPECTRAL_DWT_EXCCNT_ADDR, 0);
    spectral_reg_write(SPECTRAL_DWT_SLEEPCNT_ADDR, 0);
    spectral_reg_write(SPECTRAL_DWT_LSUCNT_ADDR, 0);
    spectral_reg_write(SPECTRAL_DWT_FOLDCNT_ADDR, 0);
    
    spectral_debug_printf("Spectral Debug: reset\n");
}

/* Timing Measurement */

void spectral_debug_timing_start(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->dwt_start = spectral_reg_read(SPECTRAL_DWT_CYCCNT_ADDR);
}

uint32_t spectral_debug_timing_end(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_VAL_IF(!ctx, 0);
    
    uint32_t end = spectral_reg_read(SPECTRAL_DWT_CYCCNT_ADDR);
    uint32_t elapsed = end - ctx->dwt_start;
    
    /* Update timing stats */
    ctx->timing.cycles_total += elapsed;
    ctx->timing.measurement_count++;
    
    if (elapsed > ctx->timing.cycles_peak) {
        ctx->timing.cycles_peak = elapsed;
    }
    if (elapsed < ctx->timing.cycles_min) {
        ctx->timing.cycles_min = elapsed;
    }
    
    /* IIR rolling average: avg += (new - avg) / 16. The delta is formed in signed
     * arithmetic so a below-average sample lowers the mean; an unsigned (new - avg)
     * would underflow and the logical >> would add ~2^28 instead of subtracting. */
    int32_t avg_delta = (int32_t)elapsed - (int32_t)ctx->timing.cycles_avg;
    ctx->timing.cycles_avg = (uint32_t)((int32_t)ctx->timing.cycles_avg + (avg_delta >> 4));
    
    /* Check for budget overrun */
    if (elapsed > ctx->timing.budget_cycles) {
        ctx->timing.overruns++;
    }
    
    /* Update real-time stats */
    spectral_debug_block_complete(ctx, elapsed);
    
    return elapsed;
}

bool spectral_debug_is_realtime(const SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_VAL_IF(!ctx, true);
    return ctx->realtime.is_realtime;
}

float spectral_debug_cpu_load(const SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_VAL_IF(!ctx, 0.0f);
    return ctx->realtime.cpu_load_percent;
}

/* Memory Tracking */

void spectral_debug_update_memory(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Update peak values */
    if (ctx->memory.sram_used > ctx->memory.sram_peak) {
        ctx->memory.sram_peak = ctx->memory.sram_used;
    }
    if (ctx->memory.sdram_used > ctx->memory.sdram_peak) {
        ctx->memory.sdram_peak = ctx->memory.sdram_used;
    }
    
    /* Update stack high-water mark */
    spectral_debug_update_stack(ctx);
}

void spectral_debug_sdram_alloc(SpectralDebugCtx* ctx, uint32_t bytes) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->memory.sdram_used += bytes;
}

void spectral_debug_sdram_free(SpectralDebugCtx* ctx, uint32_t bytes) {
    SPECTRAL_RETURN_IF(!ctx);
    if (bytes <= ctx->memory.sdram_used) {
        ctx->memory.sdram_used -= bytes;
    }
}

void spectral_debug_sram_alloc(SpectralDebugCtx* ctx, uint32_t bytes) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->memory.sram_used += bytes;
}

void spectral_debug_sram_free(SpectralDebugCtx* ctx, uint32_t bytes) {
    SPECTRAL_RETURN_IF(!ctx);
    if (bytes <= ctx->memory.sram_used) {
        ctx->memory.sram_used -= bytes;
    }
}

void spectral_debug_update_stack(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Get current stack pointer */
    register uint32_t sp __asm("sp");
    
#if defined(SPECTRAL_PLATFORM_DAISY)
    /* Stack grows down from end of SRAM */
    extern uint32_t _estack;  /* Linker symbol */
    uint32_t stack_used = (uint32_t)&_estack - sp;
#else
    /* Generic - estimate from known stack size */
    uint32_t stack_used = ctx->memory.stack_total - (sp & 0xFFF);
#endif
    
    ctx->memory.stack_used = stack_used;
    if (stack_used > ctx->memory.stack_peak) {
        ctx->memory.stack_peak = stack_used;
    }
}

/* Cache Monitoring */

void spectral_debug_update_cache(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Note: STM32H7 doesn't have cache performance counters in DWT.
     * This would require custom instrumentation or external logic analyzer.
     * For now, we estimate based on memory access patterns. */
    
    /* Calculate hit rates if we have data */
    uint32_t i_total = ctx->cache.icache_hits + ctx->cache.icache_misses;
    uint32_t d_total = ctx->cache.dcache_hits + ctx->cache.dcache_misses;
    
    if (i_total > 0) {
        ctx->cache.icache_hit_rate = (100.0f * ctx->cache.icache_hits) / i_total;
    }
    if (d_total > 0) {
        ctx->cache.dcache_hit_rate = (100.0f * ctx->cache.dcache_hits) / d_total;
    }
}

void spectral_debug_cache_invalidate(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Reset counters */
    ctx->cache.icache_hits = 0;
    ctx->cache.icache_misses = 0;
    ctx->cache.dcache_hits = 0;
    ctx->cache.dcache_misses = 0;
    ctx->cache.dcache_writebacks = 0;
    
#if defined(__ARM_ARCH_7EM__)
    /* Invalidate and clean D-cache */
    __asm volatile ("dsb sy");
    __asm volatile ("isb sy");
#endif
}

/* SDRAM Bandwidth */

void spectral_debug_sdram_start(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    ctx->sdram.bytes_read = 0;
    ctx->sdram.bytes_written = 0;
    ctx->sdram.read_cycles = 0;
    ctx->sdram.write_cycles = 0;
}

void spectral_debug_sdram_read(SpectralDebugCtx* ctx, uint32_t bytes) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->sdram.bytes_read += bytes;
}

void spectral_debug_sdram_write(SpectralDebugCtx* ctx, uint32_t bytes) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->sdram.bytes_written += bytes;
}

void spectral_debug_sdram_update(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    uint32_t now = spectral_debug_now_ms();
    uint32_t elapsed_ms = now - ctx->last_update_ms;
    
    if (elapsed_ms < 100) return;  /* Update at most 10 Hz */
    
    /* Calculate bandwidth (MB/s) */
    float elapsed_sec = elapsed_ms / SPECTRAL_MILLIS_PER_SECOND_F;
    ctx->sdram.read_bandwidth_mbps = (float)(BYTES_TO_MB(ctx->sdram.bytes_read) / elapsed_sec);
    ctx->sdram.write_bandwidth_mbps = (float)(BYTES_TO_MB(ctx->sdram.bytes_written) / elapsed_sec);
    
    /* Calculate utilization */
    uint32_t total_bytes = ctx->sdram.bytes_read + ctx->sdram.bytes_written;
    float max_bytes_per_interval =
        (spectral_debug_sdram_max_bandwidth_bytes_per_sec() / SPECTRAL_MILLIS_PER_SECOND_F) *
        elapsed_ms;
    ctx->sdram.utilization_percent = (100.0f * total_bytes) / max_bytes_per_interval;
    
    /* Reset for next interval */
    ctx->sdram.bytes_read = 0;
    ctx->sdram.bytes_written = 0;
    ctx->last_update_ms = now;
}

/* SD Card I/O */

void spectral_debug_sdcard_read(SpectralDebugCtx* ctx, uint32_t sectors, uint32_t latency_us) {
    SPECTRAL_RETURN_IF(!ctx);
    
    ctx->sdcard.sectors_read += sectors;
    
    /* IIR average latency (signed delta: a faster-than-average sample must lower
     * the mean, not underflow the unsigned subtraction). */
    if (ctx->sdcard.read_latency_us == 0) {
        ctx->sdcard.read_latency_us = latency_us;
    } else {
        int32_t lat_delta = (int32_t)latency_us - (int32_t)ctx->sdcard.read_latency_us;
        ctx->sdcard.read_latency_us = (uint32_t)((int32_t)ctx->sdcard.read_latency_us + (lat_delta >> 3));
    }
}

void spectral_debug_sdcard_write(SpectralDebugCtx* ctx, uint32_t sectors, uint32_t latency_us) {
    SPECTRAL_RETURN_IF(!ctx);
    
    ctx->sdcard.sectors_written += sectors;
    
    /* IIR average latency (signed delta: a faster-than-average sample must lower
     * the mean, not underflow the unsigned subtraction). */
    if (ctx->sdcard.write_latency_us == 0) {
        ctx->sdcard.write_latency_us = latency_us;
    } else {
        int32_t lat_delta = (int32_t)latency_us - (int32_t)ctx->sdcard.write_latency_us;
        ctx->sdcard.write_latency_us = (uint32_t)((int32_t)ctx->sdcard.write_latency_us + (lat_delta >> 3));
    }
}

void spectral_debug_sdcard_error(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->sdcard.errors++;
}

void spectral_debug_sdcard_detect(SpectralDebugCtx* ctx, bool present) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->sdcard.card_present = present;
}

/* Real-time Monitoring */

void spectral_debug_block_complete(SpectralDebugCtx* ctx, uint32_t cycles) {
    SPECTRAL_RETURN_IF(!ctx);
    
    ctx->realtime.actual_cycles = cycles;
    ctx->realtime.slack_cycles = (int32_t)ctx->realtime.deadline_cycles - (int32_t)cycles;
    
    /* Calculate CPU load */
    ctx->realtime.cpu_load_percent = (100.0f * cycles) / ctx->realtime.deadline_cycles;
    ctx->realtime.headroom_percent = 100.0f - ctx->realtime.cpu_load_percent;
    
    if (ctx->realtime.headroom_percent < 0.0f) {
        ctx->realtime.headroom_percent = 0.0f;
    }
    
    /* Check if meeting real-time deadline */
    ctx->realtime.is_realtime = (cycles <= ctx->realtime.deadline_cycles);
    
    if (!ctx->realtime.is_realtime) {
        ctx->realtime.xruns++;
    }
}

void spectral_debug_xrun(SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    ctx->realtime.xruns++;
    ctx->realtime.is_realtime = false;
}

float spectral_debug_headroom(const SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_VAL_IF(!ctx, 100.0f);
    return ctx->realtime.headroom_percent;
}

/* ITM Output */

static inline int itm_wait_ready(uint8_t port) {
    uint32_t stim_addr = spectral_itm_stim_addr(port);
    for (uint32_t spins = 0; spins < SPECTRAL_DEBUG_ITM_MAX_SPINS; spins++) {
        if (spectral_reg_read(stim_addr) & 1u) return 1;
    }
    return 0;
}

static inline void itm_send_u32(uint8_t port, uint32_t value) {
    if (spectral_reg_read(SPECTRAL_ITM_TER_ADDR) & (1u << port)) {
        if (!itm_wait_ready(port)) return;
        spectral_reg_write(spectral_itm_stim_addr(port), value);
    }
}

void spectral_debug_itm_u32(uint8_t port, uint32_t value) {
    itm_send_u32(port, value);
}

void spectral_debug_itm_report(const SpectralDebugCtx* ctx) {
    SPECTRAL_RETURN_IF(!ctx);
    
    /* Send timing */
    itm_send_u32(ITM_PORT_TIMING, ctx->timing.cycles_avg);
    itm_send_u32(ITM_PORT_TIMING, ctx->timing.cycles_peak);
    
    /* Send memory */
    itm_send_u32(ITM_PORT_MEMORY, ctx->memory.sram_used);
    itm_send_u32(ITM_PORT_MEMORY, ctx->memory.sdram_used);
    
    /* Send cache hit rates (as fixed-point percentage * 100) */
    itm_send_u32(ITM_PORT_CACHE, (uint32_t)(ctx->cache.icache_hit_rate * 100));
    itm_send_u32(ITM_PORT_CACHE, (uint32_t)(ctx->cache.dcache_hit_rate * 100));
    
    /* Send SDRAM utilization (as fixed-point percentage * 100) */
    itm_send_u32(ITM_PORT_SDRAM, (uint32_t)(ctx->sdram.utilization_percent * 100));
    
    /* Send real-time stats */
    itm_send_u32(ITM_PORT_REALTIME, (uint32_t)(ctx->realtime.cpu_load_percent * 100));
    itm_send_u32(ITM_PORT_REALTIME, ctx->realtime.xruns);
}

void spectral_debug_printf(const char* fmt, ...) {
    char buf[128];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    
    /* Send character-by-character to ITM port 31 */
    for (int i = 0; i < len && i < (int)sizeof(buf); i++) {
        if (spectral_reg_read(SPECTRAL_ITM_TER_ADDR) & (1u << ITM_PORT_PRINTF)) {
            if (!itm_wait_ready(ITM_PORT_PRINTF)) break;
            spectral_reg_write(spectral_itm_stim_addr(ITM_PORT_PRINTF), (uint8_t)buf[i]);
        }
    }
}

/* Accessors */

const SpectralTimingStats* spectral_debug_get_timing(const SpectralDebugCtx* ctx) {
    return ctx ? &ctx->timing : NULL;
}

const SpectralMemoryStats* spectral_debug_get_memory(const SpectralDebugCtx* ctx) {
    return ctx ? &ctx->memory : NULL;
}

const SpectralCacheStats* spectral_debug_get_cache(const SpectralDebugCtx* ctx) {
    return ctx ? &ctx->cache : NULL;
}

const SpectralSdramStats* spectral_debug_get_sdram(const SpectralDebugCtx* ctx) {
    return ctx ? &ctx->sdram : NULL;
}

const SpectralSdcardStats* spectral_debug_get_sdcard(const SpectralDebugCtx* ctx) {
    return ctx ? &ctx->sdcard : NULL;
}

const SpectralRealtimeStats* spectral_debug_get_realtime(const SpectralDebugCtx* ctx) {
    return ctx ? &ctx->realtime : NULL;
}

#endif /* SPECTRAL_DEBUG_ARM */
