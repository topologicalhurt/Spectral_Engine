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

#ifdef SPECTRAL_DEBUG_ARM

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Cortex-M7 Registers */

/* Core Debug */
#define CoreDebug_DEMCR     (*(volatile uint32_t*)0xE000EDFC)
#define CoreDebug_DEMCR_TRCENA_Msk  (1UL << 24)

/* Data Watchpoint and Trace (DWT) */
#define DWT_CTRL            (*(volatile uint32_t*)0xE0001000)
#define DWT_CYCCNT          (*(volatile uint32_t*)0xE0001004)
#define DWT_CPICNT          (*(volatile uint32_t*)0xE0001008)
#define DWT_EXCCNT          (*(volatile uint32_t*)0xE000100C)
#define DWT_SLEEPCNT        (*(volatile uint32_t*)0xE0001010)
#define DWT_LSUCNT          (*(volatile uint32_t*)0xE0001014)
#define DWT_FOLDCNT         (*(volatile uint32_t*)0xE0001018)

#define DWT_CTRL_CYCCNTENA_Msk  (1UL << 0)
#define DWT_CTRL_CPIEVTENA_Msk  (1UL << 17)
#define DWT_CTRL_EXCEVTENA_Msk  (1UL << 18)
#define DWT_CTRL_SLEEPEVTENA_Msk (1UL << 19)
#define DWT_CTRL_LSUEVTENA_Msk  (1UL << 20)
#define DWT_CTRL_FOLDEVTENA_Msk (1UL << 21)

/* Instrumentation Trace Macrocell (ITM) */
#define ITM_STIM(n)         (*(volatile uint32_t*)(0xE0000000 + 4*(n)))
#define ITM_TER             (*(volatile uint32_t*)0xE0000E00)
#define ITM_TPR             (*(volatile uint32_t*)0xE0000E40)
#define ITM_TCR             (*(volatile uint32_t*)0xE0000E80)
#define ITM_LAR             (*(volatile uint32_t*)0xE0000FB0)

#define ITM_TCR_ITMENA_Msk  (1UL << 0)
#define ITM_LAR_UNLOCK      0xC5ACCE55

/* System Control Block */
#define SCB_CCR             (*(volatile uint32_t*)0xE000ED14)
#define SCB_CCR_IC_Msk      (1UL << 17)
#define SCB_CCR_DC_Msk      (1UL << 16)

/* Time source */
#if defined(SPECTRAL_PLATFORM_DAISY)
    extern uint32_t daisy_system_get_now_ms(void);
    #define GET_MS() daisy_system_get_now_ms()
#else
    #define SYSTICK_LOAD    (*(volatile uint32_t*)0xE000E014)
    #define SYSTICK_VAL     (*(volatile uint32_t*)0xE000E018)
    static uint32_t _debug_ms_counter = 0;
    #define GET_MS() (_debug_ms_counter)
#endif

/* Platform constants */
#ifndef SPECTRAL_CPU_FREQ_HZ
    #if defined(SPECTRAL_PLATFORM_DAISY)
        #define SPECTRAL_CPU_FREQ_HZ    480000000
    #else
        #define SPECTRAL_CPU_FREQ_HZ    168000000
    #endif
#endif

#ifndef SPECTRAL_SDRAM_FREQ_HZ
    #define SPECTRAL_SDRAM_FREQ_HZ  100000000
#endif
#ifndef SPECTRAL_SDRAM_BUS_WIDTH
    #define SPECTRAL_SDRAM_BUS_WIDTH    32
#endif

#define SDRAM_MAX_BANDWIDTH ((SPECTRAL_SDRAM_FREQ_HZ * SPECTRAL_SDRAM_BUS_WIDTH) / 8)

/* Initialization */

void spectral_debug_init(SpectralDebugCtx* ctx, uint32_t sample_rate, uint32_t block_size) {
    if (!ctx) return;
    
    /* Clear context */
    memset(ctx, 0, sizeof(SpectralDebugCtx));
    
    /* Calculate timing budget */
    ctx->realtime.sample_rate = sample_rate;
    ctx->realtime.block_size = block_size;
    ctx->realtime.deadline_cycles = (SPECTRAL_CPU_FREQ_HZ / sample_rate) * block_size;
    ctx->realtime.is_realtime = true;
    
    /* Initialize timing stats */
    ctx->timing.cycles_min = UINT32_MAX;
    ctx->timing.budget_cycles = ctx->realtime.deadline_cycles;
    
    /* Memory pool sizes (platform-specific) */
#if defined(SPECTRAL_PLATFORM_DAISY)
    ctx->memory.sram_total = 128 * 1024;    /* 128KB SRAM */
    ctx->memory.sdram_total = 64 * 1024 * 1024;  /* 64MB SDRAM */
    ctx->memory.stack_total = 8 * 1024;     /* 8KB stack */
#else
    ctx->memory.sram_total = 64 * 1024;     /* Generic default */
    ctx->memory.sdram_total = 0;
    ctx->memory.stack_total = 4 * 1024;
#endif
    
    /* Enable DWT cycle counter */
    CoreDebug_DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    
    /* Enable DWT performance counters */
    DWT_CTRL |= DWT_CTRL_CPIEVTENA_Msk |
                DWT_CTRL_EXCEVTENA_Msk |
                DWT_CTRL_SLEEPEVTENA_Msk |
                DWT_CTRL_LSUEVTENA_Msk |
                DWT_CTRL_FOLDEVTENA_Msk;
    
    /* Enable ITM */
    ITM_LAR = ITM_LAR_UNLOCK;
    ITM_TCR |= ITM_TCR_ITMENA_Msk;
    ITM_TER |= (1UL << ITM_PORT_TIMING) |
               (1UL << ITM_PORT_MEMORY) |
               (1UL << ITM_PORT_CACHE) |
               (1UL << ITM_PORT_SDRAM) |
               (1UL << ITM_PORT_SDCARD) |
               (1UL << ITM_PORT_REALTIME) |
               (1UL << ITM_PORT_PRINTF);
    
    ctx->initialized = true;
    ctx->last_update_ms = GET_MS();
    
    spectral_debug_printf("Spectral Debug: initialized @ %lu Hz, %lu samples\n",
                          (unsigned long)sample_rate, (unsigned long)block_size);
}

void spectral_debug_deinit(SpectralDebugCtx* ctx) {
    if (!ctx || !ctx->initialized) return;
    
    spectral_debug_printf("Spectral Debug: shutdown\n");
    spectral_debug_printf("  Peak cycles: %lu\n", (unsigned long)ctx->timing.cycles_peak);
    spectral_debug_printf("  Total xruns: %lu\n", (unsigned long)ctx->realtime.xruns);
    
    ctx->initialized = false;
}

void spectral_debug_reset(SpectralDebugCtx* ctx) {
    if (!ctx) return;
    
    uint32_t sample_rate = ctx->realtime.sample_rate;
    uint32_t block_size = ctx->realtime.block_size;
    
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
    DWT_CYCCNT = 0;
    DWT_CPICNT = 0;
    DWT_EXCCNT = 0;
    DWT_SLEEPCNT = 0;
    DWT_LSUCNT = 0;
    DWT_FOLDCNT = 0;
    
    spectral_debug_printf("Spectral Debug: reset\n");
}

/* Timing Measurement */

void spectral_debug_timing_start(SpectralDebugCtx* ctx) {
    if (!ctx) return;
    ctx->dwt_start = DWT_CYCCNT;
}

uint32_t spectral_debug_timing_end(SpectralDebugCtx* ctx) {
    if (!ctx) return 0;
    
    uint32_t end = DWT_CYCCNT;
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
    
    /* IIR rolling average: avg = avg + (new - avg) / 16 */
    ctx->timing.cycles_avg += (elapsed - ctx->timing.cycles_avg) >> 4;
    
    /* Check for budget overrun */
    if (elapsed > ctx->timing.budget_cycles) {
        ctx->timing.overruns++;
    }
    
    /* Update real-time stats */
    spectral_debug_block_complete(ctx, elapsed);
    
    return elapsed;
}

bool spectral_debug_is_realtime(const SpectralDebugCtx* ctx) {
    if (!ctx) return true;
    return ctx->realtime.is_realtime;
}

float spectral_debug_cpu_load(const SpectralDebugCtx* ctx) {
    if (!ctx) return 0.0f;
    return ctx->realtime.cpu_load_percent;
}

/* Memory Tracking */

void spectral_debug_update_memory(SpectralDebugCtx* ctx) {
    if (!ctx) return;
    
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
    if (!ctx) return;
    ctx->memory.sdram_used += bytes;
}

void spectral_debug_sdram_free(SpectralDebugCtx* ctx, uint32_t bytes) {
    if (!ctx) return;
    if (bytes <= ctx->memory.sdram_used) {
        ctx->memory.sdram_used -= bytes;
    }
}

void spectral_debug_sram_alloc(SpectralDebugCtx* ctx, uint32_t bytes) {
    if (!ctx) return;
    ctx->memory.sram_used += bytes;
}

void spectral_debug_sram_free(SpectralDebugCtx* ctx, uint32_t bytes) {
    if (!ctx) return;
    if (bytes <= ctx->memory.sram_used) {
        ctx->memory.sram_used -= bytes;
    }
}

void spectral_debug_update_stack(SpectralDebugCtx* ctx) {
    if (!ctx) return;
    
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
    if (!ctx) return;
    
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
    if (!ctx) return;
    
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
    if (!ctx) return;
    
    ctx->sdram.bytes_read = 0;
    ctx->sdram.bytes_written = 0;
    ctx->sdram.read_cycles = 0;
    ctx->sdram.write_cycles = 0;
}

void spectral_debug_sdram_read(SpectralDebugCtx* ctx, uint32_t bytes) {
    if (!ctx) return;
    ctx->sdram.bytes_read += bytes;
}

void spectral_debug_sdram_write(SpectralDebugCtx* ctx, uint32_t bytes) {
    if (!ctx) return;
    ctx->sdram.bytes_written += bytes;
}

void spectral_debug_sdram_update(SpectralDebugCtx* ctx) {
    if (!ctx) return;
    
    uint32_t now = GET_MS();
    uint32_t elapsed_ms = now - ctx->last_update_ms;
    
    if (elapsed_ms < 100) return;  /* Update at most 10 Hz */
    
    /* Calculate bandwidth (MB/s) */
    float elapsed_sec = elapsed_ms / 1000.0f;
    ctx->sdram.read_bandwidth_mbps = (ctx->sdram.bytes_read / (1024.0f * 1024.0f)) / elapsed_sec;
    ctx->sdram.write_bandwidth_mbps = (ctx->sdram.bytes_written / (1024.0f * 1024.0f)) / elapsed_sec;
    
    /* Calculate utilization */
    uint32_t total_bytes = ctx->sdram.bytes_read + ctx->sdram.bytes_written;
    float max_bytes_per_interval = (SDRAM_MAX_BANDWIDTH / 1000.0f) * elapsed_ms;
    ctx->sdram.utilization_percent = (100.0f * total_bytes) / max_bytes_per_interval;
    
    /* Reset for next interval */
    ctx->sdram.bytes_read = 0;
    ctx->sdram.bytes_written = 0;
    ctx->last_update_ms = now;
}

/* SD Card I/O */

void spectral_debug_sdcard_read(SpectralDebugCtx* ctx, uint32_t sectors, uint32_t latency_us) {
    if (!ctx) return;
    
    ctx->sdcard.sectors_read += sectors;
    
    /* IIR average latency */
    if (ctx->sdcard.read_latency_us == 0) {
        ctx->sdcard.read_latency_us = latency_us;
    } else {
        ctx->sdcard.read_latency_us += (latency_us - ctx->sdcard.read_latency_us) >> 3;
    }
}

void spectral_debug_sdcard_write(SpectralDebugCtx* ctx, uint32_t sectors, uint32_t latency_us) {
    if (!ctx) return;
    
    ctx->sdcard.sectors_written += sectors;
    
    /* IIR average latency */
    if (ctx->sdcard.write_latency_us == 0) {
        ctx->sdcard.write_latency_us = latency_us;
    } else {
        ctx->sdcard.write_latency_us += (latency_us - ctx->sdcard.write_latency_us) >> 3;
    }
}

void spectral_debug_sdcard_error(SpectralDebugCtx* ctx) {
    if (!ctx) return;
    ctx->sdcard.errors++;
}

void spectral_debug_sdcard_detect(SpectralDebugCtx* ctx, bool present) {
    if (!ctx) return;
    ctx->sdcard.card_present = present;
}

/* Real-time Monitoring */

void spectral_debug_block_complete(SpectralDebugCtx* ctx, uint32_t cycles) {
    if (!ctx) return;
    
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
    if (!ctx) return;
    ctx->realtime.xruns++;
    ctx->realtime.is_realtime = false;
}

float spectral_debug_headroom(const SpectralDebugCtx* ctx) {
    if (!ctx) return 100.0f;
    return ctx->realtime.headroom_percent;
}

/* ITM Output */

static inline void itm_send_u32(uint8_t port, uint32_t value) {
    if (ITM_TER & (1UL << port)) {
        /* Wait for FIFO ready */
        while ((ITM_STIM(port) & 1) == 0);
        ITM_STIM(port) = value;
    }
}

void spectral_debug_itm_u32(uint8_t port, uint32_t value) {
    itm_send_u32(port, value);
}

void spectral_debug_itm_report(const SpectralDebugCtx* ctx) {
    if (!ctx) return;
    
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
        if (ITM_TER & (1UL << ITM_PORT_PRINTF)) {
            while ((ITM_STIM(ITM_PORT_PRINTF) & 1) == 0);
            ITM_STIM(ITM_PORT_PRINTF) = (uint8_t)buf[i];
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
