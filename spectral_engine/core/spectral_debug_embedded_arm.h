/* spectral_debug_embedded_arm.h - ARM Cortex-M Debug Monitoring via DWT/ITM
 * 
 * Provides real-time performance profiling on ARM Cortex-M targets using:
 *   - DWT (Data Watchpoint and Trace): cycle-accurate timing, memory access counts
 *   - ITM (Instrumentation Trace Macrocell): printf-style debug output via SWO
 * 
 * Features:
 *   - Timing: cycle counts, min/max/avg, budget tracking, overrun detection
 *   - Memory: SRAM/SDRAM/stack usage tracking
 *   - Cache: hit/miss rates for I-cache and D-cache
 *   - SDRAM: bandwidth measurement
 *   - Real-time: xrun detection, deadline monitoring
 * 
 * Requires debug probe with SWO trace (ST-Link, J-Link, etc.)
 * Enable with: -DSPECTRAL_DEBUG_ARM=1 or -DSPECTRAL_DEBUG=1 on ARM targets
 */

#ifndef SPECTRAL_DEBUG_ARM_H
#define SPECTRAL_DEBUG_ARM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* Configuration */

/* LED timing (ms) for embedded status indicators */
#ifndef SPECTRAL_ERROR_BLINK_ON_MS
#define SPECTRAL_ERROR_BLINK_ON_MS      100
#endif
#ifndef SPECTRAL_ERROR_BLINK_OFF_MS
#define SPECTRAL_ERROR_BLINK_OFF_MS     100
#endif
#ifndef SPECTRAL_ERROR_BLINK_PAUSE_MS
#define SPECTRAL_ERROR_BLINK_PAUSE_MS   500
#endif
#ifndef SPECTRAL_LED_BLINK_PLAYING_MS
#define SPECTRAL_LED_BLINK_PLAYING_MS   250
#endif
#ifndef SPECTRAL_LED_BLINK_DONE_MS
#define SPECTRAL_LED_BLINK_DONE_MS      100
#endif

/* Enable debug monitoring on ARM targets when DEBUG is set */
#if defined(SPECTRAL_DEBUG_RESTRICTED) || defined(SPECTRAL_DEBUG)
    #if defined(__ARM_ARCH_7EM__) || defined(CORE_CM7) || defined(STM32H750xx)
        #define SPECTRAL_DEBUG_ARM 1
    #endif
#endif

/* ITM stimulus port assignments */
#define ITM_PORT_TIMING     0
#define ITM_PORT_MEMORY     1
#define ITM_PORT_CACHE      2
#define ITM_PORT_SDRAM      3
#define ITM_PORT_SDCARD     4
#define ITM_PORT_REALTIME   5
#define ITM_PORT_PRINTF     31

#if defined(SPECTRAL_DEBUG_ARM)

/* Data Structures */

typedef struct {
    uint32_t cycles_total;
    uint32_t cycles_peak;
    uint32_t cycles_min;
    uint32_t cycles_avg;
    uint32_t measurement_count;
    uint32_t budget_cycles;
    uint32_t overruns;
} SpectralTimingStats;

typedef struct {
    uint32_t sram_used;
    uint32_t sram_peak;
    uint32_t sram_total;
    uint32_t sdram_used;
    uint32_t sdram_peak;
    uint32_t sdram_total;
    uint32_t stack_used;
    uint32_t stack_peak;
    uint32_t stack_total;
} SpectralMemoryStats;

typedef struct {
    uint32_t icache_hits;
    uint32_t icache_misses;
    uint32_t dcache_hits;
    uint32_t dcache_misses;
    uint32_t dcache_writebacks;
    float    icache_hit_rate;
    float    dcache_hit_rate;
} SpectralCacheStats;

typedef struct {
    uint32_t bytes_read;
    uint32_t bytes_written;
    uint32_t read_cycles;
    uint32_t write_cycles;
    float    read_bandwidth_mbps;
    float    write_bandwidth_mbps;
    float    utilization_percent;
} SpectralSdramStats;

typedef struct {
    uint32_t sectors_read;
    uint32_t sectors_written;
    uint32_t read_latency_us;
    uint32_t write_latency_us;
    uint32_t errors;
    bool     card_present;
} SpectralSdcardStats;

typedef struct {
    uint32_t sample_rate;
    uint32_t block_size;
    uint32_t deadline_cycles;
    uint32_t actual_cycles;
    int32_t  slack_cycles;
    float    cpu_load_percent;
    float    headroom_percent;
    uint32_t xruns;
    bool     is_realtime;
} SpectralRealtimeStats;

typedef struct {
    SpectralTimingStats   timing;
    SpectralMemoryStats   memory;
    SpectralCacheStats    cache;
    SpectralSdramStats    sdram;
    SpectralSdcardStats   sdcard;
    SpectralRealtimeStats realtime;
    uint32_t dwt_start;
    uint32_t last_update_ms;
    bool     initialized;
} SpectralDebugCtx;

/* API Functions */

void spectral_debug_init(SpectralDebugCtx* ctx, uint32_t sample_rate, uint32_t block_size);

void spectral_debug_deinit(SpectralDebugCtx* ctx);
void spectral_debug_reset(SpectralDebugCtx* ctx);

void spectral_debug_timing_start(SpectralDebugCtx* ctx);
uint32_t spectral_debug_timing_end(SpectralDebugCtx* ctx);
bool spectral_debug_is_realtime(const SpectralDebugCtx* ctx);
float spectral_debug_cpu_load(const SpectralDebugCtx* ctx);

void spectral_debug_update_memory(SpectralDebugCtx* ctx);
void spectral_debug_sdram_alloc(SpectralDebugCtx* ctx, uint32_t bytes);
void spectral_debug_sdram_free(SpectralDebugCtx* ctx, uint32_t bytes);
void spectral_debug_sram_alloc(SpectralDebugCtx* ctx, uint32_t bytes);
void spectral_debug_sram_free(SpectralDebugCtx* ctx, uint32_t bytes);
void spectral_debug_update_stack(SpectralDebugCtx* ctx);

void spectral_debug_update_cache(SpectralDebugCtx* ctx);
void spectral_debug_cache_invalidate(SpectralDebugCtx* ctx);

void spectral_debug_sdram_start(SpectralDebugCtx* ctx);
void spectral_debug_sdram_read(SpectralDebugCtx* ctx, uint32_t bytes);
void spectral_debug_sdram_write(SpectralDebugCtx* ctx, uint32_t bytes);

void spectral_debug_sdram_update(SpectralDebugCtx* ctx);

void spectral_debug_sdcard_read(SpectralDebugCtx* ctx, uint32_t sectors, uint32_t latency_us);
void spectral_debug_sdcard_write(SpectralDebugCtx* ctx, uint32_t sectors, uint32_t latency_us);
void spectral_debug_sdcard_error(SpectralDebugCtx* ctx);
void spectral_debug_sdcard_detect(SpectralDebugCtx* ctx, bool present);

void spectral_debug_block_complete(SpectralDebugCtx* ctx, uint32_t cycles);
void spectral_debug_xrun(SpectralDebugCtx* ctx);
float spectral_debug_headroom(const SpectralDebugCtx* ctx);

void spectral_debug_itm_report(const SpectralDebugCtx* ctx);
void spectral_debug_itm_u32(uint8_t port, uint32_t value);
void spectral_debug_printf(const char* fmt, ...);

const SpectralTimingStats*   spectral_debug_get_timing(const SpectralDebugCtx* ctx);
const SpectralMemoryStats*   spectral_debug_get_memory(const SpectralDebugCtx* ctx);
const SpectralCacheStats*    spectral_debug_get_cache(const SpectralDebugCtx* ctx);
const SpectralSdramStats*    spectral_debug_get_sdram(const SpectralDebugCtx* ctx);
const SpectralSdcardStats*   spectral_debug_get_sdcard(const SpectralDebugCtx* ctx);
const SpectralRealtimeStats* spectral_debug_get_realtime(const SpectralDebugCtx* ctx);

#else /* !SPECTRAL_DEBUG_ARM */

/* No-op stubs for release builds */

typedef struct { int _dummy; } SpectralDebugCtx;
typedef struct { int _dummy; } SpectralTimingStats;
typedef struct { int _dummy; } SpectralMemoryStats;
typedef struct { int _dummy; } SpectralCacheStats;
typedef struct { int _dummy; } SpectralSdramStats;
typedef struct { int _dummy; } SpectralSdcardStats;
typedef struct { int _dummy; } SpectralRealtimeStats;

/* All functions become no-ops in release builds */
#define spectral_debug_init(ctx, sr, bs)        ((void)0)
#define spectral_debug_deinit(ctx)              ((void)0)
#define spectral_debug_reset(ctx)               ((void)0)
#define spectral_debug_timing_start(ctx)        ((void)0)
#define spectral_debug_timing_end(ctx)          (0u)
#define spectral_debug_is_realtime(ctx)         (true)
#define spectral_debug_cpu_load(ctx)            (0.0f)
#define spectral_debug_update_memory(ctx)       ((void)0)
#define spectral_debug_sdram_alloc(ctx, b)      ((void)0)
#define spectral_debug_sdram_free(ctx, b)       ((void)0)
#define spectral_debug_sram_alloc(ctx, b)       ((void)0)
#define spectral_debug_sram_free(ctx, b)        ((void)0)
#define spectral_debug_update_stack(ctx)        ((void)0)
#define spectral_debug_update_cache(ctx)        ((void)0)
#define spectral_debug_cache_invalidate(ctx)    ((void)0)
#define spectral_debug_sdram_start(ctx)         ((void)0)
#define spectral_debug_sdram_read(ctx, b)       ((void)0)
#define spectral_debug_sdram_write(ctx, b)      ((void)0)
#define spectral_debug_sdram_update(ctx)        ((void)0)
#define spectral_debug_sdcard_read(c, s, l)     ((void)0)
#define spectral_debug_sdcard_write(c, s, l)    ((void)0)
#define spectral_debug_sdcard_error(ctx)        ((void)0)
#define spectral_debug_sdcard_detect(ctx, p)    ((void)0)
#define spectral_debug_block_complete(ctx, c)   ((void)0)
#define spectral_debug_xrun(ctx)                ((void)0)
#define spectral_debug_headroom(ctx)            (100.0f)
#define spectral_debug_itm_report(ctx)          ((void)0)
#define spectral_debug_itm_u32(p, v)            ((void)0)
#define spectral_debug_printf(...)              ((void)0)
#define spectral_debug_get_timing(ctx)          (NULL)
#define spectral_debug_get_memory(ctx)          (NULL)
#define spectral_debug_get_cache(ctx)           (NULL)
#define spectral_debug_get_sdram(ctx)           (NULL)
#define spectral_debug_get_sdcard(ctx)          (NULL)
#define spectral_debug_get_realtime(ctx)        (NULL)

#endif /* SPECTRAL_DEBUG_ARM */

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_DEBUG_ARM_H */
