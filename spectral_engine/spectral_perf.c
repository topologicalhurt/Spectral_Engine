/* spectral_perf.c - Performance Metrics Implementation
 * 
 * Platform-specific memory and CPU time queries:
 *   macOS: mach_task_info(), getrusage(), sysctlbyname()
 *   Linux: /proc/self/statm, getrusage(), sysconf()
 * 
 * Allocation Tracking:
 *   perf_track_alloc/free maintain running total and peak
 *   Used to measure actual memory consumption vs theoretical
 * 
 * Embedded Estimation:
 *   Models Cortex-M7 cycle costs for Q15 operations
 *   Accounts for LUT lookups, MACs, phase updates, loop overhead
 */
#include "spectral_perf.h"
#include "spectral_utils.h"
#include "spectral_q15.h"
#include <stdio.h>
#include <omp.h>
#include <stdint.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

/* Global allocation tracking state */
size_t g_peak_alloc = 0;
size_t g_current_alloc = 0;

#ifdef __APPLE__

void perf_get_memory(size_t* resident_kb, size_t* virtual_kb) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        *resident_kb = info.resident_size / 1024;
        *virtual_kb = info.virtual_size / 1024;
    } else {
        *resident_kb = *virtual_kb = 0;
    }
}

void perf_get_cpu_time(double* user_ms, double* sys_ms) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        *user_ms = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
        *sys_ms = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    } else {
        *user_ms = *sys_ms = 0;
    }
}

int perf_get_num_cores(void) {
    int cores = 0;
    size_t len = sizeof(cores);
    sysctlbyname("hw.ncpu", &cores, &len, NULL, 0);
    return (cores > 0) ? cores : 1;
}

#else /* Linux */

void perf_get_memory(size_t* resident_kb, size_t* virtual_kb) {
    *resident_kb = *virtual_kb = 0;
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        size_t vm, rss;
        if (fscanf(f, "%zu %zu", &vm, &rss) == 2) {
            long page_size = sysconf(_SC_PAGESIZE);
            *virtual_kb = (vm * page_size) / 1024;
            *resident_kb = (rss * page_size) / 1024;
        }
        fclose(f);
    }
}

void perf_get_cpu_time(double* user_ms, double* sys_ms) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        *user_ms = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
        *sys_ms = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    } else {
        *user_ms = *sys_ms = 0;
    }
}

int perf_get_num_cores(void) {
    int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return (cores > 0) ? cores : 1;
}

#endif /* __APPLE__ */

void perf_track_alloc(size_t bytes) {
    #pragma omp atomic
    g_current_alloc += bytes;
    if (g_current_alloc > g_peak_alloc) {
        #pragma omp critical
        { if (g_current_alloc > g_peak_alloc) g_peak_alloc = g_current_alloc; }
    }
}

void perf_track_free(size_t bytes) {
    #pragma omp atomic
    g_current_alloc -= bytes;
}

void perf_reset_tracking(void) {
    g_peak_alloc = 0;
    g_current_alloc = 0;
}

PerfMetrics perf_snapshot(double wall_start) {
    PerfMetrics m = {0};
    size_t res_kb, virt_kb;
    perf_get_memory(&res_kb, &virt_kb);
    perf_get_cpu_time(&m.user_time_ms, &m.sys_time_ms);
    m.current_resident_mb = res_kb / 1024;
    m.virtual_mb = virt_kb / 1024;
    m.num_cores = perf_get_num_cores();
    m.wall_time_ms = (omp_get_wtime() - wall_start) * 1000.0;
    m.tracked_allocs = g_peak_alloc;
    m.peak_resident_mb = m.current_resident_mb;
    m.cpu_utilization = (m.wall_time_ms > 0) ? 
        100.0 * (m.user_time_ms + m.sys_time_ms) / (m.wall_time_ms * m.num_cores) : 0;
    return m;
}

void perf_print(PerfMetrics* start, PerfMetrics* end, int n_threads) {
    double user_delta = end->user_time_ms - start->user_time_ms;
    double sys_delta = end->sys_time_ms - start->sys_time_ms;
    double wall_delta = end->wall_time_ms - start->wall_time_ms;
    double total_cpu = user_delta + sys_delta;
    double utilization = (wall_delta > 0) ? 100.0 * total_cpu / (wall_delta * n_threads) : 0;
    double parallelism = (wall_delta > 0) ? (total_cpu / wall_delta) : 0.0;
    
    printf("\n--- Performance Metrics ---\n");
    printf("Memory:  RSS %zu MB, Peak tracked %.1f MB\n", 
           end->current_resident_mb, g_peak_alloc / (1024.0 * 1024.0));
    printf("CPU:     User %.1f ms, Sys %.1f ms, Total %.1f ms\n", 
           user_delta, sys_delta, total_cpu);
    printf("Threads: %d / %d cores, Util %.1f%%, Parallelism %.2fx\n",
        n_threads, end->num_cores, utilization, parallelism);
}

/*
 * ARM Cortex-M7 (32-bit) type sizes for target memory estimation
 * These model the actual embedded target, not the host machine
 */
#define ARM32_PTR_SIZE      4
#define ARM32_U32_SIZE      4
#define ARM32_U16_SIZE      2
#define ARM32_Q31_SIZE      4
#define ARM32_Q15_SIZE      2

/*
 * SpectralSegmentQ15 size - packed struct, platform-independent
 * Layout: start(4) + length(2) + freq_q88(2) + phase_q15(2) + amp_q15(2) + da_q15(2)
 */
#define SEGMENT_Q15_SIZE    sizeof(SpectralSegmentQ15)

/*
 * SpectralActiveSegQ15 size - aligned(4), fixed-size types only
 * Layout: phase_acc(4) + freq_inc(4) + seg_idx(4) + amp_current(2) + amp_delta(2)
 */
#define ACTIVE_SEG_Q15_SIZE sizeof(SpectralActiveSegQ15)

/* SpectralArm32Ctx base size for 32-bit ARM (excludes active[] array) */
#define ARM32_CTX_BASE_SIZE  (                              \
    ARM32_PTR_SIZE +        /* segments */                  \
    ARM32_U32_SIZE * 5 +    /* num_segments..next_seg_idx */\
    ARM32_U16_SIZE * 2 +    /* num_active, peak_active */   \
    ARM32_PTR_SIZE +        /* osc_lut */                   \
    ARM32_U32_SIZE +        /* sample_rate */               \
    ARM32_Q15_SIZE * 2      /* amplitude_q15, stretch_q214 */\
)

EmbeddedMemoryUsage embedded_memory_usage(
    size_t   num_segments,
    uint32_t block_size,
    uint32_t osc_lut_bits,
    uint32_t max_active,
    uint32_t target_kb
) {
    EmbeddedMemoryUsage mem = {0};
    
    /* Store parameters */
    mem.num_segments = num_segments;
    mem.block_size = block_size;
    mem.max_active = max_active;
    mem.target_max_kb = target_kb;
    mem.bytes_per_segment = SEGMENT_Q15_SIZE;
    mem.active_seg_size = ACTIVE_SEG_Q15_SIZE;
    
    /* Static allocations (always present) */
    size_t osc_lut_entries = (1 << osc_lut_bits) + 1;  /* +1 for interpolation wraparound */
    mem.osc_lut_bytes = osc_lut_entries * ARM32_Q15_SIZE;
    mem.active_array_bytes = max_active * ACTIVE_SEG_Q15_SIZE;
    mem.ctx_bytes = ARM32_CTX_BASE_SIZE + mem.active_array_bytes;
    mem.static_total = mem.ctx_bytes + mem.osc_lut_bytes;
    
    /* Dynamic allocations (per audio file) */
    mem.segment_data_bytes = num_segments * SEGMENT_Q15_SIZE;
    
    /* Transient/stack (during processing) */
    mem.accum_buffer_bytes = block_size * ARM32_Q31_SIZE;
    
    /* Totals */
    mem.total_static = mem.static_total;
    mem.total_dynamic = mem.segment_data_bytes;
    mem.total_transient = mem.accum_buffer_bytes;
    mem.total_bytes = mem.total_static + mem.total_dynamic + mem.total_transient;
    mem.total_kb = (mem.total_bytes + 1023) / 1024;
    
    /* Constraint check */
    size_t target_bytes = target_kb * 1024;
    if (target_bytes == 0) {
        mem.usage_percent = 0.0;
        mem.fits_in_target = 0;
    } else {
        mem.usage_percent = 100.0 * (double)mem.total_bytes / target_bytes;
        mem.fits_in_target = (mem.total_bytes <= target_bytes) ? 1 : 0;
    }
    
    return mem;
}

void embedded_memory_print(const EmbeddedMemoryUsage* mem) {
    char buf1[32], buf2[32];
    
    TableColumn cols[] = {
        {"Category", 28, ALIGN_LEFT},
        {"Size", 14, ALIGN_RIGHT},
        {"Details", 18, ALIGN_LEFT}
    };
    TableConfig cfg = {
        .columns = cols,
        .num_columns = 3,
        .border = 1,
        .title = "EMBEDDED MEMORY USAGE"
    };
    
    table_print_header(&cfg);
    
    /* Static section */
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->ctx_bytes);
    table_print_row(&cfg, "Context structure", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->osc_lut_bytes);
    table_print_row(&cfg, "Oscillator LUT", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->active_array_bytes);
    snprintf(buf2, sizeof(buf2), "%u x %zu", mem->max_active, mem->active_seg_size);
    table_print_row(&cfg, "Active segments array", buf1, buf2);
    
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->total_static);
    snprintf(buf2, sizeof(buf2), "(%zu KB)", (mem->total_static + 1023) / 1024);
    table_print_row(&cfg, "Static subtotal", buf1, buf2);
    
    table_print_separator(&cfg);
    
    /* Dynamic section */
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->segment_data_bytes);
    snprintf(buf2, sizeof(buf2), "%zu x %zu", mem->num_segments, mem->bytes_per_segment);
    table_print_row(&cfg, "Segment data", buf1, buf2);
    
    table_print_separator(&cfg);
    
    /* Transient section */
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->accum_buffer_bytes);
    snprintf(buf2, sizeof(buf2), "%zu samp x 4", (size_t)mem->block_size);
    table_print_row(&cfg, "Accumulator buffer", buf1, buf2);
    
    table_print_separator(&cfg);
    
    /* Totals */
    snprintf(buf1, sizeof(buf1), "%zu bytes", mem->total_bytes);
    snprintf(buf2, sizeof(buf2), "(%.1f MB)", mem->total_bytes / (1024.0 * 1024.0));
    table_print_row(&cfg, "TOTAL", buf1, buf2);
    
    snprintf(buf1, sizeof(buf1), "%.1f MB", mem->target_max_kb / 1024.0);
    snprintf(buf2, sizeof(buf2), "SRAM+SDRAM");
    table_print_row(&cfg, "Target memory", buf1, buf2);
    
    snprintf(buf1, sizeof(buf1), "%.1f%%", mem->usage_percent);
    table_print_row(&cfg, "Usage", buf1, "");
    
    table_print_footer(&cfg);
    
    /* Status message with memory hierarchy context */
    size_t static_hot_bytes = mem->ctx_bytes + mem->osc_lut_bytes + mem->accum_buffer_bytes;
    
    if (!mem->fits_in_target) {
        size_t overage = mem->total_bytes - (mem->target_max_kb * 1024);
        status_print(STATUS_WARN, "Memory exceeds target by %.1f MB", overage / (1024.0 * 1024.0));
        printf("   Reduce segments (%zu) or audio duration.\n", mem->num_segments);
    } else if (mem->usage_percent > 80.0) {
        size_t headroom = (mem->target_max_kb * 1024) - mem->total_bytes;
        status_print(STATUS_WARN, "Memory usage high (%.1f%%). Headroom: %.1f MB",
                    mem->usage_percent, headroom / (1024.0 * 1024.0));
    } else {
        size_t headroom = (mem->target_max_kb * 1024) - mem->total_bytes;
        status_print(STATUS_OK, "Memory fits. Headroom: %.1f MB", headroom / (1024.0 * 1024.0));
    }
    
    /* Memory placement guidance */
    printf("   Hot path (SRAM): ctx+LUT+accum = %.1f KB\n", static_hot_bytes / 1024.0);
    printf("   Cold data (SDRAM): segments = %.1f MB\n", mem->segment_data_bytes / (1024.0 * 1024.0));
    printf("\n");
}

/* Embedded target performance estimation */

EmbeddedTargetConfig embedded_perf_default_config(void) {
    EmbeddedTargetConfig cfg = {
        .cpu_freq_mhz = EMBEDDED_DEFAULT_CPU_MHZ,
        .sample_rate = EMBEDDED_DEFAULT_SAMPLE_RATE,
        .block_size = EMBEDDED_DEFAULT_BLOCK_SIZE,
        .max_memory_kb = EMBEDDED_DEFAULT_MEMORY_KB,
        .verbose = 1
    };
    return cfg;
}

EmbeddedPerfEstimate embedded_perf_estimate(
    const EmbeddedTargetConfig* config,
    const EmbeddedOpCounts* ops,
    size_t   output_samples,
    size_t   segment_count,
    uint32_t peak_active,
    double   desktop_time_sec
) {
    EmbeddedPerfEstimate est = {0};
    
    est.desktop_time_ms = desktop_time_sec * 1000.0;
    est.segment_count = segment_count;
    est.peak_active = peak_active;
    est.output_samples = output_samples;
    
    /*
     * Calculate cycles from actual operation counts
     * These map directly to what spectral_synth_embedded.c does per sample:
     *   - LUT lookup includes shift, load, optional interpolation
     *   - MAC = amplitude scaling (sample * amp)
     *   - Phase update = accumulator advance
     *   - Loop iterations = overhead per 4 samples (unrolled)
     */
    uint64_t cycles = 0;
    cycles += ops->lut_lookups * EMBEDDED_CYCLES_LUT_LOOKUP;
    cycles += ops->mac_operations * (EMBEDDED_CYCLES_Q15_MUL + EMBEDDED_CYCLES_ACCUMULATE);
    cycles += ops->phase_updates * EMBEDDED_CYCLES_PHASE_UPDATE;
    cycles += ops->loop_iterations * EMBEDDED_CYCLES_LOOP_PER_4;
    
    /* Add amplitude delta updates (one per sample per active segment) */
    cycles += ops->lut_lookups * EMBEDDED_CYCLES_AMP_UPDATE;
    
    /* Block processing overhead: segment activation/deactivation checks,
     * memset of accumulator, output scaling, and callback overhead */
    uint64_t num_blocks = (output_samples + config->block_size - 1) / config->block_size;
    uint64_t block_overhead = 50 +                                /* callback entry/exit */
                              config->block_size * 2 +            /* memset accum buffer */
                              config->block_size * 3 +            /* final output scaling */
                              peak_active * 15;                   /* segment bounds checks */
    cycles += num_blocks * block_overhead;
    
    /* Segment activation overhead (once per segment) */
    cycles += segment_count * 30;  /* Phase/freq init, amplitude setup */
    
    est.estimated_cycles = cycles;
    
    /* Calculate metrics */
    double cpu_freq_hz = config->cpu_freq_mhz * 1000000.0;
    est.cycles_per_sample = (double)est.estimated_cycles / output_samples;
    est.cycles_available = cpu_freq_hz / config->sample_rate;
    est.cpu_load_percent = (est.cycles_per_sample / est.cycles_available) * 100.0;
    
    /* Estimated time on target */
    double target_time_sec = (double)est.estimated_cycles / cpu_freq_hz;
    est.target_time_ms = target_time_sec * 1000.0;
    
    /* Real-time ratio */
    double audio_duration_sec = (double)output_samples / config->sample_rate;
    est.realtime_ratio = audio_duration_sec / target_time_sec;
    
    return est;
}

void embedded_perf_print(const EmbeddedTargetConfig* config,
                         const EmbeddedPerfEstimate* est) {
    if (!config->verbose) return;
    
    char buf1[32], buf2[32];
    
    TableColumn cols[] = {
        {"Metric", 24, ALIGN_LEFT},
        {"Value", 16, ALIGN_RIGHT},
        {"Unit", 20, ALIGN_LEFT}
    };
    TableConfig cfg = {
        .columns = cols,
        .num_columns = 3,
        .border = 1,
        .title = "EMBEDDED TARGET PERFORMANCE"
    };
    
    table_print_header(&cfg);
    
    /* Input stats */
    snprintf(buf1, sizeof(buf1), "%.1f", est->desktop_time_ms);
    table_print_row(&cfg, "Desktop execution", buf1, "ms");
    
    snprintf(buf1, sizeof(buf1), "%zu", est->segment_count);
    table_print_row(&cfg, "Segments", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "%u", est->peak_active);
    table_print_row(&cfg, "Peak active", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "%zu", est->output_samples);
    table_print_row(&cfg, "Output samples", buf1, "");
    
    table_print_separator(&cfg);
    
    /* Target estimates */
    snprintf(buf1, sizeof(buf1), "%.1f", est->target_time_ms);
    table_print_row(&cfg, "Estimated target time", buf1, "ms");
    
    snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_per_sample);
    table_print_row(&cfg, "Cycles/sample", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "%.0f", est->cycles_available);
    snprintf(buf2, sizeof(buf2), "@ %u kHz", config->sample_rate / 1000);
    table_print_row(&cfg, "Cycles available", buf1, buf2);
    
    snprintf(buf1, sizeof(buf1), "%.1f%%", est->cpu_load_percent);
    table_print_row(&cfg, "CPU load", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "%.2fx", est->realtime_ratio);
    table_print_row(&cfg, "Real-time ratio", buf1, "");
    
    snprintf(buf1, sizeof(buf1), "M7 @ %u MHz", config->cpu_freq_mhz);
    snprintf(buf2, sizeof(buf2), "%u KB SRAM", config->max_memory_kb);
    table_print_row(&cfg, "Target", buf1, buf2);
    
    table_print_footer(&cfg);
    
    /* Status message */
    if (est->cpu_load_percent > 100.0) {
        status_print(STATUS_WARN, "CPU load exceeds 100%%. May not render in real-time.");
        printf("   Consider: fewer segments, shorter audio, or lower sample rate.\n");
    } else if (est->cpu_load_percent > 80.0) {
        status_print(STATUS_WARN, "CPU load high (%.1f%%). May have issues with complex sections.",
                    est->cpu_load_percent);
    } else {
        status_print(STATUS_OK, "Should run comfortably on target hardware.");
    }
    printf("\n");
}
