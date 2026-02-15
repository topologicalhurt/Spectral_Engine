/* spectral_perf_embedded.c - Embedded Target Performance and Memory Estimation
 *
 * Models Cortex-M7 cycle costs for Q15 operations, memory footprint
 * calculation, and real-time feasibility assessment.
 *
 * All declarations are in spectral_perf.h (no new header needed).
 */

#include "spectral_perf.h"
#include "spectral_utils.h"
#include "spectral_q15.h"
#include <stdio.h>
#include <stdint.h>

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
    ARM32_Q15_SIZE          /* amplitude_q15 */             \
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
    snprintf(buf2, sizeof(buf2), "(%.1f MB)", BYTES_TO_MB(mem->total_bytes));
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
        status_print(STATUS_WARN, "Memory exceeds target by %.1f MB", BYTES_TO_MB(overage));
        printf("   Reduce segments (%zu) or audio duration.\n", mem->num_segments);
    } else if (mem->usage_percent > 80.0) {
        size_t headroom = (mem->target_max_kb * 1024) - mem->total_bytes;
        status_print(STATUS_WARN, "Memory usage high (%.1f%%). Headroom: %.1f MB",
                    mem->usage_percent, BYTES_TO_MB(headroom));
    } else {
        size_t headroom = (mem->target_max_kb * 1024) - mem->total_bytes;
        status_print(STATUS_OK, "Memory fits. Headroom: %.1f MB", BYTES_TO_MB(headroom));
    }

    /* Memory placement guidance */
    printf("   Hot path (SRAM): ctx+LUT+accum = %.1f KB\n", BYTES_TO_KB(static_hot_bytes));
    printf("   Cold data (SDRAM): segments = %.1f MB\n", BYTES_TO_MB(mem->segment_data_bytes));
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

    /* SDRAM latency for segment activations */
    cycles += ops->sdram_accesses * EMBEDDED_CYCLES_SDRAM_ACCESS;

    /* L1 cache miss penalty */
    cycles += ops->cache_misses_est * EMBEDDED_CYCLES_CACHE_MISS;

    /* Segment scan overhead */
    cycles += ops->seg_scan_checks * EMBEDDED_CYCLES_SEG_SCAN;

    /* Block processing overhead: segment activation/deactivation checks,
     * memset of accumulator, output scaling, and callback overhead */
    uint64_t num_blocks = (output_samples + config->block_size - 1) / config->block_size;
    uint64_t block_overhead = EMBEDDED_CYCLES_CALLBACK_OVERHEAD +
                              config->block_size * EMBEDDED_CYCLES_MEMSET_PER_SAMPLE +
                              config->block_size * EMBEDDED_CYCLES_OUTPUT_PER_SAMPLE +
                              peak_active * EMBEDDED_CYCLES_SEG_BOUNDS_CHECK;
    cycles += num_blocks * block_overhead;

    /* Segment activation overhead (once per segment) */
    cycles += segment_count * EMBEDDED_CYCLES_SEG_ACTIVATION;

    est.estimated_cycles = cycles;

    /* Calculate metrics */
    double cpu_freq_hz = config->cpu_freq_mhz * 1000000.0;
    est.cycles_per_sample = (double)est.estimated_cycles / output_samples;
    est.cycles_available = cpu_freq_hz / config->sample_rate;
    est.cpu_load_percent = (est.cycles_per_sample / est.cycles_available) * 100.0;

    /* Best/worst case estimates */
    /* Best case: assume 20% faster (good cache, compiler opts) */
    est.cpu_load_best = est.cpu_load_percent * 0.80;
    /* Worst case: from peak block if available, else 30% overhead */
    if (ops->peak_block_cycles > 0) {
        double peak_cycles_per_sample = (double)ops->peak_block_cycles / config->block_size;
        est.cpu_load_worst = (peak_cycles_per_sample / est.cycles_available) * 100.0;
    } else {
        est.cpu_load_worst = est.cpu_load_percent * 1.30;
    }
    est.peak_block_cycles = ops->peak_block_cycles;
    est.peak_block_active = ops->peak_block_active;

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
    table_print_row(&cfg, "CPU load (typical)", buf1, "overall avg");

    snprintf(buf1, sizeof(buf1), "%.1f%%", est->cpu_load_best);
    table_print_row(&cfg, "CPU load (best)", buf1, "optimistic");

    snprintf(buf1, sizeof(buf1), "%.1f%%", est->cpu_load_worst);
    if (est->peak_block_active > 0) {
        snprintf(buf2, sizeof(buf2), "%u active", est->peak_block_active);
    } else {
        snprintf(buf2, sizeof(buf2), "pessimistic");
    }
    table_print_row(&cfg, "CPU load (worst)", buf1, buf2);

    snprintf(buf1, sizeof(buf1), "%.2fx", est->realtime_ratio);
    table_print_row(&cfg, "Real-time ratio", buf1, "");

    snprintf(buf1, sizeof(buf1), "M7 @ %u MHz", config->cpu_freq_mhz);
    snprintf(buf2, sizeof(buf2), "%u KB SRAM", config->max_memory_kb);
    table_print_row(&cfg, "Target", buf1, buf2);

    table_print_footer(&cfg);

    /* Status message */
    if (est->cpu_load_worst > 100.0 && est->peak_block_active > 0) {
        status_print(STATUS_WARN, "Worst-case block exceeds real-time (%u active, %.1f%% load).",
                    est->peak_block_active, est->cpu_load_worst);
        printf("   Peak block may cause audio glitches.\n");
    }
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
