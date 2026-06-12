/* spectral_perf_embedded.c - Embedded Target Workload Report + Memory Estimation
 *
 * Reports MEASURED workload counters and real-time budget arithmetic for the
 * embedded target, plus an exact memory-footprint calculation. Cycle
 * projections were retired with spectral_perf_model.*: they come from the
 * validated M7 measurement stack (m7-census / m7-stalls / m7-wcet), not
 * from in-C cost constants.
 *
 * All declarations are in spectral_perf.h (no new header needed).
 */

#include "spectral_perf.h"
#include "spectral_perf_accounting.h"
#include "spectral_utils.h"
#include "spectral_console.h"
#include "spectral_log.h"
#include "spectral_q15.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/*
 * ARM Cortex-M7 (32-bit) type sizes for target memory estimation
 * These model the actual embedded target, not the host machine
 */
enum {
    ARM32_PTR_SIZE = 4u,
    ARM32_U32_SIZE = 4u,
    ARM32_U16_SIZE = 2u,
    ARM32_Q31_SIZE = 4u,
    ARM32_Q15_SIZE = 2u
};

/*
 * SpectralSegmentQ15 size - packed struct, platform-independent
 * Layout: start(4) + length(2) + freq_q88(2) + phase_q15(2) + amp_q15(2) + da_q15(2)
 */
static const size_t kSegmentQ15Size = sizeof(SpectralSegmentQ15);

/*
 * SpectralActiveSegQ15 size - aligned(4), fixed-size types only
 * Layout: phase_acc(4) + freq_inc(4) + seg_idx(4) + amp_current(2) + amp_delta(2)
 */
static const size_t kActiveSegQ15Size = sizeof(SpectralActiveSegQ15);

/* SpectralArm32Ctx base size for 32-bit ARM (excludes active[] array) */
enum {
    ARM32_CTX_BASE_SIZE =
        ARM32_PTR_SIZE +           /* segments */
        ARM32_U32_SIZE * 5u +      /* num_segments..next_seg_idx */
        ARM32_U16_SIZE * 2u +      /* num_active, peak_active */
        ARM32_PTR_SIZE +           /* osc_lut */
        ARM32_U32_SIZE +           /* sample_rate */
        ARM32_Q15_SIZE             /* amplitude_q15 */
};

static size_t spectral_size_mul_or_max(size_t a, size_t b) {
    size_t out = 0;
    return spectral_size_mul(a, b, &out) ? out : SIZE_MAX;
}

static size_t spectral_size_add_or_max(size_t a, size_t b) {
    size_t out = 0;
    return spectral_size_add(a, b, &out) ? out : SIZE_MAX;
}

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
    mem.bytes_per_segment = kSegmentQ15Size;
    mem.active_seg_size = kActiveSegQ15Size;

    /* Static allocations (always present) */
    size_t osc_lut_entries = 0;
    if (osc_lut_bits >= (uint32_t)(sizeof(size_t) * 8u - 1u)) {
        osc_lut_entries = SIZE_MAX;
    } else {
        osc_lut_entries = ((size_t)1u << osc_lut_bits) + 1u;  /* +1 for interpolation wraparound */
    }
    mem.osc_lut_bytes = spectral_size_mul_or_max(osc_lut_entries, ARM32_Q15_SIZE);
    mem.active_array_bytes = spectral_size_mul_or_max((size_t)max_active, kActiveSegQ15Size);
    mem.ctx_bytes = spectral_size_add_or_max(ARM32_CTX_BASE_SIZE, mem.active_array_bytes);
    mem.static_total = spectral_size_add_or_max(mem.ctx_bytes, mem.osc_lut_bytes);

    /* Dynamic allocations (per audio file) */
    mem.segment_data_bytes = spectral_size_mul_or_max(num_segments, kSegmentQ15Size);

    /* Transient/stack (during processing) */
    mem.accum_buffer_bytes = spectral_size_mul_or_max((size_t)block_size, ARM32_Q31_SIZE);

    /* Totals */
    mem.total_static = mem.static_total;
    mem.total_dynamic = mem.segment_data_bytes;
    mem.total_transient = mem.accum_buffer_bytes;
    mem.total_bytes = spectral_size_add_or_max(
        spectral_size_add_or_max(mem.total_static, mem.total_dynamic), mem.total_transient);
    mem.total_kb = (mem.total_bytes + (SPECTRAL_BYTES_PER_KIB - 1u)) / SPECTRAL_BYTES_PER_KIB;

    /* Constraint check */
    size_t target_bytes = 0;
    if (!spectral_size_mul((size_t)target_kb, SPECTRAL_BYTES_PER_KIB, &target_bytes)) {
        target_bytes = SIZE_MAX;
    }
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
    snprintf(buf2, sizeof(buf2), "(%.1f KB)", BYTES_TO_KB(mem->total_static));
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

    {
        size_t target_bytes = 0;
        (void)spectral_size_mul((size_t)mem->target_max_kb, SPECTRAL_BYTES_PER_KIB, &target_bytes);
        snprintf(buf1, sizeof(buf1), "%.1f MB", BYTES_TO_MB(target_bytes));
    }
    snprintf(buf2, sizeof(buf2), "SRAM+SDRAM");
    table_print_row(&cfg, "Target memory", buf1, buf2);

    snprintf(buf1, sizeof(buf1), "%.1f%%", mem->usage_percent);
    table_print_row(&cfg, "Usage", buf1, "");

    table_print_footer(&cfg);

    /* Status message with memory hierarchy context */
    size_t static_hot_bytes = spectral_size_add_or_max(
        spectral_size_add_or_max(mem->ctx_bytes, mem->osc_lut_bytes), mem->accum_buffer_bytes);

    if (!mem->fits_in_target) {
        size_t target_bytes = 0;
        (void)spectral_size_mul((size_t)mem->target_max_kb, SPECTRAL_BYTES_PER_KIB, &target_bytes);
        size_t overage = mem->total_bytes - target_bytes;
        status_print(STATUS_WARN, "Memory exceeds target by %.1f MB", BYTES_TO_MB(overage));
        SPECTRAL_LOG_INFO("   Reduce segments (%zu) or audio duration.", mem->num_segments);
    } else if (mem->usage_percent > 80.0) {
        size_t target_bytes = 0;
        size_t headroom = 0;
        if (spectral_size_mul((size_t)mem->target_max_kb, SPECTRAL_BYTES_PER_KIB, &target_bytes) &&
            target_bytes > mem->total_bytes) {
            headroom = target_bytes - mem->total_bytes;
        }
        status_print(STATUS_WARN, "Memory usage high (%.1f%%). Headroom: %.1f MB",
                    mem->usage_percent, BYTES_TO_MB(headroom));
    } else {
        size_t target_bytes = 0;
        size_t headroom = 0;
        if (spectral_size_mul((size_t)mem->target_max_kb, SPECTRAL_BYTES_PER_KIB, &target_bytes) &&
            target_bytes > mem->total_bytes) {
            headroom = target_bytes - mem->total_bytes;
        }
        status_print(STATUS_OK, "Memory fits. Headroom: %.1f MB", BYTES_TO_MB(headroom));
    }

    /* Memory placement guidance */
    SPECTRAL_LOG_INFO("   Hot path (SRAM): ctx+LUT+accum = %.1f KB", BYTES_TO_KB(static_hot_bytes));
    SPECTRAL_LOG_INFO("   Cold data (SDRAM): segments = %.1f MB", BYTES_TO_MB(mem->segment_data_bytes));
    SPECTRAL_LOG_INFO("");
}

/* Embedded target workload report */

EmbeddedTargetConfig embedded_perf_default_config(void) {
    EmbeddedTargetConfig cfg = {
        .cpu_freq_mhz = SPECTRAL_EMBEDDED_DEFAULT_CPU_MHZ,
        .sample_rate = SPECTRAL_EMBEDDED_DEFAULT_SAMPLE_RATE,
        .block_size = SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE,
        .max_memory_kb = SPECTRAL_EMBEDDED_DEFAULT_MEMORY_KB,
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
    EmbeddedTargetConfig fallback_cfg = embedded_perf_default_config();
    const EmbeddedTargetConfig* cfg = config ? config : &fallback_cfg;

    est.desktop_time_ms = desktop_time_sec * 1000.0;
    est.segment_count = segment_count;
    est.peak_active = peak_active;
    est.output_samples = output_samples;

    if (!ops || output_samples == 0 || cfg->sample_rate == 0 ||
        cfg->cpu_freq_mhz == 0 || cfg->block_size == 0) {
        return est;
    }

    est.voice_samples = ops->voice_samples;
    est.seg_scan_checks = ops->seg_scan_checks;
    est.segment_activations = ops->sdram_accesses;
    est.accum_rw_words = ops->accum_rw_words;
    est.peak_block_cycles = ops->peak_block_cycles;     /* measured (DWT) or 0 */
    est.peak_block_active = ops->peak_block_active;

    {
        double cpu_freq_hz = cfg->cpu_freq_mhz * 1000000.0;
        est.cycles_per_sample_budget = cpu_freq_hz / cfg->sample_rate;
        est.cycles_per_block_budget = est.cycles_per_sample_budget * cfg->block_size;
        if (peak_active > 0) {
            est.budget_per_voice_sample =
                est.cycles_per_sample_budget / (double)peak_active;
        }
    }

    return est;
}

void embedded_perf_print(const EmbeddedTargetConfig* config,
                         const EmbeddedPerfEstimate* est) {
    EmbeddedTargetConfig fallback_cfg = embedded_perf_default_config();
    const EmbeddedTargetConfig* cfg = config ? config : &fallback_cfg;
    if (!cfg->verbose || !est) return;

    char buf1[48], buf2[48];

    TableColumn cols[] = {
        {"Metric", 24, ALIGN_LEFT},
        {"Value", 16, ALIGN_RIGHT},
        {"Unit", 20, ALIGN_LEFT}
    };
    TableConfig table_main = {
        .columns = cols,
        .num_columns = 3,
        .border = 1,
        .title = "EMBEDDED TARGET WORKLOAD"
    };

    table_print_header(&table_main);

    snprintf(buf1, sizeof(buf1), "%.1f", est->desktop_time_ms);
    table_print_row(&table_main, "Desktop execution", buf1, "ms");

    snprintf(buf1, sizeof(buf1), "%zu", est->segment_count);
    table_print_row(&table_main, "Segments", buf1, "");

    snprintf(buf1, sizeof(buf1), "%u", est->peak_active);
    table_print_row(&table_main, "Peak active", buf1, "");

    snprintf(buf1, sizeof(buf1), "%zu", est->output_samples);
    table_print_row(&table_main, "Output samples", buf1, "");

    table_print_separator(&table_main);

    snprintf(buf1, sizeof(buf1), "%llu", (unsigned long long)est->voice_samples);
    table_print_row(&table_main, "Voice-samples", buf1, "measured");

    snprintf(buf1, sizeof(buf1), "%llu", (unsigned long long)est->seg_scan_checks);
    table_print_row(&table_main, "Segment scan checks", buf1, "measured");

    snprintf(buf1, sizeof(buf1), "%llu", (unsigned long long)est->segment_activations);
    table_print_row(&table_main, "Segment activations", buf1, "measured (SDRAM)");

    snprintf(buf1, sizeof(buf1), "%llu", (unsigned long long)est->accum_rw_words);
    table_print_row(&table_main, "Accum RMW words", buf1, "measured (q31)");

    table_print_separator(&table_main);

    snprintf(buf1, sizeof(buf1), "%.0f", est->cycles_per_sample_budget);
    snprintf(buf2, sizeof(buf2), "@ %u kHz", cfg->sample_rate / 1000);
    table_print_row(&table_main, "Budget cyc/sample", buf1, buf2);

    snprintf(buf1, sizeof(buf1), "%.0f", est->cycles_per_block_budget);
    snprintf(buf2, sizeof(buf2), "%u-sample block", cfg->block_size);
    table_print_row(&table_main, "Budget cyc/block", buf1, buf2);

    if (est->budget_per_voice_sample > 0.0) {
        snprintf(buf1, sizeof(buf1), "%.1f", est->budget_per_voice_sample);
        snprintf(buf2, sizeof(buf2), "at peak %u active", est->peak_active);
        table_print_row(&table_main, "Budget cyc/voice-smp", buf1, buf2);
    }

    if (est->peak_block_cycles > 0) {
        snprintf(buf1, sizeof(buf1), "%llu", (unsigned long long)est->peak_block_cycles);
        snprintf(buf2, sizeof(buf2), "DWT, %u active", est->peak_block_active);
        table_print_row(&table_main, "Peak block cycles", buf1, buf2);
    }

    snprintf(buf1, sizeof(buf1), "M7 @ %u MHz", cfg->cpu_freq_mhz);
    snprintf(buf2, sizeof(buf2), "%u KB SRAM", cfg->max_memory_kb);
    table_print_row(&table_main, "Target", buf1, buf2);

    table_print_footer(&table_main);

    /* Real-time feasibility: with no on-device measurement, compare the
     * budget against the VALIDATED modeled per-voice-sample cost from the
     * M7 stack rather than fabricating one here. */
    if (est->peak_block_cycles > 0 && est->cycles_per_block_budget > 0.0) {
        double load = 100.0 * (double)est->peak_block_cycles / est->cycles_per_block_budget;
        if (load > 100.0) {
            status_print(STATUS_WARN, "Measured worst block exceeds real-time (%.1f%%).", load);
        } else {
            status_print(STATUS_OK, "Measured worst block within budget (%.1f%%).", load);
        }
    } else {
        SPECTRAL_LOG_INFO("   Cycle projections: python -m spectral_tools.testing."
                          "benchmark_workflow m7-census | m7-stalls | m7-wcet "
                          "[modeled, validated]");
    }
    SPECTRAL_LOG_INFO("");
}
