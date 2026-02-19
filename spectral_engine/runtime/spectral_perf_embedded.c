/* spectral_perf_embedded.c - Embedded Target Performance and Memory Estimation
 *
 * Models Cortex-M7 cycle costs for Q15 operations, memory footprint
 * calculation, and real-time feasibility assessment.
 *
 * All declarations are in spectral_perf.h (no new header needed).
 */

#include "spectral_perf.h"
#include "spectral_perf_accounting.h"
#include "spectral_perf_model.h"
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

/* Embedded target performance estimation */

static uint64_t scale_cycles_u64(uint64_t cycles, double factor) {
    if (!spectral_is_finite_positive_f64(factor)) return cycles;
    double scaled = (double)cycles * factor;
    if (scaled <= 0.0) return 0;
    if (scaled >= (double)UINT64_MAX) return UINT64_MAX;
    return (uint64_t)(scaled + 0.5);
}

static double clamp_confidence(double v) {
    if (!spectral_is_finite_f64(v)) return 0.0;
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

static const char* confidence_label(double v) {
    if (v >= 0.70) return "moderate";
    if (v >= 0.45) return "low-moderate";
    return "low";
}

static double estimate_model_confidence(const SpectralPerfModelProfile* profile,
                                        const SpectralPerfModelInput* in) {
    const uint32_t miss_threshold = profile ? profile->cache_miss_threshold_active : 24u;
    double conf = clamp_confidence(profile ? profile->confidence : 0.58);
    if (!in || in->output_samples == 0 || in->block_size == 0) return 0.0;

    if (!profile) conf *= 0.85;
    if (in->peak_active == 0) conf *= 0.75;
    if (in->peak_block_cycles_observed == 0) conf *= 0.80;
    if (in->cache_misses_est == 0 && in->peak_active > miss_threshold) conf *= 0.82;
    if (in->segment_count < 16) conf *= 0.85;
    if (in->output_samples < (size_t)in->block_size * 4u) conf *= 0.88;
    if (in->peak_block_active_observed > 0 &&
        in->peak_block_active_observed + 4u < in->peak_active) {
        conf *= 0.90;
    }
    if (conf > 0.85) conf = 0.85;
    if (conf < 0.15) conf = 0.15;
    return clamp_confidence(conf);
}

EmbeddedTargetConfig embedded_perf_default_config(void) {
    EmbeddedTargetConfig cfg = {
        .cpu_freq_mhz = SPECTRAL_EMBEDDED_DEFAULT_CPU_MHZ,
        .sample_rate = SPECTRAL_EMBEDDED_DEFAULT_SAMPLE_RATE,
        .block_size = SPECTRAL_EMBEDDED_DEFAULT_BLOCK_SIZE,
        .max_memory_kb = SPECTRAL_EMBEDDED_DEFAULT_MEMORY_KB,
        .perf_profile = SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST,
        .pessimism_override = 0.0,
        .include_cold_start = 1,
        .verbose = 1
    };
#if defined(SPECTRAL_SIMULATION_TARGET_DAISY)
    cfg.perf_profile = SPECTRAL_PERF_PROFILE_DAISY_H750_WORST;
#endif
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
    est.profile_name = spectral_perf_model_profile_name((SpectralPerfProfileId)cfg->perf_profile);
    est.profile_version = spectral_perf_model_profile_version((SpectralPerfProfileId)cfg->perf_profile);

    if (!ops || output_samples == 0 || cfg->sample_rate == 0 ||
        cfg->cpu_freq_mhz == 0 || cfg->block_size == 0) {
        return est;
    }

    {
        const SpectralPerfModelProfile* profile =
            spectral_perf_model_profile((SpectralPerfProfileId)cfg->perf_profile);
        SpectralPerfModelInput input = {
            .output_samples = output_samples,
            .segment_count = segment_count,
            .block_size = cfg->block_size,
            .peak_active = peak_active,
            .lut_lookups = ops->lut_lookups,
            .mac_operations = ops->mac_operations,
            .phase_updates = ops->phase_updates,
            .loop_iterations = ops->loop_iterations,
            .sdram_accesses = ops->sdram_accesses,
            .cache_misses_est = ops->cache_misses_est,
            .seg_scan_checks = ops->seg_scan_checks,
            .peak_block_cycles_observed = ops->peak_block_cycles,
            .peak_block_active_observed = ops->peak_block_active
        };
        SpectralPerfStageCycles stage = {0};

        if (!profile) profile = spectral_perf_model_default_profile();
        if (!spectral_perf_model_estimate(profile, &input, &stage)) return est;

        est.model_confidence = estimate_model_confidence(profile, &input);

        est.pessimism_factor = profile->pessimism_factor;
        if (spectral_is_finite_positive_f64(cfg->pessimism_override) &&
            cfg->pessimism_override > est.pessimism_factor) {
            est.pessimism_factor = cfg->pessimism_override;
        }

        est.cycles_scan = scale_cycles_u64(stage.scan_cycles, est.pessimism_factor);
        est.cycles_oscillator = scale_cycles_u64(stage.oscillator_cycles, est.pessimism_factor);
        est.cycles_accumulation = scale_cycles_u64(stage.accumulation_cycles, est.pessimism_factor);
        est.cycles_output = scale_cycles_u64(stage.output_cycles, est.pessimism_factor);
        est.cycles_overhead = scale_cycles_u64(stage.overhead_cycles, est.pessimism_factor);
        est.cycles_memory = scale_cycles_u64(stage.memory_cycles, est.pessimism_factor);

        est.estimated_cycles = est.cycles_scan + est.cycles_oscillator +
                               est.cycles_accumulation + est.cycles_output +
                               est.cycles_overhead + est.cycles_memory;

        est.estimated_cycles_cold = est.estimated_cycles;
        if (cfg->include_cold_start) {
            est.estimated_cycles_cold = scale_cycles_u64(est.estimated_cycles, profile->cold_start_multiplier);
        }

        est.peak_block_cycles = scale_cycles_u64(stage.worst_block_cycles, est.pessimism_factor);
        est.peak_block_active = stage.worst_block_active;
        if (est.peak_block_cycles == 0 && cfg->block_size > 0) {
            uint64_t blocks = ((uint64_t)output_samples + cfg->block_size - 1u) / cfg->block_size;
            if (blocks == 0) blocks = 1;
            est.peak_block_cycles = (est.estimated_cycles / blocks);
            est.peak_block_cycles = scale_cycles_u64(est.peak_block_cycles, 1.15);
            est.peak_block_active = peak_active;
        }
    }

    {
        double cpu_freq_hz = cfg->cpu_freq_mhz * 1000000.0;
        double audio_duration_sec = (double)output_samples / cfg->sample_rate;
        double target_time_sec = (double)est.estimated_cycles / cpu_freq_hz;
        double target_time_cold_sec = (double)est.estimated_cycles_cold / cpu_freq_hz;

        est.target_time_ms = target_time_sec * 1000.0;
        est.target_time_cold_ms = target_time_cold_sec * 1000.0;
        est.cycles_available = cpu_freq_hz / cfg->sample_rate;
        est.cycles_per_sample = (double)est.estimated_cycles / output_samples;
        est.cpu_load_percent = (est.cycles_per_sample / est.cycles_available) * 100.0;
        est.cpu_load_best = est.cpu_load_percent / ((est.pessimism_factor > 1.0) ? est.pessimism_factor : 1.0);
        est.cpu_load_cold = ((double)est.estimated_cycles_cold / output_samples) / est.cycles_available * 100.0;
        est.realtime_ratio = (target_time_sec > 0.0) ? (audio_duration_sec / target_time_sec) : 0.0;

        if (est.peak_block_cycles > 0 && cfg->block_size > 0) {
            double peak_cycles_per_sample = (double)est.peak_block_cycles / cfg->block_size;
            est.cpu_load_worst = (peak_cycles_per_sample / est.cycles_available) * 100.0;
        }
        if (est.cpu_load_worst < est.cpu_load_cold) est.cpu_load_worst = est.cpu_load_cold;
    }

    return est;
}

void embedded_perf_print(const EmbeddedTargetConfig* config,
                         const EmbeddedPerfEstimate* est) {
    EmbeddedTargetConfig fallback_cfg = embedded_perf_default_config();
    const EmbeddedTargetConfig* cfg = config ? config : &fallback_cfg;
    if (!cfg->verbose || !est) return;

    char buf1[48], buf2[48];
    double ms_per_cycle = (cfg->cpu_freq_mhz > 0) ? (1000.0 / (cfg->cpu_freq_mhz * 1000000.0)) : 0.0;

    TableColumn cols[] = {
        {"Metric", 24, ALIGN_LEFT},
        {"Value", 16, ALIGN_RIGHT},
        {"Unit", 20, ALIGN_LEFT}
    };
    TableConfig table_main = {
        .columns = cols,
        .num_columns = 3,
        .border = 1,
        .title = "EMBEDDED TARGET PERFORMANCE"
    };

    table_print_header(&table_main);

    snprintf(buf1, sizeof(buf1), "%s", est->profile_name ? est->profile_name : "unknown");
    snprintf(buf2, sizeof(buf2), "%s", est->profile_version ? est->profile_version : "n/a");
    table_print_row(&table_main, "Perf profile", buf1, buf2);

    snprintf(buf1, sizeof(buf1), "%.2f", est->model_confidence);
    table_print_row(&table_main, "Model confidence", buf1, "0..1");

    snprintf(buf1, sizeof(buf1), "%s", confidence_label(est->model_confidence));
    table_print_row(&table_main, "Confidence tier", buf1, "predictive quality");

    snprintf(buf1, sizeof(buf1), "%.2f", est->pessimism_factor);
    table_print_row(&table_main, "Pessimism factor", buf1, "x warm estimate");

    snprintf(buf1, sizeof(buf1), "%.1f", est->desktop_time_ms);
    table_print_row(&table_main, "Desktop execution", buf1, "ms");

    snprintf(buf1, sizeof(buf1), "%zu", est->segment_count);
    table_print_row(&table_main, "Segments", buf1, "");

    snprintf(buf1, sizeof(buf1), "%u", est->peak_active);
    table_print_row(&table_main, "Peak active", buf1, "");

    snprintf(buf1, sizeof(buf1), "%zu", est->output_samples);
    table_print_row(&table_main, "Output samples", buf1, "");

    table_print_separator(&table_main);

    snprintf(buf1, sizeof(buf1), "%.1f", est->target_time_ms);
    table_print_row(&table_main, "Target time (warm)", buf1, "ms");

    snprintf(buf1, sizeof(buf1), "%.1f", est->target_time_cold_ms);
    table_print_row(&table_main, "Target time (cold)", buf1, "ms");

    snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_per_sample);
    table_print_row(&table_main, "Cycles/sample (warm)", buf1, "");

    snprintf(buf1, sizeof(buf1), "%.0f", est->cycles_available);
    snprintf(buf2, sizeof(buf2), "@ %u kHz", cfg->sample_rate / 1000);
    table_print_row(&table_main, "Cycles available", buf1, buf2);

    snprintf(buf1, sizeof(buf1), "%.1f%%", est->cpu_load_percent);
    table_print_row(&table_main, "CPU load (warm)", buf1, "overall avg");

    snprintf(buf1, sizeof(buf1), "%.1f%%", est->cpu_load_cold);
    table_print_row(&table_main, "CPU load (cold)", buf1, "first-run envelope");

    snprintf(buf1, sizeof(buf1), "%.1f%%", est->cpu_load_worst);
    if (est->peak_block_active > 0) {
        snprintf(buf2, sizeof(buf2), "%u active", est->peak_block_active);
    } else {
        snprintf(buf2, sizeof(buf2), "envelope");
    }
    table_print_row(&table_main, "CPU load (worst)", buf1, buf2);

    snprintf(buf1, sizeof(buf1), "%.2fx", est->realtime_ratio);
    table_print_row(&table_main, "Real-time ratio", buf1, "warm path");

    snprintf(buf1, sizeof(buf1), "M7 @ %u MHz", cfg->cpu_freq_mhz);
    snprintf(buf2, sizeof(buf2), "%u KB SRAM", cfg->max_memory_kb);
    table_print_row(&table_main, "Target", buf1, buf2);

    table_print_footer(&table_main);

    {
        TableConfig table_stages = {
            .columns = cols,
            .num_columns = 3,
            .border = 1,
            .title = "EMBEDDED STAGE BREAKDOWN (WARM, PESSIMISTIC)"
        };
        table_print_header(&table_stages);

        snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_scan * ms_per_cycle);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->cycles_scan);
        table_print_row(&table_stages, "Scan", buf1, buf2);

        snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_oscillator * ms_per_cycle);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->cycles_oscillator);
        table_print_row(&table_stages, "Oscillator", buf1, buf2);

        snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_accumulation * ms_per_cycle);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->cycles_accumulation);
        table_print_row(&table_stages, "Accumulation", buf1, buf2);

        snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_output * ms_per_cycle);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->cycles_output);
        table_print_row(&table_stages, "Output conversion", buf1, buf2);

        snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_memory * ms_per_cycle);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->cycles_memory);
        table_print_row(&table_stages, "Memory/cache", buf1, buf2);

        snprintf(buf1, sizeof(buf1), "%.1f", est->cycles_overhead * ms_per_cycle);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->cycles_overhead);
        table_print_row(&table_stages, "Overhead", buf1, buf2);

        table_print_separator(&table_stages);
        snprintf(buf1, sizeof(buf1), "%.1f", est->target_time_ms);
        snprintf(buf2, sizeof(buf2), "%llu cyc", (unsigned long long)est->estimated_cycles);
        table_print_row(&table_stages, "Total", buf1, buf2);
        table_print_footer(&table_stages);
    }

    if (est->model_confidence < 0.45) {
        status_print(STATUS_WARN, "Model confidence is low (%.2f); treat as conservative envelope.",
                    est->model_confidence);
    }

    if (est->cpu_load_worst > 100.0 && est->peak_block_active > 0) {
        status_print(STATUS_WARN, "Worst-case block exceeds real-time (%u active, %.1f%% load).",
                    est->peak_block_active, est->cpu_load_worst);
        SPECTRAL_LOG_INFO("   Peak block may cause audio glitches.");
    } else if (est->cpu_load_cold > 100.0) {
        status_print(STATUS_WARN, "Cold-start estimate exceeds real-time (%.1f%%).",
                    est->cpu_load_cold);
        SPECTRAL_LOG_INFO("   Warm steady-state may pass; first-run risk remains.");
    } else if (est->cpu_load_percent > 85.0) {
        status_print(STATUS_WARN, "CPU load high (warm %.1f%%, cold %.1f%%).",
                    est->cpu_load_percent, est->cpu_load_cold);
    } else {
        status_print(STATUS_OK, "Conservative model indicates headroom.");
    }
    SPECTRAL_LOG_INFO("");
}
