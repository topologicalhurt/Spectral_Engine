/* spectral_perf_model.h - Profile-driven embedded performance model
 *
 * Keeps cycle-cost assumptions isolated from simulation/business logic.
 * The estimator consumes operation counts and returns stage-wise cycle
 * projections for conservative embedded planning.
 */
#ifndef SPECTRAL_PERF_MODEL_H
#define SPECTRAL_PERF_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST = 0,
    SPECTRAL_PERF_PROFILE_DAISY_H750_WORST = 1,
    SPECTRAL_PERF_PROFILE_COUNT
} SpectralPerfProfileId;

typedef struct {
    SpectralPerfProfileId id;
    const char* name;
    const char* version;

    /* Confidence in absolute-cycle projection fidelity [0, 1]. */
    double confidence;

    /* Conservative multiplier applied to projected warm cycles. */
    double pessimism_factor;

    /* Additional first-run penalty multiplier (cache cold/startup effects). */
    double cold_start_multiplier;

    /* Memory hierarchy assumptions. */
    uint32_t cache_miss_threshold_active;
    uint32_t bandwidth_pressure_threshold_active;

    /* Core synthesis stage costs. */
    uint32_t cycles_phase_update;
    uint32_t cycles_lut_lookup;
    uint32_t cycles_q15_mul;
    uint32_t cycles_accumulate;
    uint32_t cycles_amp_update;
    uint32_t cycles_loop_per_4;

    /* Scan / scheduling / memory penalties. */
    uint32_t cycles_seg_scan;
    uint32_t cycles_sdram_activation;
    uint32_t cycles_cache_miss;
    uint32_t cycles_bandwidth_pressure;
    uint32_t cycles_pipeline_stall;

    /* Block and activation overheads. */
    uint32_t cycles_callback_overhead;
    uint32_t cycles_memset_per_sample;
    uint32_t cycles_output_per_sample;
    uint32_t cycles_seg_bounds_check;
    uint32_t cycles_seg_activation;
} SpectralPerfModelProfile;

typedef struct {
    size_t output_samples;
    size_t segment_count;
    uint32_t block_size;
    uint32_t peak_active;

    uint64_t lut_lookups;
    uint64_t mac_operations;
    uint64_t phase_updates;
    uint64_t loop_iterations;
    uint64_t sdram_accesses;
    uint64_t cache_misses_est;
    uint64_t seg_scan_checks;

    /* Optional direct observation from simulation loop. */
    uint64_t peak_block_cycles_observed;
    uint32_t peak_block_active_observed;
} SpectralPerfModelInput;

typedef struct {
    uint64_t scan_cycles;
    uint64_t oscillator_cycles;
    uint64_t accumulation_cycles;
    uint64_t output_cycles;
    uint64_t overhead_cycles;
    uint64_t memory_cycles;
    uint64_t total_cycles;
    uint64_t worst_block_cycles;
    uint32_t worst_block_active;
} SpectralPerfStageCycles;

const SpectralPerfModelProfile* spectral_perf_model_profile(SpectralPerfProfileId id);
const SpectralPerfModelProfile* spectral_perf_model_default_profile(void);
const char* spectral_perf_model_profile_name(SpectralPerfProfileId id);
const char* spectral_perf_model_profile_version(SpectralPerfProfileId id);

int spectral_perf_model_estimate(const SpectralPerfModelProfile* profile,
                                 const SpectralPerfModelInput* input,
                                 SpectralPerfStageCycles* out);

uint64_t spectral_perf_model_estimate_block_cycles(
    const SpectralPerfModelProfile* profile,
    uint32_t block_len,
    uint32_t active_count,
    uint32_t block_activations,
    uint32_t seg_scan_checks,
    uint64_t lut_lookups,
    uint64_t mac_operations,
    uint64_t phase_updates,
    uint64_t loop_iterations);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PERF_MODEL_H */
