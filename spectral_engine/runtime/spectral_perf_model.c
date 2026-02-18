/* spectral_perf_model.c - Profile-driven embedded performance model */
#include "spectral_perf_model.h"

#include <string.h>

/* Conservative defaults. Values are intentionally pessimistic. */
static const SpectralPerfModelProfile k_profiles[SPECTRAL_PERF_PROFILE_COUNT] = {
    {
        .id = SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST,
        .name = "m7_generic_worst",
        .version = "v1",
        .confidence = 0.70,
        .pessimism_factor = 1.25,
        .cold_start_multiplier = 1.10,
        .cache_miss_threshold_active = 24,
        .bandwidth_pressure_threshold_active = 96,
        .cycles_phase_update = 3,
        .cycles_lut_lookup = 7,
        .cycles_q15_mul = 1,
        .cycles_accumulate = 4,
        .cycles_amp_update = 1,
        .cycles_loop_per_4 = 5,
        .cycles_seg_scan = 6,
        .cycles_sdram_activation = 14,
        .cycles_cache_miss = 7,
        .cycles_bandwidth_pressure = 1,
        .cycles_pipeline_stall = 1,
        .cycles_callback_overhead = 80,
        .cycles_memset_per_sample = 3,
        .cycles_output_per_sample = 4,
        .cycles_seg_bounds_check = 18,
        .cycles_seg_activation = 42
    },
    {
        .id = SPECTRAL_PERF_PROFILE_DAISY_H750_WORST,
        .name = "daisy_h750_worst",
        .version = "v1",
        .confidence = 0.76,
        .pessimism_factor = 1.35,
        .cold_start_multiplier = 1.15,
        .cache_miss_threshold_active = 20,
        .bandwidth_pressure_threshold_active = 80,
        .cycles_phase_update = 3,
        .cycles_lut_lookup = 8,
        .cycles_q15_mul = 1,
        .cycles_accumulate = 4,
        .cycles_amp_update = 1,
        .cycles_loop_per_4 = 6,
        .cycles_seg_scan = 7,
        .cycles_sdram_activation = 18,
        .cycles_cache_miss = 9,
        .cycles_bandwidth_pressure = 2,
        .cycles_pipeline_stall = 2,
        .cycles_callback_overhead = 96,
        .cycles_memset_per_sample = 3,
        .cycles_output_per_sample = 4,
        .cycles_seg_bounds_check = 22,
        .cycles_seg_activation = 48
    }
};

static const SpectralPerfModelProfile* clamp_profile(const SpectralPerfModelProfile* p) {
    return p ? p : &k_profiles[SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST];
}

const SpectralPerfModelProfile* spectral_perf_model_profile(SpectralPerfProfileId id) {
    if ((int)id < 0 || id >= SPECTRAL_PERF_PROFILE_COUNT) return NULL;
    return &k_profiles[id];
}

const SpectralPerfModelProfile* spectral_perf_model_default_profile(void) {
    return &k_profiles[SPECTRAL_PERF_PROFILE_M7_GENERIC_WORST];
}

const char* spectral_perf_model_profile_name(SpectralPerfProfileId id) {
    const SpectralPerfModelProfile* p = spectral_perf_model_profile(id);
    return p ? p->name : spectral_perf_model_default_profile()->name;
}

const char* spectral_perf_model_profile_version(SpectralPerfProfileId id) {
    const SpectralPerfModelProfile* p = spectral_perf_model_profile(id);
    return p ? p->version : spectral_perf_model_default_profile()->version;
}

uint64_t spectral_perf_model_estimate_block_cycles(
    const SpectralPerfModelProfile* profile,
    uint32_t block_len,
    uint32_t active_count,
    uint32_t block_activations,
    uint32_t seg_scan_checks,
    uint64_t lut_lookups,
    uint64_t mac_operations,
    uint64_t phase_updates,
    uint64_t loop_iterations)
{
    const SpectralPerfModelProfile* p = clamp_profile(profile);

    uint64_t scan_cycles = (uint64_t)seg_scan_checks * p->cycles_seg_scan;
    uint64_t osc_cycles = lut_lookups * p->cycles_lut_lookup +
                          phase_updates * p->cycles_phase_update +
                          lut_lookups * p->cycles_amp_update +
                          loop_iterations * p->cycles_loop_per_4;
    uint64_t accum_cycles = mac_operations * ((uint64_t)p->cycles_q15_mul + p->cycles_accumulate);
    uint64_t output_cycles = (uint64_t)block_len * p->cycles_output_per_sample;
    uint64_t overhead_cycles = p->cycles_callback_overhead +
                               (uint64_t)block_len * p->cycles_memset_per_sample +
                               (uint64_t)active_count * p->cycles_seg_bounds_check +
                               (uint64_t)block_activations * p->cycles_seg_activation;
    uint64_t memory_cycles = (uint64_t)block_activations * p->cycles_sdram_activation;

    if (active_count > p->cache_miss_threshold_active) {
        uint64_t miss_units = (uint64_t)(active_count - p->cache_miss_threshold_active) * block_len;
        memory_cycles += miss_units * p->cycles_cache_miss;
        overhead_cycles += ((miss_units + 3u) >> 2) * p->cycles_pipeline_stall;
    }

    if (active_count > p->bandwidth_pressure_threshold_active) {
        uint64_t pressure_units = (uint64_t)(active_count - p->bandwidth_pressure_threshold_active) * block_len;
        memory_cycles += pressure_units * p->cycles_bandwidth_pressure;
    }

    return scan_cycles + osc_cycles + accum_cycles + output_cycles + overhead_cycles + memory_cycles;
}

int spectral_perf_model_estimate(const SpectralPerfModelProfile* profile,
                                 const SpectralPerfModelInput* input,
                                 SpectralPerfStageCycles* out) {
    const SpectralPerfModelProfile* p = clamp_profile(profile);
    if (!input || !out) return 0;
    memset(out, 0, sizeof(*out));
    if (input->output_samples == 0 || input->block_size == 0) return 0;

    const uint64_t block_count =
        ((uint64_t)input->output_samples + input->block_size - 1u) / input->block_size;

    out->scan_cycles = input->seg_scan_checks * p->cycles_seg_scan;

    out->oscillator_cycles =
        input->lut_lookups * p->cycles_lut_lookup +
        input->phase_updates * p->cycles_phase_update +
        input->lut_lookups * p->cycles_amp_update +
        input->loop_iterations * p->cycles_loop_per_4;

    out->accumulation_cycles =
        input->mac_operations * ((uint64_t)p->cycles_q15_mul + p->cycles_accumulate);

    out->output_cycles = (uint64_t)input->output_samples * p->cycles_output_per_sample;

    out->overhead_cycles =
        block_count * p->cycles_callback_overhead +
        (uint64_t)input->output_samples * p->cycles_memset_per_sample +
        block_count * (uint64_t)input->peak_active * p->cycles_seg_bounds_check +
        (uint64_t)input->segment_count * p->cycles_seg_activation;

    out->memory_cycles =
        input->sdram_accesses * p->cycles_sdram_activation +
        input->cache_misses_est * p->cycles_cache_miss;

    if (input->peak_active > p->bandwidth_pressure_threshold_active) {
        uint64_t pressure_units =
            (uint64_t)(input->peak_active - p->bandwidth_pressure_threshold_active) *
            input->output_samples;
        out->memory_cycles += pressure_units * p->cycles_bandwidth_pressure;
    }

    if (input->peak_active > p->cache_miss_threshold_active) {
        uint64_t stall_units =
            (uint64_t)(input->peak_active - p->cache_miss_threshold_active) *
            ((input->output_samples + 3u) >> 2);
        out->overhead_cycles += stall_units * p->cycles_pipeline_stall;
    }

    out->total_cycles = out->scan_cycles + out->oscillator_cycles +
                        out->accumulation_cycles + out->output_cycles +
                        out->overhead_cycles + out->memory_cycles;

    if (input->peak_block_cycles_observed > 0) {
        out->worst_block_cycles = input->peak_block_cycles_observed;
        out->worst_block_active = input->peak_block_active_observed;
        return 1;
    }

    {
        uint64_t lut_per_block = (block_count > 0) ? (input->lut_lookups / block_count) : 0;
        uint64_t mac_per_block = (block_count > 0) ? (input->mac_operations / block_count) : 0;
        uint64_t phase_per_block = (block_count > 0) ? (input->phase_updates / block_count) : 0;
        uint64_t loop_per_block = (block_count > 0) ? (input->loop_iterations / block_count) : 0;
        uint32_t acts_per_block = (uint32_t)((block_count > 0) ? (input->sdram_accesses / block_count) : 0);
        uint32_t scans_per_block = (uint32_t)((block_count > 0) ? (input->seg_scan_checks / block_count) : 0);

        out->worst_block_cycles = spectral_perf_model_estimate_block_cycles(
            p,
            input->block_size,
            input->peak_active,
            acts_per_block,
            scans_per_block,
            lut_per_block,
            mac_per_block,
            phase_per_block,
            loop_per_block);
        out->worst_block_active = input->peak_active;
    }

    return 1;
}
