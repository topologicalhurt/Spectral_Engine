/* spectral_perf_accounting.h - Canonical perf accounting contract
 *
 * This layer defines the semantic counters used by both simulation and
 * embedded profiling builds. Keep instrumentation updates here to avoid
 * model drift as synthesis code evolves.
 */
#ifndef SPECTRAL_PERF_ACCOUNTING_H
#define SPECTRAL_PERF_ACCOUNTING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpectralPerfCounters {
    uint64_t lut_lookups;
    uint64_t mac_operations;
    uint64_t phase_updates;
    uint64_t loop_iterations;
    uint64_t sdram_accesses;       /* segment activations requiring SDRAM read */
    uint64_t cache_misses_est;     /* estimated L1 cache misses */
    uint64_t seg_scan_checks;      /* total segment scan iterations */
    uint32_t peak_block_active;    /* worst-case active count in any block */
    uint64_t peak_block_cycles;    /* cycles for worst-case block */
    /* Accumulator memory traffic (q31 read-modify-write). Segment-major touches a
     * slot once PER VOICE (O(voices*samples)); a voice-parallel nest touches it once
     * per sample (O(samples)). This counter makes that memory-bandwidth difference
     * measurable rather than asserted (ULTRAPLAN A2). */
    uint64_t accum_rw_words;       /* q31 accumulator loads + stores */
} SpectralPerfCounters;

static inline void spectral_perf_counters_reset(SpectralPerfCounters* c) {
    if (!c) return;
    c->lut_lookups = 0;
    c->mac_operations = 0;
    c->phase_updates = 0;
    c->loop_iterations = 0;
    c->sdram_accesses = 0;
    c->cache_misses_est = 0;
    c->seg_scan_checks = 0;
    c->peak_block_active = 0;
    c->peak_block_cycles = 0;
    c->accum_rw_words = 0;
}

static inline uint64_t spectral_perf_loop_iters_for_samples(uint32_t sample_count) {
    return ((uint64_t)sample_count + 3u) >> 2;
}

static inline void spectral_perf_count_segment_scan(SpectralPerfCounters* c, uint32_t checks) {
    if (!c) return;
    c->seg_scan_checks += checks;
}

static inline void spectral_perf_count_segment_activations(SpectralPerfCounters* c,
                                                           uint32_t activations) {
    if (!c) return;
    c->sdram_accesses += activations;
}

static inline void spectral_perf_count_segment_samples(SpectralPerfCounters* c,
                                                       uint32_t sample_count) {
    if (!c || sample_count == 0) return;
    c->lut_lookups += sample_count;
    c->mac_operations += sample_count;
    c->phase_updates += sample_count;
    c->loop_iterations += spectral_perf_loop_iters_for_samples(sample_count);
    /* Segment-major today: each sample is a read-modify-write of one accumulator
     * slot (1 load + 1 store). A voice-parallel nest would account this once per
     * sample per block instead of per voice. */
    c->accum_rw_words += (uint64_t)sample_count * 2u;
}

static inline void spectral_perf_count_cache_pressure(SpectralPerfCounters* c,
                                                      uint32_t active_count,
                                                      uint32_t miss_threshold_active,
                                                      uint32_t block_len) {
    if (!c || block_len == 0 || active_count <= miss_threshold_active) return;
    c->cache_misses_est += (uint64_t)(active_count - miss_threshold_active) * block_len;
}

static inline void spectral_perf_record_peak_block(SpectralPerfCounters* c,
                                                   uint64_t block_cycles,
                                                   uint32_t block_active) {
    if (!c) return;
    if (block_cycles > c->peak_block_cycles) {
        c->peak_block_cycles = block_cycles;
        c->peak_block_active = block_active;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PERF_ACCOUNTING_H */
