/* spectral_perf_accounting.h - Canonical perf accounting contract
 *
 * MEASURED workload counters shared by the simulation and embedded profiling
 * builds. Every field is a counted quantity of the real kernel's structure —
 * cycle PROJECTIONS are not made here (or anywhere in C): they come from the
 * validated M7 measurement stack (tools/spectral_tools/performance/embedded/,
 * M7_PERF_MODEL_PLAN). The old per-op cost taxonomy (lut_lookups & friends)
 * was retired: it priced a kernel shape that no longer exists (the
 * coupled-form oscillator replaced the per-sample LUT gather), and its
 * constants were uncalibrated.
 */
#ifndef SPECTRAL_PERF_ACCOUNTING_H
#define SPECTRAL_PERF_ACCOUNTING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SpectralPerfCounters {
    uint64_t voice_samples;        /* per-voice samples synthesized (osc+MAC work units) */
    uint64_t sdram_accesses;       /* segment activations requiring SDRAM read */
    uint64_t seg_scan_checks;      /* total segment scan iterations */
    uint32_t peak_block_active;    /* worst-case active count in any block */
    /* MEASURED on-device only (DWT cycle counter under SPECTRAL_RESTRICTED_PROFILE);
     * 0 everywhere else — never synthesized from a cost model. */
    uint64_t peak_block_cycles;
    /* Accumulator memory traffic (q31 read-modify-write). Segment-major touches a
     * slot once PER VOICE (O(voices*samples)); a voice-parallel nest touches it once
     * per sample (O(samples)). This counter makes that memory-bandwidth difference
     * measurable rather than asserted. */
    uint64_t accum_rw_words;       /* q31 accumulator loads + stores */
} SpectralPerfCounters;

static inline void spectral_perf_counters_reset(SpectralPerfCounters* c) {
    if (!c) return;
    c->voice_samples = 0;
    c->sdram_accesses = 0;
    c->seg_scan_checks = 0;
    c->peak_block_active = 0;
    c->peak_block_cycles = 0;
    c->accum_rw_words = 0;
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
    c->voice_samples += sample_count;
    /* Segment-major today: each sample is a read-modify-write of one accumulator
     * slot (1 load + 1 store). A voice-parallel nest would account this once per
     * sample per block instead of per voice. */
    c->accum_rw_words += (uint64_t)sample_count * 2u;
}

/* A voice folded into a partner's accumulator pass via the dual-MAC (SMLALD): it adds
 * its own oscillator/MAC work but shares the partner's single accumulator
 * read-modify-write, so it adds NO accumulator traffic. This is what makes the
 * dual-MAC's halved accumulator traffic show up in accum_rw_words. */
static inline void spectral_perf_count_paired_voice(SpectralPerfCounters* c,
                                                    uint32_t sample_count) {
    if (!c || sample_count == 0) return;
    c->voice_samples += sample_count;
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
