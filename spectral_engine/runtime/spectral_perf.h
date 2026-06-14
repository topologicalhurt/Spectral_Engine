/* spectral_perf.h - Performance Metrics and Embedded Target Estimation
 * 
 * Desktop performance tracking:
 *   - Memory usage (RSS, virtual, tracked allocations)
 *   - CPU time (user, system, wall clock)
 *   - CPU utilization and thread efficiency
 * 
 * Embedded target reporting:
 *   - MEASURED workload counters (voice-samples, scan checks, activations,
 *     accumulator traffic) from running the real kernel
 *   - Real-time budget arithmetic (cycles available per block at the target
 *     clock — arithmetic, not a model)
 *   - Memory footprint calculation (exact, from the data structures)
 *
 * Cycle PROJECTIONS are deliberately absent: they come from the validated M7
 * measurement stack (m7-census / m7-stalls / m7-wcet — M7_PERF_MODEL_PLAN),
 * never from in-C cost constants. The old profile-driven estimator
 * (spectral_perf_model.*) was retired: uncalibrated constants pricing a
 * kernel shape that no longer exists.
 */
#ifndef SPECTRAL_PERF_H
#define SPECTRAL_PERF_H

#include <stddef.h>
#include <stdint.h>
#include "spectral_config.h"

struct SpectralPerfCounters;

#ifdef __cplusplus
extern "C" {
#endif

/* Portable high-resolution timing
 * Uses OpenMP when available, falls back to clock_gettime.
 * See spectral_omp.h for the non-OMP fallback of omp_get_wtime. */
#include "spectral_omp.h"
#define spectral_get_time_sec() omp_get_wtime()

typedef struct {
    size_t current_resident_mb;
    double user_time_ms;
    double sys_time_ms;
    double wall_time_ms;
    int num_cores;
} PerfMetrics;

extern size_t g_peak_alloc;
extern size_t g_current_alloc;

void perf_get_memory(size_t* resident_kb, size_t* virtual_kb);
void perf_get_cpu_time(double* user_ms, double* sys_ms);
int perf_get_num_cores(void);

void perf_track_alloc(size_t bytes);
void perf_track_free(size_t bytes);
void perf_reset_tracking(void);

PerfMetrics perf_snapshot(double wall_start);
void perf_print(PerfMetrics* start, PerfMetrics* end, int n_threads);

/* Embedded target performance estimation (used by simulation workflow) */

typedef struct {
    uint32_t cpu_freq_mhz;
    uint32_t sample_rate;
    uint32_t block_size;
    uint32_t max_memory_kb;
    int      verbose;
} EmbeddedTargetConfig;

/* Measured workload + budget arithmetic. No field here is a cycle projection:
 * peak_block_cycles is a DWT measurement on restricted-profile device builds
 * (0 = unavailable), everything else is a count or plain arithmetic. */
typedef struct {
    double   desktop_time_ms;
    size_t   segment_count;
    uint32_t peak_active;
    size_t   output_samples;

    /* Measured workload (real kernel structure). */
    uint64_t voice_samples;
    uint64_t seg_scan_checks;
    uint64_t segment_activations;
    uint64_t accum_rw_words;

    /* Real-time budget arithmetic at the target clock. */
    double   cycles_per_sample_budget;   /* cpu_hz / sample_rate */
    double   cycles_per_block_budget;    /* budget * block_size */
    double   budget_per_voice_sample;    /* block budget / (peak_active*block) */

    /* Measured on-device only (DWT, SPECTRAL_RESTRICTED_PROFILE); else 0. */
    uint64_t peak_block_cycles;
    uint32_t peak_block_active;
} EmbeddedPerfEstimate;

/*
 * Default target specs (Arm Cortex M7 / STM32H750 + 64MB SDRAM)
 * 
 * Memory hierarchy:
 *   - DTCM:     128 KB @ 0 wait states (tightly coupled, fastest)
 *   - SRAM1/2:  384 KB @ 0-1 wait states (internal)
 *   - SDRAM:     64 MB @ variable latency (external, cached)
 * 
 * Total usable: ~65 MB. Exact apportioning between memory regions
 * depends on linker script and runtime allocation strategy. This
 * estimator reports total usage vs total capacity - actual placement
 * of hot data (active segments, accum buffer) in fast SRAM vs cold
 * data (segment pool) in SDRAM is a deployment consideration.
 */
typedef struct SpectralPerfCounters EmbeddedOpCounts;

/* Embedded memory usage (exact calculation from data structures) */

typedef struct {
    size_t ctx_bytes;
    size_t osc_lut_bytes;
    size_t active_array_bytes;
    size_t static_total;
    
    size_t segment_data_bytes;
    size_t num_segments;
    size_t bytes_per_segment;
    
    size_t accum_buffer_bytes;
    size_t block_size;
    
    size_t total_static;
    size_t total_dynamic;
    size_t total_transient;
    size_t total_bytes;
    size_t total_kb;
    
    size_t target_max_kb;
    double usage_percent;
    int    fits_in_target;
    
    size_t active_seg_size;
    uint32_t max_active;
} EmbeddedMemoryUsage;

EmbeddedMemoryUsage embedded_memory_usage(
    size_t   num_segments,
    uint32_t block_size,
    uint32_t osc_lut_bits,
    uint32_t max_active,
    uint32_t target_kb
);

void embedded_memory_print(const EmbeddedMemoryUsage* mem);
EmbeddedTargetConfig embedded_perf_default_config(void);

EmbeddedPerfEstimate embedded_perf_estimate(
    const EmbeddedTargetConfig* config,
    const EmbeddedOpCounts* ops,
    size_t   output_samples,
    size_t   segment_count,
    uint32_t peak_active,
    double   desktop_time_sec
);

void embedded_perf_print(const EmbeddedTargetConfig* config,
                         const EmbeddedPerfEstimate* est);

#ifdef __cplusplus
}
#endif

#endif /* SPECTRAL_PERF_H */
