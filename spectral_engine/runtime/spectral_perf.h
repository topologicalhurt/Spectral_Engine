/* spectral_perf.h - Performance Metrics and Embedded Target Estimation
 * 
 * Desktop performance tracking:
 *   - Memory usage (RSS, virtual, tracked allocations)
 *   - CPU time (user, system, wall clock)
 *   - CPU utilization and thread efficiency
 * 
 * Embedded target estimation:
 *   - Cycle counting based on Q15 operation costs
 *   - Real-time feasibility assessment
 *   - Memory footprint calculation
 * 
 * The embedded estimation allows simulation on desktop before deployment,
 * predicting whether audio will render in real-time on Cortex-M7.
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
    size_t peak_resident_mb;
    size_t current_resident_mb;
    size_t virtual_mb;
    double user_time_ms;
    double sys_time_ms;
    double wall_time_ms;
    int num_cores;
    double cpu_utilization;
    size_t tracked_allocs;
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
    uint32_t perf_profile;        /* SpectralPerfProfileId */
    double   pessimism_override;  /* 0 = profile default, >0 clamps to >= default */
    int      include_cold_start;  /* include cold-start envelope estimate */
    int      verbose;
} EmbeddedTargetConfig;

typedef struct {
    double   desktop_time_ms;
    uint64_t estimated_cycles;
    uint64_t estimated_cycles_cold;
    double   target_time_ms;
    double   target_time_cold_ms;
    double   cycles_per_sample;
    double   cycles_available;
    double   cpu_load_percent;
    double   cpu_load_best;        /* optimistic: good cache + compiler opts */
    double   cpu_load_worst;       /* pessimistic: peak block or +30% overhead */
    double   cpu_load_cold;
    double   realtime_ratio;
    double   model_confidence;     /* confidence in absolute estimate [0,1] */
    double   pessimism_factor;     /* explicit worst-case safety multiplier */
    const char* profile_name;      /* stable perf profile name */
    const char* profile_version;   /* model profile revision */
    size_t   segment_count;
    uint32_t peak_active;
    size_t   output_samples;
    uint64_t peak_block_cycles;    /* cycles for worst-case block */
    uint32_t peak_block_active;    /* active count in worst-case block */
    uint64_t cycles_scan;
    uint64_t cycles_oscillator;
    uint64_t cycles_accumulation;
    uint64_t cycles_output;
    uint64_t cycles_overhead;
    uint64_t cycles_memory;
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
/*
 * Cycle-cost assumptions live in runtime/spectral_perf_model.[ch] as
 * versioned profile data. This keeps model tuning isolated from users
 * of the estimator API.
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
