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
 * The embedded estimation allows testing on desktop before deployment,
 * predicting whether audio will render in real-time on Cortex-M7.
 */
#ifndef SPECTRAL_PERF_H
#define SPECTRAL_PERF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Portable high-resolution timing
 * Uses OpenMP when available, falls back to clock_gettime. */
#ifdef _OPENMP
#include <omp.h>
#define spectral_get_time_sec() omp_get_wtime()
#else
#include <time.h>
static inline double spectral_get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

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

/* Embedded target performance estimation (used by emulator) */

typedef struct {
    uint32_t cpu_freq_mhz;
    uint32_t sample_rate;
    uint32_t block_size;
    uint32_t max_memory_kb;
    int      verbose;
} EmbeddedTargetConfig;

typedef struct {
    double   desktop_time_ms;
    uint64_t estimated_cycles;
    double   target_time_ms;
    double   cycles_per_sample;
    double   cycles_available;
    double   cpu_load_percent;
    double   realtime_ratio;
    size_t   segment_count;
    uint32_t peak_active;
    size_t   output_samples;
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
#define EMBEDDED_DEFAULT_CPU_MHZ     480       /* STM32H750 max frequency */
#define EMBEDDED_DEFAULT_SAMPLE_RATE 48000     /* Audio sample rate */
#define EMBEDDED_DEFAULT_BLOCK_SIZE  256       /* Samples per callback */

/* Memory targets */
#define EMBEDDED_SRAM_KB             512       /* Internal SRAM (DTCM + SRAM1/2) */
#define EMBEDDED_SDRAM_KB            65536     /* External SDRAM (64 MB) */
#define EMBEDDED_DEFAULT_MEMORY_KB   (EMBEDDED_SRAM_KB + EMBEDDED_SDRAM_KB)  /* Total: ~65 MB */

/*
 * Estimated cycle costs for Q15 synthesis on ARM Cortex-M7
 * 
 * These model the inner loop of spectral_synth_embedded.c:
 *   1. Phase accumulator update: phase += freq_inc (ADD, 1 cycle + memory)
 *   2. LUT lookup: lut[phase >> 22] with interpolation (load + mul + add)
 *   3. Amplitude scale: sample * amplitude (SMULBB, 1 cycle)
 *   4. Accumulate: accum[i] += scaled (load + add + store)
 *   5. Amplitude update: amp += d_amp (ADD)
 * 
 * M7 has single-cycle 32x32->64 multiply (SMULL), dual-issue pipeline.
 * SRAM access is typically 0-wait-state. Cache effects not modeled.
 * 
 * These are conservative estimates; actual may be ~20% faster with
 * good cache behavior and compiler optimization (-O2/-O3).
 */
#define EMBEDDED_CYCLES_PHASE_UPDATE    2   /* 32-bit ADD + store */
#define EMBEDDED_CYCLES_LUT_LOOKUP      6   /* Load + shift + load + interpolate */
#define EMBEDDED_CYCLES_Q15_MUL         1   /* SMULBB (single cycle on M7) */
#define EMBEDDED_CYCLES_Q15_MAC         2   /* SMLABB (mul-accumulate) */
#define EMBEDDED_CYCLES_ACCUMULATE      3   /* Load + add + store to accum buffer */
#define EMBEDDED_CYCLES_AMP_UPDATE      1   /* 16-bit ADD */
#define EMBEDDED_CYCLES_LOOP_OVERHEAD   4   /* Branch, counter update, pipeline */

/* Per-sample cost for one active segment */
#define EMBEDDED_CYCLES_PER_SAMPLE  (EMBEDDED_CYCLES_PHASE_UPDATE + \
                                     EMBEDDED_CYCLES_LUT_LOOKUP +   \
                                     EMBEDDED_CYCLES_Q15_MUL +      \
                                     EMBEDDED_CYCLES_ACCUMULATE +   \
                                     EMBEDDED_CYCLES_AMP_UPDATE)

/* Loop overhead amortized over 4 samples (unrolled inner loop) */
#define EMBEDDED_CYCLES_LOOP_PER_4  (EMBEDDED_CYCLES_LOOP_OVERHEAD)

typedef struct {
    uint64_t lut_lookups;
    uint64_t mac_operations;
    uint64_t phase_updates;
    uint64_t loop_iterations;
} EmbeddedOpCounts;

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
