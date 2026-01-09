/* spectral_common.h - Core types and utilities */

#ifndef SPECTRAL_COMMON_H
#define SPECTRAL_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define PI          3.14159265358979323846f
#define TWO_PI      6.283185307179586f
#define INV_TWO_PI  0.159154943091895f
#define PI_SQ       9.8696044f

#define DEFAULT_N_FFT     4096
#define DEFAULT_HOP       128
#define DEFAULT_DB_THRESH -85.0f
#define CACHE_ALIGN       64

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

typedef struct __attribute__((aligned(64))) {
    float start, length, phase, freq_hz, df, amp, da, width;
    float _pad[8];  // Pad to 64-byte cache line
} Segment;

#define PREFETCH_READ(addr)  __builtin_prefetch((addr), 0, 3)
#define PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)

typedef struct {
    Segment* segs;
    size_t count;
    size_t capacity;
} SegmentArray;

typedef struct {
    float stretch, inv_stretch, inv_stretch_sq, pitch_factor;
    uint32_t out_len, num_segments;
} SynthParams;

#ifndef __CUDACC__

static inline void* spectral_aligned_alloc(size_t size) {
    return aligned_alloc(CACHE_ALIGN, (size + CACHE_ALIGN - 1) & ~(CACHE_ALIGN - 1));
}

static inline float fast_atan2(float y, float x) {
    float abs_x = fabsf(x), abs_y = fabsf(y);
    float a = (abs_x < abs_y) ? abs_x / (abs_y + 1e-10f) : abs_y / (abs_x + 1e-10f);
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    if (abs_y > abs_x) r = 1.57079637f - r;
    if (x < 0) r = 3.14159274f - r;
    if (y < 0) r = -r;
    return r;
}

static inline float fast_sin(float x) {
    x = x - TWO_PI * floorf(x * INV_TWO_PI + 0.5f);
    float x2 = x * x;
    return x * (PI_SQ - x2) / (PI_SQ + 0.25f * x2);
}

static inline SynthParams make_synth_params(float stretch, float pitch, size_t out_len, size_t num_segs) {
    return (SynthParams){
        .stretch = stretch,
        .inv_stretch = 1.0f / stretch,
        .inv_stretch_sq = 1.0f / (stretch * stretch),
        .pitch_factor = powf(2.0f, pitch / 12.0f),
        .out_len = (uint32_t)out_len,
        .num_segments = (uint32_t)num_segs
    };
}

#endif

#ifdef __CUDACC__
__device__ __forceinline__ float fast_sin_device(float x) {
    x = x - TWO_PI * floorf(x * INV_TWO_PI + 0.5f);
    float x2 = x * x;
    return x * (PI_SQ - x2) / (PI_SQ + 0.25f * x2);
}
#endif

/* Performance metrics */
#include <omp.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

typedef struct {
    size_t peak_resident_mb;
    size_t current_resident_mb;
    size_t virtual_mb;
    double user_time_ms;
    double sys_time_ms;
    double wall_time_ms;
    int num_cores;
    double cpu_utilization;  // 0-100%
    size_t tracked_allocs;   // Manual tracking
} PerfMetrics;

static inline void perf_get_memory(size_t* resident_kb, size_t* virtual_kb) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        *resident_kb = info.resident_size / 1024;
        *virtual_kb = info.virtual_size / 1024;
    } else {
        *resident_kb = *virtual_kb = 0;
    }
}

static inline void perf_get_cpu_time(double* user_ms, double* sys_ms) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        *user_ms = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
        *sys_ms = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    } else {
        *user_ms = *sys_ms = 0;
    }
}

static inline int perf_get_num_cores(void) {
    int cores = 0;
    size_t len = sizeof(cores);
    sysctlbyname("hw.ncpu", &cores, &len, NULL, 0);
    return cores;
}

static size_t g_peak_alloc = 0;
static size_t g_current_alloc = 0;

static inline void perf_track_alloc(size_t bytes) {
    #pragma omp atomic
    g_current_alloc += bytes;
    if (g_current_alloc > g_peak_alloc) {
        #pragma omp critical
        { if (g_current_alloc > g_peak_alloc) g_peak_alloc = g_current_alloc; }
    }
}

static inline void perf_track_free(size_t bytes) {
    #pragma omp atomic
    g_current_alloc -= bytes;
}

static inline void perf_reset_tracking(void) {
    g_peak_alloc = 0;
    g_current_alloc = 0;
}

static inline PerfMetrics perf_snapshot(double wall_start) {
    PerfMetrics m;
    size_t res_kb, virt_kb;
    perf_get_memory(&res_kb, &virt_kb);
    perf_get_cpu_time(&m.user_time_ms, &m.sys_time_ms);
    m.current_resident_mb = res_kb / 1024;
    m.virtual_mb = virt_kb / 1024;
    m.num_cores = perf_get_num_cores();
    m.wall_time_ms = (omp_get_wtime() - wall_start) * 1000.0;
    m.tracked_allocs = g_peak_alloc;
    m.peak_resident_mb = m.current_resident_mb; // OS-level approximation
    m.cpu_utilization = (m.wall_time_ms > 0) ? 
        100.0 * (m.user_time_ms + m.sys_time_ms) / (m.wall_time_ms * m.num_cores) : 0;
    return m;
}

static inline void perf_print(PerfMetrics* start, PerfMetrics* end, int n_threads) {
    double user_delta = end->user_time_ms - start->user_time_ms;
    double sys_delta = end->sys_time_ms - start->sys_time_ms;
    double wall_delta = end->wall_time_ms - start->wall_time_ms;
    double total_cpu = user_delta + sys_delta;
    double utilization = (wall_delta > 0) ? 100.0 * total_cpu / (wall_delta * n_threads) : 0;
    
    printf("\n--- Performance Metrics ---\n");
    printf("Memory (physical):\n");
    printf("  RSS:            %zu MB\n", end->current_resident_mb);
    printf("  Peak tracked:   %.1f MB\n", g_peak_alloc / (1024.0 * 1024.0));
    printf("CPU Time:\n");
    printf("  User:           %.1f ms\n", user_delta);
    printf("  System:         %.1f ms\n", sys_delta);
    printf("  Total CPU:      %.1f ms\n", total_cpu);
    printf("Utilization:\n");
    printf("  Threads used:   %d / %d cores\n", n_threads, end->num_cores);
    printf("  Core util:      %.1f%% (of %d threads)\n", utilization, n_threads);
    printf("  Parallelism:    %.2fx effective\n", total_cpu / wall_delta);
}

#else
/* Linux fallback using /proc */
#include <sys/resource.h>

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

static size_t g_peak_alloc = 0;
static size_t g_current_alloc = 0;

static inline void perf_track_alloc(size_t bytes) {
    #pragma omp atomic
    g_current_alloc += bytes;
    if (g_current_alloc > g_peak_alloc) {
        #pragma omp critical
        { if (g_current_alloc > g_peak_alloc) g_peak_alloc = g_current_alloc; }
    }
}

static inline void perf_track_free(size_t bytes) {
    #pragma omp atomic
    g_current_alloc -= bytes;
}

static inline void perf_reset_tracking(void) {
    g_peak_alloc = 0;
    g_current_alloc = 0;
}

static inline PerfMetrics perf_snapshot(double wall_start) {
    PerfMetrics m = {0};
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        m.user_time_ms = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
        m.sys_time_ms = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
        m.peak_resident_mb = usage.ru_maxrss / 1024; // Linux: KB, macOS: bytes
    }
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        size_t vm, rss;
        fscanf(f, "%zu %zu", &vm, &rss);
        fclose(f);
        long page_size = sysconf(_SC_PAGESIZE);
        m.virtual_mb = (vm * page_size) / (1024 * 1024);
        m.current_resident_mb = (rss * page_size) / (1024 * 1024);
    }
    m.num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    m.wall_time_ms = (omp_get_wtime() - wall_start) * 1000.0;
    m.tracked_allocs = g_peak_alloc;
    return m;
}

static inline void perf_print(PerfMetrics* start, PerfMetrics* end, int n_threads) {
    double user_delta = end->user_time_ms - start->user_time_ms;
    double sys_delta = end->sys_time_ms - start->sys_time_ms;
    double wall_delta = end->wall_time_ms - start->wall_time_ms;
    double total_cpu = user_delta + sys_delta;
    double utilization = (wall_delta > 0) ? 100.0 * total_cpu / (wall_delta * n_threads) : 0;
    
    printf("\n--- Performance Metrics ---\n");
    printf("Memory (physical):\n");
    printf("  RSS:            %zu MB\n", end->current_resident_mb);
    printf("  Peak RSS:       %zu MB\n", end->peak_resident_mb);
    printf("  Peak tracked:   %.1f MB\n", g_peak_alloc / (1024.0 * 1024.0));
    printf("CPU Time:\n");
    printf("  User:           %.1f ms\n", user_delta);
    printf("  System:         %.1f ms\n", sys_delta);
    printf("  Total CPU:      %.1f ms\n", total_cpu);
    printf("Utilization:\n");
    printf("  Threads used:   %d / %d cores\n", n_threads, end->num_cores);
    printf("  Core util:      %.1f%% (of %d threads)\n", utilization, n_threads);
    printf("  Parallelism:    %.2fx effective\n", total_cpu / wall_delta);
}
#endif

#endif
