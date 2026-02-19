/* spectral_perf.c - Desktop Performance Metrics
 *
 * Platform-specific memory and CPU time queries:
 *   macOS: mach_task_info(), getrusage(), sysctlbyname()
 *   Linux: /proc/self/statm, getrusage(), sysconf()
 *
 * Allocation Tracking:
 *   perf_track_alloc/free maintain running total and peak
 *   Used to measure actual memory consumption vs theoretical
 *
 * Embedded estimation is in runtime/spectral_perf_embedded.c
 */
#include "spectral_perf.h"
#include "spectral_utils.h"
#include "spectral_log.h"
#include <stdio.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

/* Global allocation tracking state */
size_t g_peak_alloc = 0;
size_t g_current_alloc = 0;

#ifdef __APPLE__

void perf_get_memory(size_t* resident_kb, size_t* virtual_kb) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        *resident_kb = (size_t)(info.resident_size / SPECTRAL_BYTES_PER_KIB);
        *virtual_kb = (size_t)(info.virtual_size / SPECTRAL_BYTES_PER_KIB);
    } else {
        *resident_kb = *virtual_kb = 0;
    }
}

void perf_get_cpu_time(double* user_ms, double* sys_ms) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        *user_ms = usage.ru_utime.tv_sec * SPECTRAL_MILLIS_PER_SECOND_D +
                   usage.ru_utime.tv_usec / SPECTRAL_MILLIS_PER_SECOND_D;
        *sys_ms = usage.ru_stime.tv_sec * SPECTRAL_MILLIS_PER_SECOND_D +
                  usage.ru_stime.tv_usec / SPECTRAL_MILLIS_PER_SECOND_D;
    } else {
        *user_ms = *sys_ms = 0;
    }
}

int perf_get_num_cores(void) {
    int cores = 0;
    size_t len = sizeof(cores);
    sysctlbyname("hw.ncpu", &cores, &len, NULL, 0);
    return (cores > 0) ? cores : 1;
}

#else /* Linux */

void perf_get_memory(size_t* resident_kb, size_t* virtual_kb) {
    *resident_kb = *virtual_kb = 0;
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        size_t vm, rss;
        if (fscanf(f, "%zu %zu", &vm, &rss) == 2) {
            long page_size = sysconf(_SC_PAGESIZE);
            size_t vm_bytes = 0;
            size_t rss_bytes = 0;
            if (page_size > 0 &&
                spectral_size_mul(vm, (size_t)page_size, &vm_bytes) &&
                spectral_size_mul(rss, (size_t)page_size, &rss_bytes)) {
                *virtual_kb = vm_bytes / SPECTRAL_BYTES_PER_KIB;
                *resident_kb = rss_bytes / SPECTRAL_BYTES_PER_KIB;
            }
        }
        fclose(f);
    }
}

void perf_get_cpu_time(double* user_ms, double* sys_ms) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        *user_ms = usage.ru_utime.tv_sec * SPECTRAL_MILLIS_PER_SECOND_D +
                   usage.ru_utime.tv_usec / SPECTRAL_MILLIS_PER_SECOND_D;
        *sys_ms = usage.ru_stime.tv_sec * SPECTRAL_MILLIS_PER_SECOND_D +
                  usage.ru_stime.tv_usec / SPECTRAL_MILLIS_PER_SECOND_D;
    } else {
        *user_ms = *sys_ms = 0;
    }
}

int perf_get_num_cores(void) {
    int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return (cores > 0) ? cores : 1;
}

#endif /* __APPLE__ */

void perf_track_alloc(size_t bytes) {
    #pragma omp critical
    {
        g_current_alloc += bytes;
        if (g_current_alloc > g_peak_alloc) g_peak_alloc = g_current_alloc;
    }
}

void perf_track_free(size_t bytes) {
    #pragma omp critical
    {
        if (bytes >= g_current_alloc) g_current_alloc = 0;
        else g_current_alloc -= bytes;
    }
}

void perf_reset_tracking(void) {
    g_peak_alloc = 0;
    g_current_alloc = 0;
}

PerfMetrics perf_snapshot(double wall_start) {
    PerfMetrics m = {0};
    size_t res_kb, virt_kb;
    perf_get_memory(&res_kb, &virt_kb);
    perf_get_cpu_time(&m.user_time_ms, &m.sys_time_ms);
    m.current_resident_mb = res_kb / SPECTRAL_BYTES_PER_KIB;
    m.virtual_mb = virt_kb / SPECTRAL_BYTES_PER_KIB;
    m.num_cores = perf_get_num_cores();
    m.wall_time_ms = (omp_get_wtime() - wall_start) * SPECTRAL_MILLIS_PER_SECOND_D;
    m.tracked_allocs = g_peak_alloc;
    m.peak_resident_mb = m.current_resident_mb;
    m.cpu_utilization = (m.wall_time_ms > 0) ?
        100.0 * (m.user_time_ms + m.sys_time_ms) / (m.wall_time_ms * m.num_cores) : 0;
    return m;
}

void perf_print(PerfMetrics* start, PerfMetrics* end, int n_threads) {
    double user_delta = end->user_time_ms - start->user_time_ms;
    double sys_delta = end->sys_time_ms - start->sys_time_ms;
    double wall_delta = end->wall_time_ms - start->wall_time_ms;
    double total_cpu = user_delta + sys_delta;
    double utilization = (wall_delta > 0) ? 100.0 * total_cpu / (wall_delta * n_threads) : 0;
    double parallelism = (wall_delta > 0) ? (total_cpu / wall_delta) : 0.0;

    SPECTRAL_LOG_INFO("\n--- Performance Metrics ---");
    SPECTRAL_LOG_INFO("Memory:  RSS %zu MB, Peak tracked %.1f MB",
                      end->current_resident_mb, BYTES_TO_MB(g_peak_alloc));
    SPECTRAL_LOG_INFO("CPU:     User %.1f ms, Sys %.1f ms, Total %.1f ms",
                      user_delta, sys_delta, total_cpu);
    SPECTRAL_LOG_INFO("Threads: %d / %d cores, Util %.1f%%, Parallelism %.2fx",
                      n_threads, end->num_cores, utilization, parallelism);
}
