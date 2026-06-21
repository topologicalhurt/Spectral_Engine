/* spectral_omp.h - OpenMP Compatibility Shim
 *
 * When _OPENMP is defined, simply includes <omp.h>.
 * Otherwise provides stub implementations so code compiles
 * single-threaded without #ifdef blocks everywhere.
 *
 * Include this instead of <omp.h> in all source files.
 */
#ifndef SPECTRAL_OMP_H
#define SPECTRAL_OMP_H

#include "spectral_config.h"

#ifdef _OPENMP
#include <omp.h>
#else

#include <time.h>

static inline int    omp_get_max_threads(void) { return 1; }
static inline int    omp_get_thread_num(void)  { return 0; }
static inline void   omp_set_num_threads(int n) { (void)n; }

static inline double omp_get_wtime(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
    return 0.0;
}

#endif /* _OPENMP */

static inline int spectral_omp_effective_thread_count(void)
{
    int n_threads = omp_get_max_threads();

    if (n_threads < 1) return 1;
    if (n_threads > SPECTRAL_MAX_THREADS) return SPECTRAL_MAX_THREADS;
    return n_threads;
}

#endif /* SPECTRAL_OMP_H */
