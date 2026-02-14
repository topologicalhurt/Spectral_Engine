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

#ifdef _OPENMP
#include <omp.h>
#else

#include <time.h>

static inline int    omp_get_max_threads(void) { return 1; }
static inline int    omp_get_thread_num(void)  { return 0; }
static inline void   omp_set_num_threads(int n) { (void)n; }

static inline double omp_get_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#endif /* _OPENMP */

#endif /* SPECTRAL_OMP_H */
