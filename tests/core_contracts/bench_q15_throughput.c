/* bench_q15_throughput.c - measure-first perf probe for the opt-in Q15 path (Q3b #75).
 *
 * NOT a ctest (timing is machine-dependent): a standalone manual probe that
 * renders the SAME segment through production timbre_synth_segment() under three
 * configs and prints ns/sample, so the slice-2 SIMD-kernel decision is made on
 * data, not a guess:
 *
 *   float-scalar : spectral_osc_set_dispatch(ALL_SCALAR), Q15 off  (apples-to-apples ref)
 *   q15-scalar   : spectral_osc_set_dispatch(ALL_SCALAR), Q15 on   (the scalar Q15 oracle)
 *   float-simd   : spectral_osc_set_dispatch(ALL_SIMD),   Q15 off  (real-world default)
 *
 * The Q15 path is scalar regardless of dispatch (dispatch only steers the float
 * path), so float-scalar vs q15-scalar is the honest equal-vectorization compare.
 * Float-simd is the shipping baseline a Q15 SIMD kernel would have to beat. The
 * key question this answers: does Q15-eval-over-float-phase (which adds a
 * float->Q15->float round-trip per sample) actually buy throughput for the cheap
 * algebraic timbres, or only plausibly for sine (LUT vs sinf)?
 *
 * Run: cmake --build build --target bench_q15_throughput \
 *      && build/bench_q15_throughput
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "spectral_oscillator.h"
#include "spectral_oscillator_dispatch.h"
#include "spectral_synth_internal.h"

#define SEG_LEN  2048   /* sustain-dominated single segment */
#define ITERS    20000  /* ~40M samples/measurement -> stable timing */

typedef struct { SpectralTimbre timbre; const char* name; } BenchTimbre;
static const BenchTimbre k_timbres[] = {
    { TIMBRE_SINE,     "sine"     },
    { TIMBRE_SAW,      "saw"      },
    { TIMBRE_SQUARE,   "square"   },
    { TIMBRE_TRIANGLE, "triangle" },
    { TIMBRE_PARABOLA, "parabola" },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

static void fill_lp(SegmentLoopParams* lp) {
    memset(lp, 0, sizeof(*lp));
    lp->start_idx = 0;
    lp->length    = SEG_LEN;
    lp->alpha     = 0.0576f;   /* ~440 Hz @ 48k */
    lp->beta      = 0.0f;
    lp->c2        = 0.0f;
    lp->c3        = 0.0f;
    lp->phase     = 0.30f;
    lp->amp       = 0.80f;
    lp->d_amp     = -2.0e-4f;
    lp->width     = 0.0f;
    lp->valid     = 1;
}

/* Returns ns/sample; folds buf into *sink to defeat dead-code elimination. */
static double bench(SpectralTimbre timbre, double* sink) {
    SegmentLoopParams lp;
    fill_lp(&lp);
    float buf[SEG_LEN];

    /* Warm up (and prime caches / sine LUT if enabled). */
    for (int it = 0; it < 64; it++) {
        memset(buf, 0, sizeof(buf));
        timbre_synth_segment(buf, &lp, timbre);
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double acc = 0.0;
    for (int it = 0; it < ITERS; it++) {
        memset(buf, 0, sizeof(buf));
        timbre_synth_segment(buf, &lp, timbre);
        acc += buf[SEG_LEN / 2];   /* touch output */
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *sink += acc;

    double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);
    return ns / ((double)ITERS * (double)SEG_LEN);
}

int main(void) {
    double sink = 0.0;
    printf("== opt-in Q15 throughput probe (ns/sample, %d-sample segment, %d iters) ==\n",
           SEG_LEN, ITERS);
    printf("  %-9s %12s %12s %12s   %10s %10s\n",
           "timbre", "float-scl", "q15-scl", "float-simd",
           "q15/fscl", "q15/fsimd");

    for (size_t t = 0; t < N_TIMBRES; t++) {
        SpectralTimbre tb = k_timbres[t].timbre;

        spectral_osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR);
        spectral_osc_set_q15_enable(0);
        double f_scalar = bench(tb, &sink);

        spectral_osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR);
        spectral_osc_set_q15_enable(OSC_Q15_BIT(tb));
        double q_scalar = bench(tb, &sink);
        spectral_osc_set_q15_enable(0);

        spectral_osc_set_dispatch(OSC_DISPATCH_ALL_SIMD);
        double f_simd = bench(tb, &sink);

        printf("  %-9s %12.3f %12.3f %12.3f   %9.2fx %9.2fx\n",
               k_timbres[t].name, f_scalar, q_scalar, f_simd,
               q_scalar / f_scalar, q_scalar / f_simd);
    }

    printf("(ratios < 1.00x = Q15 faster; sink=%.3e)\n", sink);
    return 0;
}
