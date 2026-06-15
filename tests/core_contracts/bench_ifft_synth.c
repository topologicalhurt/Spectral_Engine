/* bench_ifft_synth.c - measure-first speedup probe: IFFT synthesis vs the naive
 * per-partial oscillator loop, swept over partial count (F-stream F2/F2b).
 *
 * NOT a ctest (timing is machine-dependent): a standalone manual probe. It renders
 * the SAME dense stationary-sine set both ways and prints ns/sample + the speedup,
 * so the ~8x claim is MEASURED, not asserted, and the crossover (below which the
 * oscillator loop is cheaper than the per-frame FFT overhead) is visible. The IFFT
 * cost is ~constant per frame; the oscillator cost scales with the partial count,
 * so the speedup grows with density.
 *
 * This is the DESKTOP probe (host iFFT port = vDSP on Apple, portable radix-2
 * elsewhere). The embedded M7 speedup is a separate measurement in cycles via the
 * perf model — the host FFT is not the M7 FFT, so the M7 ratio must be measured on
 * its own, not inferred from this number.
 *
 * Run: cmake --build build --target bench_ifft_synth && build/bench_ifft_synth
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "spectral_consts.h"
#include "spectral_synth_ifft.h"

#include "../support/xorshift_rng.h"

#define N_FFT   512u
#define TOTAL   8192u
#define ITERS   200      /* * TOTAL samples per measurement -> stable ns/sample */

static const unsigned k_counts[] = { 2, 4, 8, 16, 32, 64, 128 };
#define N_COUNTS (sizeof(k_counts) / sizeof(k_counts[0]))

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void) {
    SpectralIfftSynth* s = spectral_ifft_synth_create(N_FFT);
    if (!s) { printf("create failed\n"); return 1; }

    static SpectralIfftPartial parts[128];
    static float out[TOTAL];
    double sink = 0.0;

    printf("== IFFT synthesis vs naive oscillator loop ==\n");
    printf("   N_FFT=%u, %u samples/render, %d iters, host iFFT port\n\n",
           N_FFT, TOTAL, ITERS);
    printf("  %8s %14s %14s %12s\n", "partials", "osc ns/smp", "ifft ns/smp", "speedup");

    for (size_t c = 0; c < N_COUNTS; c++) {
        unsigned n = k_counts[c];

        unsigned rng = 0xc0ffee11u;
        for (unsigned p = 0; p < n; p++) {
            parts[p].bin = (float)(8.0 + 232.0 * ((double)p / (double)n)
                                   + 0.37 * xorshift32_unit(&rng));
            parts[p].amp = (float)((0.05 + 0.95 * xorshift32_unit(&rng)) / n);
            parts[p].phase0 = (float)(2.0 * SPECTRAL_PI_D * xorshift32_unit(&rng) - SPECTRAL_PI_D);
        }

        /* Warm up both paths (caches, the sine LUT, the FFT plan). */
        for (int w = 0; w < 16; w++) {
            spectral_ifft_synth_render(s, parts, n, out, TOTAL);
            for (unsigned p = 0; p < n; p++) {
                float omega = (float)(2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT);
                for (size_t t = 0; t < TOTAL; t++) out[t] += parts[p].amp * cosf(omega * (float)t + parts[p].phase0);
            }
        }

        /* Naive oscillator loop: amp*cos per partial per sample (what the IFFT replaces). */
        uint64_t t0 = now_ns();
        for (int it = 0; it < ITERS; it++) {
            for (size_t t = 0; t < TOTAL; t++) out[t] = 0.0f;
            for (unsigned p = 0; p < n; p++) {
                float omega = (float)(2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT);
                float ph = parts[p].phase0;
                for (size_t t = 0; t < TOTAL; t++) out[t] += parts[p].amp * cosf(omega * (float)t + ph);
            }
            sink += out[TOTAL / 2 + it % 7];
        }
        uint64_t osc_ns = now_ns() - t0;

        /* IFFT synthesis path. */
        t0 = now_ns();
        for (int it = 0; it < ITERS; it++) {
            spectral_ifft_synth_render(s, parts, n, out, TOTAL);
            sink += out[TOTAL / 2 + it % 7];
        }
        uint64_t ifft_ns = now_ns() - t0;

        double osc_per = (double)osc_ns / ((double)ITERS * (double)TOTAL);
        double ifft_per = (double)ifft_ns / ((double)ITERS * (double)TOTAL);
        printf("  %8u %14.3f %14.3f %11.2fx%s\n", n, osc_per, ifft_per,
               osc_per / ifft_per, (osc_per > ifft_per) ? "" : "   <- osc wins");
    }

    printf("\n(speedup > 1 = IFFT faster; sink=%.3e)\n", sink);
    spectral_ifft_synth_destroy(s);
    return 0;
}
