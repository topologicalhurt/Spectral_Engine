/* test_ifft_synth_parity.c - F2 contract test for the IFFT synthesis path.
 *
 * Three rungs (IFFT_SYNTHESIS_PLAN parity ladder, now over the REAL float
 * path with the build's live iFFT port — vDSP on Apple, portable radix-2
 * elsewhere):
 *   1. Backend contract: inverse of a random Hermitian half-spectrum matches
 *      a double-precision O(N^2) reference iDFT (catches packing/scale bugs
 *      in either port immediately).
 *   2. Stream parity: 64 dense stationary partials rendered via
 *      spectral_ifft_synth_render() vs the exact double oscillator sum —
 *      must hit the F1-measured floors (-55 dBFS frame class at K=8) with
 *      float headroom: max <= -60 dBFS... see asserts (F1 floors -68 max /
 *      -83 RMS in double; float port budgeted ~8 dB).
 *   3. Determinism: two renders are byte-identical.
 * Also prints a MEASURED throughput comparison vs a plain float oscillator
 * loop (informational; perf gating lives in the S4 bench world, not ctest).
 *
 * Run: cmake --build build --target ifft_synth_parity_test
 *      && ctest --test-dir build -R ifft_synth_parity
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "spectral_consts.h"
#include "spectral_synth_ifft.h"

#define N_FFT 512u
#define TOTAL 8192u
#define N_PARTIALS 64u

#include "../support/check.h"

#include "../support/xorshift_rng.h"

static unsigned rng_state = 0x5eed5eedu;

static double dbfs(double x) { return 20.0 * log10(x > 1e-300 ? x : 1e-300); }

static uint64_t now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Rung 1: the port's inverse vs a double reference iDFT. */
static void test_backend_contract(void) {
    printf("test_backend_contract:\n");
    SpectralIfftBackend* b = spectral_ifft_backend_create(N_FFT);
    CHECK(b != NULL, "backend create");
    if (!b) return;

    float re[N_FFT / 2], im[N_FFT / 2], out[N_FFT];
    double fre[N_FFT], fim[N_FFT];
    memset(fre, 0, sizeof fre); memset(fim, 0, sizeof fim);

    re[0] = (float)(xorshift32_unit(&rng_state) - 0.5);              /* DC */
    im[0] = (float)(xorshift32_unit(&rng_state) - 0.5);              /* Nyquist (packed) */
    fre[0] = re[0];
    fre[N_FFT / 2] = im[0];
    for (size_t k = 1; k < N_FFT / 2; k++) {
        re[k] = (float)(xorshift32_unit(&rng_state) - 0.5);
        im[k] = (float)(xorshift32_unit(&rng_state) - 0.5);
        fre[k] = re[k];           fim[k] = im[k];
        fre[N_FFT - k] = re[k];   fim[N_FFT - k] = -im[k];   /* Hermitian */
    }

    spectral_ifft_backend_inverse(b, re, im, out);

    double worst = 0.0;
    for (size_t n = 0; n < N_FFT; n++) {
        double acc = 0.0;
        for (size_t k = 0; k < N_FFT; k++) {
            double a = 2.0 * SPECTRAL_PI_D * (double)k * (double)n / (double)N_FFT;
            acc += fre[k] * cos(a) - fim[k] * sin(a);
        }
        acc /= (double)N_FFT;
        double err = fabs((double)out[n] - acc);
        if (err > worst) worst = err;
    }
    printf("  backend max err vs reference iDFT: %.3e\n", worst);
    CHECK(worst < 1e-4, "port inverse must match the textbook iDFT (got %.3e)", worst);
    spectral_ifft_backend_destroy(b);
}

/* Rungs 2+3: stream parity + determinism + measured throughput. */
static void test_stream_parity(void) {
    printf("test_stream_parity:\n");
    SpectralIfftSynth* s = spectral_ifft_synth_create(N_FFT);
    CHECK(s != NULL, "synth create");
    if (!s) return;

    static SpectralIfftPartial parts[N_PARTIALS];
    rng_state = 0xc0ffee11u;
    for (size_t p = 0; p < N_PARTIALS; p++) {
        parts[p].bin = (float)(10.0 + 230.0 * xorshift32_unit(&rng_state));
        parts[p].amp = (float)((0.05 + 0.95 * xorshift32_unit(&rng_state)) / N_PARTIALS);
        parts[p].phase0 = (float)(2.0 * SPECTRAL_PI_D * xorshift32_unit(&rng_state) - SPECTRAL_PI_D);
    }

    static float out[TOTAL], out2[TOTAL];
    uint64_t t0 = now_ns();
    CHECK(spectral_ifft_synth_render(s, parts, N_PARTIALS, out, TOTAL) == 0, "render rc");
    uint64_t t_ifft = now_ns() - t0;

    /* Exact reference: double oscillator sum. */
    double worst = 0.0, sumsq = 0.0;
    for (size_t t = 0; t < TOTAL; t++) {
        double want = 0.0;
        for (size_t p = 0; p < N_PARTIALS; p++) {
            double omega = 2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT;
            want += (double)parts[p].amp * cos(omega * (double)t + (double)parts[p].phase0);
        }
        double err = fabs((double)out[t] - want);
        if (err > worst) worst = err;
        sumsq += err * err;
    }
    double rms = sqrt(sumsq / TOTAL);
    printf("  stream parity: max=%.2f dBFS rms=%.2f dBFS (%u partials)\n",
           dbfs(worst), dbfs(rms), (unsigned)N_PARTIALS);
    CHECK(dbfs(worst) < -60.0, "stream max err must beat -60 dBFS (got %.2f)", dbfs(worst));
    CHECK(dbfs(rms) < -75.0, "stream rms err must beat -75 dBFS (got %.2f)", dbfs(rms));

    /* Determinism. */
    CHECK(spectral_ifft_synth_render(s, parts, N_PARTIALS, out2, TOTAL) == 0, "render rc 2");
    CHECK(memcmp(out, out2, sizeof out) == 0, "render must be deterministic");

    /* MEASURED throughput vs a plain float oscillator loop (info only). */
    {
        static float ref[TOTAL];
        uint64_t t1 = now_ns();
        memset(ref, 0, sizeof ref);
        for (size_t p = 0; p < N_PARTIALS; p++) {
            float omega = (float)(2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT);
            float ph = parts[p].phase0;
            for (size_t t = 0; t < TOTAL; t++) {
                ref[t] += parts[p].amp * cosf(omega * (float)t + ph);
            }
        }
        uint64_t t_osc = now_ns() - t1;

        /* Volatile sink AFTER timing: every element feeds the reduction, so
         * the rendered buffer stays live without perturbing the timed loop. */
        {
            static volatile float sink;
            float acc = 0.0f;
            for (size_t t = 0; t < TOTAL; t++) acc += ref[t];
            sink = acc;
            (void)sink;
        }
        printf("  MEASURED: ifft %.2f ms vs naive-osc %.2f ms -> %.1fx "
               "(info; not a gate)\n",
               t_ifft / 1e6, t_osc / 1e6,
               t_ifft > 0 ? (double)t_osc / (double)t_ifft : 0.0);
    }

    spectral_ifft_synth_destroy(s);
}

int main(void) {
    test_backend_contract();
    test_stream_parity();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
