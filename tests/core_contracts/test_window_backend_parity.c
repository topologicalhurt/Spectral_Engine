/* test_window_backend_parity.c - the analysis windows must be backend-invariant
 * and match the documented symmetric (N-1 denominator) raised-cosine SSOT.
 *
 * Regression for a confirmed defect: the vDSP build used Apple's
 * vDSP_hann_window/hamm/blkman, which emit the PERIODIC (2*pi*n/N) convention,
 * silently diverging from the symmetric (2*pi*n/(N-1)) form documented in
 * spectral_windows.h / CORE_CONTRACTS.md and used by every other backend. The
 * divergence flowed into magsq/leakage in the live STFT path with no parity
 * test to catch it.
 *
 * This test is compiled with SPECTRAL_USE_VDSP=1 (the desktop default) on
 * purpose: if the vDSP window generators are ever reintroduced, the window
 * becomes periodic and these assertions against the symmetric reference fail.
 *
 * Run: cmake --build build --target window_backend_parity_test
 *      && ctest --test-dir build -R window_backend_parity
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "spectral_windows.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../support/check.h"

/* Documented SSOT references in double precision, symmetric (N-1) with the
 * canonical textbook coefficients — so a convention change (N vs N-1) OR a
 * coefficient drift both fail. */
static double ref_hann(size_t i, size_t N) {
    return 0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (double)(N - 1)));
}
static double ref_hamming(size_t i, size_t N) {
    return 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / (double)(N - 1));
}
static double ref_blackman(size_t i, size_t N) {
    double a = 2.0 * M_PI * (double)i / (double)(N - 1);
    return 0.42 - 0.5 * cos(a) + 0.08 * cos(2.0 * a);
}

typedef void (*window_fn)(float*, size_t);
typedef double (*ref_fn)(size_t, size_t);

static void compare(const char* name, window_fn gen, ref_fn ref) {
    /* The symmetric vs periodic gap is largest at small N; sweep a few sizes. */
    const size_t sizes[] = { 16, 17, 64, 256, 1024 };
    const double tol = 1e-5; /* float-eval slack; periodic gap is O(1e-3), far above */
    printf("%s:\n", name);
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        size_t N = sizes[s];
        float* w = (float*)malloc(N * sizeof(float));
        if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }
        gen(w, N);
        double max_dev = 0.0;
        size_t worst = 0;
        for (size_t i = 0; i < N; i++) {
            double dev = fabs((double)w[i] - ref(i, N));
            if (dev > max_dev) { max_dev = dev; worst = i; }
        }
        CHECK(max_dev <= tol,
              "%s N=%zu deviates from symmetric SSOT by %.3g at i=%zu "
              "(periodic-convention regression?)", name, N, max_dev, worst);
        /* Endpoints: symmetric Hann/Blackman are exactly 0 at i=0 and i=N-1;
         * the periodic form is NOT zero at i=N-1. This is the sharpest tell. */
        free(w);
    }
}

/* Endpoint contract: symmetric raised-cosine windows are 0 at both ends (Hann,
 * Blackman) — the periodic convention's last sample is nonzero. */
static void test_endpoints_symmetric(void) {
    printf("test_endpoints_symmetric:\n");
    const size_t N = 64;
    float* w = (float*)malloc(N * sizeof(float));
    if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }

    spectral_window_hann(w, N);
    CHECK(fabs((double)w[0]) < 1e-6, "hann[0] must be 0, got %.6g", (double)w[0]);
    CHECK(fabs((double)w[N - 1]) < 1e-6, "hann[N-1] must be 0 (symmetric), got %.6g", (double)w[N - 1]);

    spectral_window_blackman(w, N);
    CHECK(fabs((double)w[N - 1]) < 1e-6, "blackman[N-1] must be ~0 (symmetric), got %.6g", (double)w[N - 1]);
    free(w);
}

int main(void) {
    compare("test_hann_matches_symmetric", spectral_window_hann, ref_hann);
    compare("test_hamming_matches_symmetric", spectral_window_hamming, ref_hamming);
    compare("test_blackman_matches_symmetric", spectral_window_blackman, ref_blackman);
    test_endpoints_symmetric();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
