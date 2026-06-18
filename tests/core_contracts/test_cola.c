/* test_cola.c - COLA/WOLA reconstruction-invariant contract test (CAMPAIGN_2_MASTER_PLAN B0).
 *
 * Exercises spectral_overlap_add_is_constant / _envelope_stats (the
 * Constant/Weighted OverLap-Add invariant in spectral_contracts.h) against:
 *   - a constructed PERIODIC window (the form that satisfies COLA), and
 *   - the engine's own spectral_window_* generators (which are SYMMETRIC,
 *     N-1 denominator, "unnormalized conventional window shapes" per
 *     spectral_windows.h) so the test records exactly where the shipping windows
 *     do and do not reconstruct gain-flat.
 *
 * This is the prerequisite for the Phase-B guarantee registry: the registry
 * cannot record "we relaxed COLA" until COLA is first defined and tested.
 * References: Allen (1977); Allen & Rabiner (1977); Griffin & Lim (1984);
 * Harris (1978) for window shapes. See docs/core_audit/reference/CORE_CONTRACTS.md.
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "spectral_contracts.h"
#include "spectral_windows.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../support/check.h"

/* Periodic (DFT-even) Hann: w[n] = 0.5*(1 - cos(2*pi*n/N)). Unlike the symmetric
 * N-1 form, this satisfies COLA at hop = N/2, N/4, ... exactly. */
static void periodic_hann(float* w, size_t n) {
    for (size_t i = 0; i < n; i++) {
        w[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * (double)i / (double)n)));
    }
}

static double rel_dev(const float* a, const float* s, size_t n, size_t hop) {
    double mn = 0.0, mx = 0.0, mean = 0.0;
    if (!spectral_overlap_add_envelope_stats(a, s, n, hop, &mn, &mx, &mean)) return -1.0;
    if (mean <= 0.0) return -1.0;
    return (mx - mn) / mean;
}

static void test_periodic_hann_cola(void) {
    const size_t N = 1024;
    float* w = (float*)malloc(N * sizeof(float));
    double gain = 0.0;
    printf("test_periodic_hann_cola:\n");
    if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }
    periodic_hann(w, N);

    /* hop = N/2: COLA gain is exactly 1. */
    CHECK(spectral_overlap_add_is_constant(w, NULL, N, N / 2, 1e-5, &gain),
          "periodic Hann should satisfy COLA at hop N/2");
    CHECK(fabs(gain - 1.0) < 1e-4, "COLA gain at hop N/2 should be ~1.0 (got %.6f)", gain);

    /* hop = N/4: still COLA, gain = 2. */
    gain = 0.0;
    CHECK(spectral_overlap_add_is_constant(w, NULL, N, N / 4, 1e-5, &gain),
          "periodic Hann should satisfy COLA at hop N/4");
    CHECK(fabs(gain - 2.0) < 1e-4, "COLA gain at hop N/4 should be ~2.0 (got %.6f)", gain);
    free(w);
}

static void test_rectangular_cola(void) {
    const size_t N = 1024;
    float* w = (float*)malloc(N * sizeof(float));
    double gain = 0.0;
    printf("test_rectangular_cola:\n");
    if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }
    spectral_window_rectangular(w, N);

    /* Rectangular trivially COLA for any hop dividing N; gain = N/hop. */
    CHECK(spectral_overlap_add_is_constant(w, NULL, N, N / 4, 1e-6, &gain),
          "rectangular should satisfy COLA at hop N/4");
    CHECK(fabs(gain - 4.0) < 1e-5, "rectangular COLA gain at hop N/4 should be 4.0 (got %.6f)", gain);
}

static void test_wola_self_window(void) {
    /* WOLA with analysis == synthesis (Griffin-Lim sum-of-squares condition):
     * periodic Hann squared also overlap-adds to a constant. */
    const size_t N = 1024;
    float* w = (float*)malloc(N * sizeof(float));
    double gain = 0.0;
    printf("test_wola_self_window:\n");
    if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }
    periodic_hann(w, N);

    CHECK(spectral_overlap_add_is_constant(w, w, N, N / 4, 1e-5, &gain),
          "periodic Hann^2 (WOLA, analysis==synthesis) should be constant at hop N/4");
    CHECK(gain > 0.0, "WOLA self-window gain should be positive (got %.6f)", gain);
    free(w);
}

static void test_symmetric_engine_hann(void) {
    /* The engine's spectral_window_hann is SYMMETRIC (N-1 denominator). It does
     * NOT satisfy strict COLA, but the deviation is O(1/N). We record that:
     *   - it FAILS at a tight tolerance, and
     *   - it PASSES at a loose tolerance, quantifying the actual error. */
    const size_t N = 1024;
    float* w = (float*)malloc(N * sizeof(float));
    double dev = 0.0;
    printf("test_symmetric_engine_hann:\n");
    if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }
    spectral_window_hann(w, N);

    dev = rel_dev(w, NULL, N, N / 2);
    printf("  symmetric Hann hop N/2: relative envelope deviation = %.3e\n", dev);
    CHECK(dev >= 0.0, "envelope stats should succeed for the engine Hann");
    CHECK(!spectral_overlap_add_is_constant(w, NULL, N, N / 2, 1e-6, NULL),
          "symmetric Hann should FAIL strict COLA (tol 1e-6)");
    CHECK(spectral_overlap_add_is_constant(w, NULL, N, N / 2, 1e-2, NULL),
          "symmetric Hann should pass loose COLA (tol 1e-2)");
    /* The N-1 form's deviation is ~ a few / N; assert the order of magnitude so a
     * future window change that worsens it is caught. */
    CHECK(dev < 1e-2, "symmetric Hann deviation should be O(1/N) small (got %.3e)", dev);
    free(w);
}

static void test_negative_cases(void) {
    const size_t N = 1024;
    float* w = (float*)malloc(N * sizeof(float));
    printf("test_negative_cases:\n");
    if (!w) { printf("  FAIL: oom\n"); g_fail = 1; return; }
    periodic_hann(w, N);

    /* hop does not divide length -> rejected. */
    CHECK(!spectral_overlap_add_is_constant(w, NULL, N, 300, 1e-2, NULL),
          "non-dividing hop should be rejected");
    /* hop == length (no overlap): Hann endpoints are ~0 -> envelope hits 0,
     * not constant -> not COLA. */
    CHECK(!spectral_overlap_add_is_constant(w, NULL, N, N, 1e-2, NULL),
          "no-overlap (hop==N) Hann should not satisfy COLA");
    /* NULL analysis / zero length / zero hop -> rejected. */
    CHECK(!spectral_overlap_add_is_constant(NULL, NULL, N, N / 2, 1e-2, NULL),
          "NULL analysis should be rejected");
    CHECK(!spectral_overlap_add_is_constant(w, NULL, 0, 0, 1e-2, NULL),
          "zero length/hop should be rejected");
    free(w);
}

int main(void) {
    test_periodic_hann_cola();
    test_rectangular_cola();
    test_wola_self_window();
    test_symmetric_engine_hann();
    test_negative_cases();

    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
