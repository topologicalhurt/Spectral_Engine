/* test_peak_interp_parabolic.c - the real log-power parabolic peak interpolator
 * (spectral_window_interp_magsq_parabolic), exercising BOTH its fast two-log
 * branch and its overflow three-log fallback against a double-precision
 * reference.
 *
 * Replaces a tautological Python test that compared two algebraically-identical
 * Python helpers (and never touched the C estimator). Here the shipping C
 * function is driven directly, including extreme magnitude triplets that force
 * the fast path's denom non-finite so the three-log fallback executes — the
 * branch the old test only implied it covered.
 *
 * Run: cmake --build build --target peak_interp_parabolic_test
 *      && ctest --test-dir build -R peak_interp_parabolic
 */
#include <math.h>
#include <stdio.h>

#include "spectral_windows.h"

#include "../support/check.h"

#define LOG_FLOOR 1.0e-30
#define DENOM_EPS 1.0e-20

/* Double-precision reference: Smith's three-log quadratic peak offset with the
 * same magnitude floor and [-0.5, 0.5] clamp the C code uses. Independent of
 * the engine's fast_peak_log, so it protects against wrong-formula / wrong-sign
 * / wrong-branch regressions (within the tolerance for the fast-log approx). */
static double ref_parabolic(double left, double center, double right) {
    double l = fmax(left, LOG_FLOOR), c = fmax(center, LOG_FLOOR), r = fmax(right, LOG_FLOOR);
    double denom = log(l) - 2.0 * log(c) + log(r);
    if (!isfinite(denom) || fabs(denom) < DENOM_EPS) return 0.0;
    double p = 0.5 * (log(l) - log(r)) / denom;
    if (p > 0.5) return 0.5;
    if (p < -0.5) return -0.5;
    return p;
}

/* Symmetric triplets (left==right) must give EXACTLY 0 regardless of the log
 * approximation: the numerator log(l)-log(r) is identically 0. */
static void test_symmetric_is_zero(void) {
    printf("test_symmetric_is_zero:\n");
    const float vals[] = { 0.01f, 0.5f, 1.0f, 4.0f, 100.0f };
    for (int ci = 0; ci < 5; ci++) {
        for (int si = 0; si < 5; si++) {
            float side = vals[si], center = vals[ci];
            float p = spectral_window_interp_magsq_parabolic(side, center, side);
            CHECK(fabsf(p) < 1e-6f, "symmetric (l=r=%.3g,c=%.3g) must be 0, got %.6g",
                  (double)side, (double)center, (double)p);
        }
    }
}

/* Normal asymmetric triplets exercise the fast two-log branch; compare to the
 * double reference within a tolerance that covers fast_peak_log's accuracy. */
static void test_fast_branch_matches_reference(void) {
    printf("test_fast_branch_matches_reference:\n");
    const float vals[] = { 0.05f, 0.3f, 1.0f, 3.0f, 25.0f };
    const double tol = 5e-3; /* fast_peak_log approximation slack */
    for (int li = 0; li < 5; li++)
        for (int ci = 0; ci < 5; ci++)
            for (int ri = 0; ri < 5; ri++) {
                float l = vals[li], c = vals[ci], r = vals[ri];
                float p = spectral_window_interp_magsq_parabolic(l, c, r);
                double ref = ref_parabolic(l, c, r);
                CHECK(isfinite(p) && p >= -0.5f && p <= 0.5f,
                      "fast (l=%.3g,c=%.3g,r=%.3g) must be finite in [-.5,.5], got %.6g",
                      (double)l, (double)c, (double)r, (double)p);
                CHECK(fabs((double)p - ref) <= tol,
                      "fast (l=%.3g,c=%.3g,r=%.3g): C %.6g vs ref %.6g (diff %.3g)",
                      (double)l, (double)c, (double)r, (double)p, ref, fabs((double)p - ref));
            }
}

/* Extreme triplets: center clamped to the floor while sides are huge makes the
 * fast path's left/center overflow to +Inf, so log(Inf) makes denom non-finite
 * and the three-log fallback executes. The result must still be finite, bounded
 * and track the reference (left>right => peak biased toward left => p<0). */
static void test_overflow_forces_fallback(void) {
    printf("test_overflow_forces_fallback:\n");
    struct { float l, c, r; } cases[] = {
        { 1.0e30f, 1.0e-30f, 1.0e30f },   /* symmetric extreme -> 0 */
        { 1.0e30f, 1.0e-30f, 1.0e20f },   /* left heavier -> p < 0 */
        { 1.0e20f, 1.0e-30f, 1.0e30f },   /* right heavier -> p > 0 */
        { 3.0e38f, 1.0e-30f, 1.0e10f },   /* near float max */
    };
    for (int i = 0; i < 4; i++) {
        float l = cases[i].l, c = cases[i].c, r = cases[i].r;
        /* Confirm this input really does overflow the fast path's ratio. */
        CHECK(!isfinite((double)(l / c)) || (double)(l / c) > 1e38,
              "case %d should drive left/center toward overflow", i);
        float p = spectral_window_interp_magsq_parabolic(l, c, r);
        double ref = ref_parabolic(l, c, r);
        CHECK(isfinite(p) && p >= -0.5f && p <= 0.5f,
              "fallback case %d must be finite in [-.5,.5], got %.6g", i, (double)p);
        CHECK(fabs((double)p - ref) <= 5e-3,
              "fallback case %d: C %.6g vs ref %.6g", i, (double)p, ref);
    }
}

/* Degenerate inputs must never produce NaN/Inf or escape the bin clamp. */
static void test_degenerate_inputs_bounded(void) {
    printf("test_degenerate_inputs_bounded:\n");
    float ps[] = {
        spectral_window_interp_magsq_parabolic(0.0f, 0.0f, 0.0f),
        spectral_window_interp_magsq_parabolic(1.0f, 1.0f, 1.0f),
        spectral_window_interp_magsq_parabolic(INFINITY, 1.0f, 1.0f),
        spectral_window_interp_magsq_parabolic(1.0f, INFINITY, 1.0f),
        spectral_window_interp_magsq_parabolic(0.0f, 1.0f, 0.0f),
    };
    for (int i = 0; i < 5; i++)
        CHECK(isfinite(ps[i]) && ps[i] >= -0.5f && ps[i] <= 0.5f,
              "degenerate case %d must be finite in [-.5,.5], got %.6g", i, (double)ps[i]);
}

int main(void) {
    test_symmetric_is_zero();
    test_fast_branch_matches_reference();
    test_overflow_forces_fallback();
    test_degenerate_inputs_bounded();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
