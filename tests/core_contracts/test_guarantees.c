/* test_guarantees.c - Active-guarantee manifest + drift contract (ULTRAPLAN B1/B2).
 *
 * Two responsibilities, both compiled (not string-grep):
 *
 *  B2 (self-report): the SPECTRAL_ACTIVE_GUARANTEES bitset and the runtime query
 *      API (spectral_active_guarantees / _guarantee_holds / _guarantees_satisfy)
 *      agree, are wired to the correct gate macro, and fail closed when a required
 *      guarantee is relaxed in this build.
 *
 *  B1 (manifest drift): each approximation gated by SPECTRAL_ENABLE_APPROX_* stays
 *      within its documented error budget versus the libm reference. The budgets
 *      below are the measured worst case rounded up ~2-3x; if an approximation's
 *      polynomial is retuned and drifts past budget, this fails.
 *
 * The SAME source is compiled twice (see cmake/targets/core-guarantees-test.cmake):
 *   core_guarantees_test        - default gates (all "exact" bits active).
 *   core_guarantees_drift_test  - APPROX_* forced on; asserts those bits cleared
 *                                  AND the approximations stay within budget.
 *
 * See docs/core_audit/CORE_CONTRACTS.md (Guarantee registry).
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "spectral_guarantees.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../support/check.h"

/* Measured worst case (build with all APPROX_* on):
 *   sin   8.26e-07 abs over [-4pi,4pi]  (degree-9 odd minimax quadrant fold;
 *                                        ~1.4 ULP over the [-pi,pi] operating
 *                                        range, the 4-period sweep adds reduction
 *                                        error the oscillator never sees)
 *   atan2 2.03e-04 rad
 *   rsqrt 4.71e-06 rel over [1e-3,1e3]
 *   log   6.80e-07 abs over [1e-4,1e4]
 * Budgets are loose multiples so float-FP variance across platforms is allowed
 * but a real polynomial regression is caught. */
#define BUDGET_SIN       2.0e-6
#define BUDGET_ATAN2     5.0e-4
#define BUDGET_RSQRT_REL 1.0e-5
#define BUDGET_PEAK_LOG  2.0e-6

static void expect_bit(uint32_t bit, int gate_holds, const char* name) {
    CHECK(spectral_guarantee_holds(bit) == (gate_holds ? 1 : 0),
          "guarantee bit '%s' = %d but gate says %d",
          name, spectral_guarantee_holds(bit), gate_holds ? 1 : 0);
}

static void test_bitset_wiring(void) {
    printf("SPECTRAL_ACTIVE_GUARANTEES = 0x%02x\n",
           (unsigned)spectral_active_guarantees());

    /* compile-time set and runtime query must agree */
    CHECK(spectral_active_guarantees() == (uint32_t)SPECTRAL_ACTIVE_GUARANTEES,
          "runtime active set != compile-time SPECTRAL_ACTIVE_GUARANTEES");
    CHECK((spectral_active_guarantees() & ~SPECTRAL_GUARANTEE_ALL_MASK) == 0u,
          "active set has bits outside ALL_MASK");

    /* each bit is wired to the gate the source actually branches on */
    expect_bit(SPECTRAL_GUARANTEE_IEEE_STRICT_FP,
               SPECTRAL_CUSTOM_FAST_MATH_MODE == 0, "ieee_strict_fp");
    expect_bit(SPECTRAL_GUARANTEE_EXACT_TRIG,
               SPECTRAL_ENABLE_APPROX_TRIG == 0, "exact_trig");
    expect_bit(SPECTRAL_GUARANTEE_EXACT_ATAN2,
               SPECTRAL_ENABLE_APPROX_ATAN2 == 0, "exact_atan2");
    expect_bit(SPECTRAL_GUARANTEE_EXACT_INV_SQRT,
               SPECTRAL_ENABLE_APPROX_INV_SQRT == 0, "exact_inv_sqrt");
    expect_bit(SPECTRAL_GUARANTEE_EXACT_PEAK_LOG,
               SPECTRAL_ENABLE_APPROX_PEAK_LOG == 0, "exact_peak_log");
    expect_bit(SPECTRAL_GUARANTEE_EXACT_GPU_FP,
               SPECTRAL_METAL_FAST_MATH == 0, "exact_gpu_fp");
    expect_bit(SPECTRAL_GUARANTEE_DETERMINISTIC_REDUCTION,
               SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS > 0, "deterministic_reduction");
}

static void test_fail_closed(void) {
    uint32_t active = spectral_active_guarantees();

    CHECK(spectral_guarantees_satisfy(0u) == 1, "empty requirement must be satisfied");
    CHECK(spectral_guarantees_satisfy(active) == 1, "active set must satisfy itself");
    CHECK(spectral_guarantee_holds(0u) == 0, "zero mask must never hold");

    /* fail-closed: requiring any relaxed guarantee must return 0 */
    uint32_t inactive = SPECTRAL_GUARANTEE_ALL_MASK & ~active;
    if (inactive != 0u) {
        uint32_t b = inactive & (uint32_t)(-(int32_t)inactive); /* lowest set bit */
        CHECK(spectral_guarantees_satisfy(active | b) == 0,
              "requiring relaxed guarantee 0x%02x must fail closed", (unsigned)b);
        CHECK(spectral_guarantee_holds(b) == 0, "relaxed guarantee must not hold");
    } else {
        printf("  note: all guarantees active in this build; fail-closed path"
               " is exercised by core_guarantees_drift_test\n");
    }

    /* descriptor table covers exactly the manifest bits */
    size_t n = 0;
    const spectral_guarantee_info_t* tbl = spectral_guarantee_table(&n);
    CHECK(n == (size_t)SPECTRAL_GUARANTEE_COUNT, "table has %zu rows, expected %u",
          n, (unsigned)SPECTRAL_GUARANTEE_COUNT);
    uint32_t union_bits = 0u;
    for (size_t i = 0; i < n; i++) union_bits |= tbl[i].bit;
    CHECK(union_bits == SPECTRAL_GUARANTEE_ALL_MASK,
          "table bit union 0x%02x != ALL_MASK 0x%02x",
          (unsigned)union_bits, (unsigned)SPECTRAL_GUARANTEE_ALL_MASK);
}

static void test_drift(void) {
    double m;

    /* sin over [-4pi, 4pi] */
    m = 0.0;
    for (int i = 0; i <= 200000; i++) {
        double t = -4.0 * M_PI + 8.0 * M_PI * (double)i / 200000.0;
        double e = fabs((double)fast_sin((float)t) - sin(t));
        if (e > m) m = e;
    }
    printf("fast_sin      max abs err = %.3e (budget %.1e)\n", m, BUDGET_SIN);
    CHECK(m <= BUDGET_SIN, "fast_sin drift %.3e > budget %.1e", m, BUDGET_SIN);

    /* atan2 over a grid (skip origin) */
    m = 0.0;
    for (int yi = -300; yi <= 300; yi++) {
        for (int xi = -300; xi <= 300; xi++) {
            if (xi == 0 && yi == 0) continue;
            double y = (double)yi / 30.0, x = (double)xi / 30.0;
            double e = fabs((double)fast_atan2((float)y, (float)x) - atan2(y, x));
            if (e > m) m = e;
        }
    }
    printf("fast_atan2    max abs err = %.3e rad (budget %.1e)\n", m, BUDGET_ATAN2);
    CHECK(m <= BUDGET_ATAN2, "fast_atan2 drift %.3e > budget %.1e", m, BUDGET_ATAN2);

    /* inv_sqrt relative over [1e-3, 1e3] */
    m = 0.0;
    for (int i = 1; i <= 200000; i++) {
        double x = 1e-3 + (1e3 - 1e-3) * (double)i / 200000.0;
        double ref = 1.0 / sqrt(x);
        double e = fabs(((double)fast_inv_sqrt((float)x) - ref) / ref);
        if (e > m) m = e;
    }
    printf("fast_inv_sqrt max rel err = %.3e (budget %.1e)\n", m, BUDGET_RSQRT_REL);
    CHECK(m <= BUDGET_RSQRT_REL, "fast_inv_sqrt drift %.3e > budget %.1e", m, BUDGET_RSQRT_REL);

    /* peak_log abs over [1e-4, 1e4] */
    m = 0.0;
    for (int i = 1; i <= 200000; i++) {
        double x = 1e-4 + (1e4 - 1e-4) * (double)i / 200000.0;
        double e = fabs((double)fast_peak_log((float)x) - log(x));
        if (e > m) m = e;
    }
    printf("fast_peak_log max abs err = %.3e (budget %.1e)\n", m, BUDGET_PEAK_LOG);
    CHECK(m <= BUDGET_PEAK_LOG, "fast_peak_log drift %.3e > budget %.1e", m, BUDGET_PEAK_LOG);
}

/* Large accumulated phase (long sustained segments: phase = phase0 + alpha*j grows with
 * the segment-relative sample index j) reduces through spectral_normalize_phase with float
 * catastrophic cancellation, so the wrapped value loses precision and can land a fraction
 * of an ULP outside [-pi, pi). The dangerous consequence is asinf's domain (NaN), guarded by
 * a clamp in spectral_osc_asin. This locks the guarantee that EVERY timbre stays finite and
 * within full scale at very large phase, so that clamp (and the wrap) cannot silently regress. */
static void test_large_phase_safety(void) {
    /* up to ~1.6e7 rad: a high partial sustained for minutes. */
    const float ps[] = { 2.765e4f, 2.765e5f, 1.0e6f, 5.0e6f, 1.6e7f };
    for (size_t i = 0; i < sizeof(ps) / sizeof(ps[0]); i++) {
        float rads = spectral_normalize_phase(ps[i]);
        CHECK(fabsf(rads) <= (float)M_PI + 1.0e-3f,
              "normalize_phase(%.3e) = %.6f outside [-pi, pi]", (double)ps[i], (double)rads);
        const float w[6] = {
            spectral_osc_sine(rads, 0.5f),     spectral_osc_saw(rads, 0.5f),
            spectral_osc_square(rads, 0.5f),   spectral_osc_triangle(rads, 0.5f),
            spectral_osc_parabola(rads, 0.5f), spectral_osc_asin(rads, 0.5f),
        };
        const char* nm[6] = { "sine", "saw", "square", "triangle", "parabola", "asin" };
        for (int k = 0; k < 6; k++) {
            CHECK(isfinite(w[k]) && fabsf(w[k]) <= 1.0001f,
                  "timbre %s at phase %.3e -> %g (must be finite, |w| <= 1)",
                  nm[k], (double)ps[i], (double)w[k]);
        }
    }
    printf("large-phase safety: all timbres finite and within full scale up to 1.6e7 rad\n");
}

int main(void) {
    printf("== guarantee manifest / self-report (B1/B2) ==\n");
    test_bitset_wiring();
    test_fail_closed();
    test_drift();
    test_large_phase_safety();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
