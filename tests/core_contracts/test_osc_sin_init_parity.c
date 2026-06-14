/* test_osc_sin_init_parity.c - Parity gate for spectral_osc_sin_init_f64, the
 * self-contained DOUBLE-precision init sine that seeds the coupled-form Q31
 * oscillator's rotation constants (cos_w, sin_w) and initial state (c, s).
 *
 * Why this exists: the Q31 rotation constants need ~1e-9 accuracy (see
 * spectral_osc_q31.h header). The float minimax sine (~1e-7) is explicitly
 * REJECTED there because its angle error drifts phase over a long partial and
 * collapses SNR (measured erratic -3..60 dB). There is no shared
 * double-precision sine to route through, so this function is its own
 * reference-grade implementation -- and therefore needs its own parity gate
 * against libm to keep it that way.
 *
 * Domain: derived from the only callers (spectral_coupled_init in
 * arch/arm/spectral_synth_arm32.c). omega/phi come from a uint32 phase times
 * SPECTRAL_PHASE_U32_TO_RAD (2*pi/2^32), so both lie in [0, 2*pi).
 * coupled_init evaluates sin on {omega, omega+pi/2, phi, phi+pi/2}, giving a
 * real input domain of [0, 2*pi + pi/2). We sweep that whole range densely.
 *
 * Tolerance (justified, not a tautology): measured max |f - libm sin| over
 * this domain is 6.0e-12 (< 1 Q31 LSB = 4.66e-10). The gate is 1e-9 -- AT the
 * header's stated accuracy requirement, ~160x above the measured residual (not
 * flaky). Regressions fail hard: truncating the degree-15 Taylor to degree-9,
 * or substituting the float ~1e-7 minimax, each produce ~3.5e-6 error (~3500x
 * over the gate). The exact endpoints (sin 0 = 0, sin pi/2 = 1) and odd
 * symmetry are also asserted so a sign/fold regression trips even if a
 * coincidentally-accurate polynomial slipped the absolute band.
 *
 * Run: cmake --build build --target osc_sin_init_parity_test
 *      && ctest --test-dir build -R osc_sin_init_parity
 */
#include "spectral_osc_q31.h"   /* spectral_osc_sin_init_f64, spectral_coupled_init */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>             /* labs */

#include "../support/check.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Mirror the caller's phase->rad map exactly (SPECTRAL_PHASE_U32_TO_RAD). */
#define U32_TO_RAD  (6.283185307179586232 / 4294967296.0)
#define HALF_PI     1.57079632679489661923

/* Required accuracy for the Q31 rotation constants (spectral_osc_q31.h). */
#define TOL_ABS     1.0e-9

/* Dense sweep of the REAL caller input domain [0, 2*pi + pi/2): sample the
 * uint32 phase space (as the activator does), map to radians, and feed both
 * the bare angle (omega/phi) and angle+pi/2 (the cos seed) through the
 * function -- the two args coupled_init actually passes. Compare to libm. */
static void test_parity_over_caller_domain(void) {
    printf("test_parity_over_caller_domain (domain [0, 2pi+pi/2), tol %.0e):\n", TOL_ABS);
    const uint64_t N = 4000001ULL;          /* deterministic dense grid */
    double max_err = 0.0, max_x = 0.0;
    for (uint64_t i = 0; i < N; i++) {
        double frac = (double)i / (double)(N - 1);          /* [0, 1] */
        uint32_t u  = (uint32_t)(frac * 4294967295.0);      /* full uint32 phase */
        double base = (double)u * U32_TO_RAD;               /* [0, 2*pi) */
        double args[2] = { base, base + HALF_PI };          /* [0, 2*pi+pi/2) */
        for (int a = 0; a < 2; a++) {
            double x   = args[a];
            double got = spectral_osc_sin_init_f64(x);
            double ref = sin(x);
            double e   = fabs(got - ref);
            if (e > max_err) { max_err = e; max_x = x; }
        }
    }
    printf("  max |init_sin - libm sin| = %.3e at x = %.6f\n", max_err, max_x);
    CHECK(max_err <= TOL_ABS,
          "init sine deviates %.3e from libm (> %.0e) over caller domain",
          max_err, TOL_ABS);
}

/* The Q31 rotation constants are what the recurrence actually consumes. Round
 * both the function and libm to Q31 (using the shipping rounder) and require
 * the codes never differ by more than 1 LSB -- the measured worst case. A
 * coarser polynomial would diverge by many LSBs here. */
static void test_q31_codes_within_1lsb(void) {
    printf("test_q31_codes_within_1lsb:\n");
    const uint64_t N = 4000001ULL;
    long max_lsb = 0;
    for (uint64_t i = 0; i < N; i++) {
        double frac = (double)i / (double)(N - 1);
        uint32_t u  = (uint32_t)(frac * 4294967295.0);
        double base = (double)u * U32_TO_RAD;
        double args[2] = { base, base + HALF_PI };
        for (int a = 0; a < 2; a++) {
            double x = args[a];
            q31_t got = spectral_double_to_q31_round(spectral_osc_sin_init_f64(x));
            q31_t ref = spectral_double_to_q31_round(sin(x));
            long d = (long)got - (long)ref;
            if (d < 0) d = -d;
            if (d > max_lsb) max_lsb = d;
        }
    }
    printf("  max Q31 code diff vs libm = %ld LSB\n", max_lsb);
    CHECK(max_lsb <= 1,
          "Q31 rotation-constant code differs from libm by %ld LSB (> 1)", max_lsb);
}

/* Exact-value anchors and odd symmetry: catch a sign/fold/parity regression
 * that an absolute band might not (these are independent of polynomial slack).
 * sin(0)=0 exactly; sin(pi/2)=1 within the function's own residual; the fold
 * must preserve sin(-x) = -sin(x) across multiple-of-pi boundaries. The init
 * domain is non-negative, but coupled_init's fold logic handles both signs and
 * the negative branch must not silently rot. */
static void test_exact_anchors_and_symmetry(void) {
    printf("test_exact_anchors_and_symmetry:\n");
    CHECK(spectral_osc_sin_init_f64(0.0) == 0.0,
          "sin(0) must be exactly 0, got %.3e", spectral_osc_sin_init_f64(0.0));
    CHECK(fabs(spectral_osc_sin_init_f64(HALF_PI) - 1.0) <= TOL_ABS,
          "sin(pi/2) must be ~1, got %.12f", spectral_osc_sin_init_f64(HALF_PI));
    /* Odd symmetry across several reduction intervals. */
    const double xs[] = { 0.3, 1.2, HALF_PI, 2.0, M_PI - 0.1,
                          M_PI + 0.4, 2.0 * M_PI - 0.2, 2.0 * M_PI + HALF_PI - 0.1 };
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        double sp = spectral_osc_sin_init_f64(xs[i]);
        double sn = spectral_osc_sin_init_f64(-xs[i]);
        CHECK(fabs(sp + sn) <= TOL_ABS,
              "odd symmetry broken at x=%.4f: sin(x)=%.12f sin(-x)=%.12f",
              xs[i], sp, sn);
    }
}

/* End-to-end via the real entry point: spectral_coupled_init must produce Q31
 * rotation constants that match cos(omega)/sin(omega) from libm to 1 LSB, for a
 * spread of real frequencies (Hz/fs -> omega) and phases. This proves the gate
 * holds through the actual call path, including the omega+pi/2 cos route. */
static void test_coupled_init_constants(void) {
    printf("test_coupled_init_constants:\n");
    const double fs = 48000.0;
    const double freqs[] = { 20.0, 27.5, 440.0, 1000.0, 6000.0, 12000.0, 20000.0 };
    const double phis[]  = { 0.0, 0.31, 1.5, 3.0, 6.0 };
    long max_lsb = 0;
    for (size_t fi = 0; fi < sizeof(freqs) / sizeof(freqs[0]); fi++) {
        double omega = 2.0 * M_PI * freqs[fi] / fs;
        for (size_t pi = 0; pi < sizeof(phis) / sizeof(phis[0]); pi++) {
            double phi = phis[pi];
            SpectralCoupledOsc o;
            q31_t cos_w, sin_w;
            spectral_coupled_init(&o, omega, phi, &cos_w, &sin_w);
            q31_t rcw = spectral_double_to_q31_round(cos(omega));
            q31_t rsw = spectral_double_to_q31_round(sin(omega));
            q31_t rc  = spectral_double_to_q31_round(cos(phi));
            q31_t rs  = spectral_double_to_q31_round(sin(phi));
            long d;
            d = labs((long)cos_w - (long)rcw); if (d > max_lsb) max_lsb = d;
            d = labs((long)sin_w - (long)rsw); if (d > max_lsb) max_lsb = d;
            d = labs((long)o.c   - (long)rc ); if (d > max_lsb) max_lsb = d;
            d = labs((long)o.s   - (long)rs ); if (d > max_lsb) max_lsb = d;
        }
    }
    printf("  max coupled_init Q31 const diff vs libm = %ld LSB\n", max_lsb);
    CHECK(max_lsb <= 1,
          "coupled_init rotation constant off by %ld LSB vs libm (> 1)", max_lsb);
}

int main(void) {
    printf("== spectral_osc_sin_init_f64 parity vs libm ==\n");
    test_parity_over_caller_domain();
    test_q31_codes_within_1lsb();
    test_exact_anchors_and_symmetry();
    test_coupled_init_constants();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
