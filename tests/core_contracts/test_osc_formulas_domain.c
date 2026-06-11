/* test_osc_formulas_domain.c - domain/edge-case contract for the oscillator
 * waveform formulas (core/spectral_osc_formulas.h), the single source of truth
 * shared by the CPU and CUDA backends.
 *
 * Anchored on a confirmed signed-overflow UB in spectral_osc_quantized: the
 * upper range guard was `scaled > (float)INT_MAX`, but (float)INT_MAX rounds
 * up to 2^31, so scaled == 2^31 slipped past the guard into `(int)scaled` —
 * undefined behavior. The contract these formulas promise is that ANY
 * non-finite or out-of-domain input yields a finite, bounded output (0.0f for
 * the reject cases), never UB and never a NaN/Inf leak into the synth path.
 *
 * Compiled under -fsanitize=undefined so a reintroduced overflow ALSO traps,
 * but the behavioral assertions catch the regression on their own in any build.
 *
 * Run: cmake --build build --target osc_formulas_domain_test
 *      && ctest --test-dir build -R osc_formulas_domain
 */
#include <math.h>
#include <stdio.h>

#include "spectral_osc_formulas.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

static int is_finite_f(float x) { return isfinite(x) != 0; }

/* The confirmed UB boundary and its neighborhood. */
static void test_quantized_int_max_boundary(void) {
    printf("test_quantized_int_max_boundary:\n");

    /* rads=2.0, width=2^30 -> scaled = 2^31 exactly. The pre-fix '>' guard let
     * this reach (int)2^31 (UB, ~2.0 at -O0); the contract is reject -> 0.0f. */
    float at_2_31 = spectral_osc_quantized(2.0f, 1073741824.0f);
    CHECK(at_2_31 == 0.0f, "scaled==2^31 must reject to 0.0f, got %.9g", at_2_31);

    /* Just below 2^31 (next representable float down, 2147483520) is a valid
     * int and must quantize, not reject. scaled = 2147483520 -> int the same. */
    float w = 2147483520.0f; /* 2^31 - 128, exactly representable */
    float just_below = spectral_osc_quantized(1.0f, w);
    CHECK(is_finite_f(just_below), "scaled just below 2^31 must be finite, got %.9g", just_below);
    CHECK(just_below != 0.0f || w == 0.0f, "valid in-range scaled should not spuriously reject");

    /* Symmetric lower edge: (float)INT_MIN == -2^31 is exactly representable and
     * (int)(-2^31) == INT_MIN is valid, so it must NOT reject. */
    float at_neg = spectral_osc_quantized(-2.0f, 1073741824.0f);
    CHECK(is_finite_f(at_neg), "scaled==-2^31 must be finite (valid int), got %.9g", at_neg);
}

/* Every formula must reject non-finite / out-of-domain inputs to a finite
 * output, never propagate NaN/Inf into the accumulator. */
static void test_nonfinite_inputs_are_contained(void) {
    printf("test_nonfinite_inputs_are_contained:\n");
    const float bad[] = { NAN, INFINITY, -INFINITY };
    for (int i = 0; i < 3; i++) {
        float b = bad[i];
        CHECK(spectral_osc_quantized(b, 1.0f) == 0.0f, "quantized(nonfinite rads) -> 0");
        CHECK(spectral_osc_quantized(1.0f, b) == 0.0f, "quantized(nonfinite width) -> 0");
        CHECK(is_finite_f(spectral_osc_pwm(b, 0.5f)), "pwm(nonfinite rads) finite");
        CHECK(is_finite_f(spectral_osc_pwm(1.0f, b)), "pwm(nonfinite width) finite");
    }
    /* width <= 0 is out of domain for the quantizer -> reject. */
    CHECK(spectral_osc_quantized(1.0f, 0.0f) == 0.0f, "quantized(width=0) -> 0");
    CHECK(spectral_osc_quantized(1.0f, -4.0f) == 0.0f, "quantized(width<0) -> 0");
}

/* Sanity: in the normal domain the quantizer actually quantizes (staircases). */
static void test_quantized_normal_domain(void) {
    printf("test_quantized_normal_domain:\n");
    /* width=4 quantizes rads*4 to integer steps then divides back: the result
     * is floor-toward-zero(rads*4)/4, so two nearby rads in the same step map
     * to the same output. */
    float a = spectral_osc_quantized(0.30f, 4.0f); /* 1.2 -> 1 -> 0.25 */
    float b = spectral_osc_quantized(0.40f, 4.0f); /* 1.6 -> 1 -> 0.25 */
    CHECK(a == 0.25f, "quantized(0.30,4) == 0.25, got %.9g", a);
    CHECK(b == 0.25f, "quantized(0.40,4) == 0.25, got %.9g", b);
    CHECK(is_finite_f(a) && is_finite_f(b), "quantized normal-domain finite");
}

int main(void) {
    test_quantized_int_max_boundary();
    test_nonfinite_inputs_are_contained();
    test_quantized_normal_domain();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
