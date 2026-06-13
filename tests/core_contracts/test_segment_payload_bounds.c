/* test_segment_payload_bounds.c - the Segment payload contract must upper-bound
 * |width|, not just require finiteness.
 *
 * Defense-in-depth for the rads*width int-conversion boundary: both confirmed
 * INT_MAX defects (one scalar, one SIMD) reached the oscillator with
 * a finite-but-astronomical width that spectral_segment_payload_valid happily
 * accepted (it only checked isfinite). The formula-level guards are correct
 * now, but a validated Segment must never carry a width that even approaches
 * the (float)INT_MAX envelope: the contract bound is SPECTRAL_SEGMENT_WIDTH_MAX
 * (2^24) on |width|, ~40x under the pi*width UB threshold of ~2^31/pi.
 *
 * Fails if the bound is removed from spectral_segment_payload_valid (a
 * finite 1e9f width validating again is the bug) and if the bound is
 * over-tightened (the inclusive boundary and every normal width must validate).
 *
 * Run: cmake --build build --target segment_payload_bounds_test
 *      && ctest --test-dir build -R segment_payload_bounds
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "spectral_common.h"
#include "spectral_contracts.h"

#include "../support/check.h"

static void populate_valid(Segment* s) {
    memset(s, 0, sizeof(*s));
    s->start = 0.0f; s->length = 256.0f; s->phase = 0.25f;
    s->omega = 0.1f; s->df = 0.0f; s->amp = 0.5f; s->da = 0.0f;
    s->width = SPECTRAL_TRACK_DEFAULT_WIDTH;
}

static void test_width_in_domain_validates(void) {
    Segment s;
    printf("test_width_in_domain_validates:\n");

    populate_valid(&s);
    CHECK(spectral_segment_payload_valid(&s), "default width must validate");
    CHECK(spectral_segment_valid_for_synth(&s), "default segment must be synth-valid");

    s.width = 0.0f;
    CHECK(spectral_segment_payload_valid(&s), "width=0 is in payload domain");
    s.width = 1.0f;
    CHECK(spectral_segment_payload_valid(&s), "pwm duty domain must validate");
    s.width = -1.0f;
    CHECK(spectral_segment_payload_valid(&s), "small negative width is in payload domain");

    /* Inclusive boundary: exactly SPECTRAL_SEGMENT_WIDTH_MAX is still valid. */
    s.width = SPECTRAL_SEGMENT_WIDTH_MAX;
    CHECK(spectral_segment_payload_valid(&s), "width == WIDTH_MAX must validate (inclusive bound)");
    s.width = -SPECTRAL_SEGMENT_WIDTH_MAX;
    CHECK(spectral_segment_payload_valid(&s), "width == -WIDTH_MAX must validate (inclusive bound)");
}

static void test_width_beyond_bound_rejects(void) {
    Segment s;
    printf("test_width_beyond_bound_rejects:\n");

    populate_valid(&s);
    s.width = nextafterf(SPECTRAL_SEGMENT_WIDTH_MAX, INFINITY);
    CHECK(!spectral_segment_payload_valid(&s), "width just above WIDTH_MAX must reject");

    /* The actual defect class: finite width large enough that rads*width can
     * cross (float)INT_MAX. The scalar and SIMD reproducers live here. */
    s.width = 1e9f;
    CHECK(!spectral_segment_payload_valid(&s), "width=1e9 (UB envelope class) must reject");
    s.width = 1073741824.0f; /* 2^30, the osc_formulas_domain reproducer width */
    CHECK(!spectral_segment_payload_valid(&s), "width=2^30 must reject");
    s.width = -1e9f;
    CHECK(!spectral_segment_payload_valid(&s), "bound is on |width|: -1e9 must reject");

    /* Synth validity inherits the payload bound. */
    s.width = 1e9f;
    CHECK(!spectral_segment_valid_for_synth(&s), "synth validity must inherit the width bound");

    /* Pre-existing finiteness behavior is preserved. */
    s.width = NAN;
    CHECK(!spectral_segment_payload_valid(&s), "NaN width must reject");
    s.width = INFINITY;
    CHECK(!spectral_segment_payload_valid(&s), "Inf width must reject");
}

static void test_array_contract_reports_offender(void) {
    Segment segs[3];
    uint32_t bad_idx = 0;
    printf("test_array_contract_reports_offender:\n");

    for (int i = 0; i < 3; i++) populate_valid(&segs[i]);
    CHECK(spectral_segment_array_payload_valid(segs, 3, &bad_idx), "all-valid array must validate");

    segs[1].width = 1e9f;
    CHECK(!spectral_segment_array_payload_valid(segs, 3, &bad_idx), "array with one oversized width must reject");
    CHECK(bad_idx == 1u, "offending index must be reported, got %u", bad_idx);
}

int main(void) {
    test_width_in_domain_validates();
    test_width_beyond_bound_rejects();
    test_array_contract_reports_offender();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
