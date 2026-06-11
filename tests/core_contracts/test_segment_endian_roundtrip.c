/* test_segment_endian_roundtrip.c - the Segment endian swap must cover the
 * ENTIRE serialized 64-byte payload, including the _pad_w cubic-MQ phase
 * annotation words (c2/c3/flag).
 *
 * Regression for a confirmed defect: spectral_segment_swap_endian swapped only
 * start..width (8 words) and skipped _pad_w[0..7] (offsets 32..60). When
 * SPECTRAL_PRECISE_PHASE is enabled, _pad_w[0..2] carry cubic phase coefficients
 * that ARE serialized to the segment .bin / seg cache, so a cross-endian file
 * exchange (big-endian writer / little-endian reader or vice versa) decoded
 * garbage cubic coefficients — and a swapped non-zero flag word could even make
 * spectral_segment_has_cubic() spuriously true.
 *
 * The swap is a no-op on the little-endian host this runs on, so the production
 * wrapper's effect is unobservable here. We test the unconditional core,
 * spectral_segment_swap_words(), which the wrapper calls on big-endian hosts:
 * (1) it must change EVERY word of a fully-populated Segment (a skipped word is
 * the bug), and (2) it must be its own inverse (round-trip identity).
 *
 * Run: cmake --build build --target segment_endian_roundtrip_test
 *      && ctest --test-dir build -R segment_endian_roundtrip
 */
#include <stdio.h>
#include <string.h>

#include "spectral_common.h"
#include "spectral_endian.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

/* Distinct, non-byte-palindrome float values so each word visibly changes under
 * a byte swap (e.g. 1.5f = 0x3FC00000 -> 0x0000C03F). */
static void populate(Segment* s) {
    memset(s, 0, sizeof(*s));
    s->start = 1.5f; s->length = 2.5f; s->phase = 3.5f; s->omega = 4.5f;
    s->df = 5.5f; s->amp = 6.5f; s->da = 7.5f; s->width = 8.5f;
    for (int i = 0; i < 8; i++) s->_pad_w[i] = 9.5f + (float)i;
}

static int word_count_unchanged(const Segment* a, const Segment* b) {
    /* Compare the 16 serialized float words bitwise. */
    const float* wa = &a->start;  /* start..da are 7 contiguous floats */
    const float* wb = &b->start;
    int unchanged = 0;
    for (int i = 0; i < 7; i++)
        if (memcmp(&wa[i], &wb[i], sizeof(float)) == 0) unchanged++;
    if (memcmp(&a->width, &b->width, sizeof(float)) == 0) unchanged++;
    for (int i = 0; i < 8; i++)
        if (memcmp(&a->_pad_w[i], &b->_pad_w[i], sizeof(float)) == 0) unchanged++;
    return unchanged;
}

static void test_swap_covers_every_word(void) {
    printf("test_swap_covers_every_word:\n");
    Segment orig, work;
    populate(&orig);
    work = orig;

    spectral_segment_swap_words(&work);
    int unchanged = word_count_unchanged(&work, &orig);
    CHECK(unchanged == 0,
          "swap left %d of 16 words unchanged (a skipped word, e.g. _pad_w cubic)", unchanged);

    spectral_segment_swap_words(&work);
    CHECK(memcmp(&work, &orig, sizeof(Segment)) == 0,
          "swap is not its own inverse (round-trip changed the struct)");
}

/* The cubic annotation specifically must survive (set via the public helper,
 * read via the public accessors) — proving _pad_w[0..2] are covered. */
static void test_cubic_annotation_round_trips(void) {
    printf("test_cubic_annotation_round_trips:\n");
    Segment s;
    memset(&s, 0, sizeof(s));
    spectral_segment_set_cubic(&s, -0.125f, 0.0625f);  /* c2, c3 -> _pad_w[0],_pad_w[1]; flag _pad_w[2] */
    CHECK(spectral_segment_has_cubic(&s), "precondition: annotation set");

    Segment swapped = s;
    spectral_segment_swap_words(&swapped);
    /* After a single swap the bytes are foreign-order; one more swap restores. */
    spectral_segment_swap_words(&swapped);
    CHECK(memcmp(&swapped, &s, sizeof(Segment)) == 0, "cubic segment round-trip identity");
    CHECK(spectral_segment_has_cubic(&swapped), "has_cubic must survive round-trip");
    CHECK(spectral_segment_cubic_c2(&swapped) == -0.125f, "c2 must survive round-trip");
    CHECK(spectral_segment_cubic_c3(&swapped) == 0.0625f, "c3 must survive round-trip");
}

int main(void) {
    test_swap_covers_every_word();
    test_cubic_annotation_round_trips();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
