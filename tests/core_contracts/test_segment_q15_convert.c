/* test_segment_q15_convert.c - the float->Q15 runtime converter must REJECT a grain whose
 * stretched length exceeds the uint16 length field, not silently truncate it.
 *
 * SpectralSegmentQ15.length is uint16 (<= 65535 output samples). The converter once clamped
 * length_d to 65535 and returned success, so a partial stretched past 65535 output samples
 * rendered only its first 65535 and then went silent mid-note on the embedded path, while
 * desktop float rendered the full duration. The converter's contract is "faithfully convert
 * or drop"; this pins that a non-representable length is a drop (return 0), and that lengths
 * up to the limit still convert.
 *
 * Run: cmake --build build --target segment_q15_convert_test
 *      && ctest --test-dir build -R segment_q15_convert
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "spectral_common.h"
#include "spectral_segment_convert.h"

#include "../support/check.h"

static Segment make_seg(float length) {
    Segment s;
    memset(&s, 0, sizeof s);
    s.start  = 0.0f;
    s.length = length;
    s.omega  = 0.10f;   /* in (0, pi) */
    s.phase  = 0.30f;
    s.amp    = 0.50f;
    s.df     = 0.0f;
    s.da     = 0.0f;
    return s;
}

static SynthParams make_params(float stretch) {
    SynthParams p;
    memset(&p, 0, sizeof p);
    p.stretch        = stretch;
    p.inv_stretch    = 1.0f / stretch;
    p.inv_stretch_sq = p.inv_stretch * p.inv_stretch;
    p.pitch_factor   = 1.0f;        /* pitch 0 semitones */
    p.out_len        = (size_t)1 << 22;
    p.num_segments   = 1u;
    return p;
}

/* length_d = src.length * stretch; the uint16 cap is 65535. */
static int convert(float length, float stretch, SpectralSegmentQ15* out) {
    Segment s = make_seg(length);
    SynthParams p = make_params(stretch);
    return spectral_segment_to_q15_runtime(&s, out, 1.0f, &p, p.out_len);
}

int main(void) {
    printf("test_segment_q15_convert:\n");
    SpectralSegmentQ15 q;

    /* Well within the field: converts and carries the exact stretched length. */
    CHECK(convert(512.0f, 10.0f, &q) == 1, "length 5120 must convert");
    CHECK(q.length == 5120u, "length must be the exact stretched value (got %u)", (unsigned)q.length);

    /* Just under the limit: length_d = 512*127 = 65024 <= 65535, converts. */
    CHECK(convert(512.0f, 127.0f, &q) == 1, "length 65024 (<=65535) must convert");
    CHECK(q.length == 65024u, "boundary length must be exact (got %u)", (unsigned)q.length);

    /* Over the limit: length_d = 512*128 = 65536 > 65535. Must REJECT, not truncate to 65535. */
    CHECK(convert(512.0f, 128.0f, &q) == 0, "length 65536 (>65535) must be rejected, not truncated");

    /* Extreme stretch within SPECTRAL_MAX_STRETCH: 512*200 = 102400, rejected. */
    CHECK(convert(512.0f, 200.0f, &q) == 0, "length 102400 must be rejected");

    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
