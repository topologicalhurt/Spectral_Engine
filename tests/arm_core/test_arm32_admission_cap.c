/* test_arm32_admission_cap.c - the per-block admission cap bounds active voices.
 *
 * The real-time arm32 synth admits at most SPECTRAL_ARM32_ACTIVE_CAP voices per
 * block (the activation scan in spectral_synth_arm32.c). That cap is DISTINCT
 * from the storage bound SPECTRAL_ARM32_MAX_ACTIVE: storage sizes the state
 * arrays; the cap bounds per-block render WORK so a dense .spq cannot exceed the
 * WCET budget. This harness forces the cap LOW (below the storage bound, via the
 * cmake target define) and feeds MORE simultaneously-live segments than the cap,
 * then asserts the kernel activates exactly the cap -- never more -- with the
 * surplus deferred (natural backpressure), not stored past capacity.
 *
 * Fail-on-bug: the fixture overlaps 2.5x the cap, so if admission were bounded by
 * storage (512) rather than the cap, peak_active would reach the full overlap and
 * the `== cap` CHECK would fail. (Verified by building once against the storage
 * bound: peak hit the overlap count, tripping the assert.)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_synth_arm32.h"
#include "spectral_lut.h"
#include "spectral_q.h"
#include "spectral_config.h"   /* SPECTRAL_ARM32_ACTIVE_CAP / _MAX_ACTIVE */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../support/check.h"

int main(void) {
    const uint32_t sr = 48000;
    const uint32_t cap = (uint32_t)SPECTRAL_ARM32_ACTIVE_CAP;

    static q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1];
    spectral_lut_init_sine(lut);

    printf("test_arm32_admission_cap (cap=%u, storage=%u):\n",
           cap, (unsigned)SPECTRAL_ARM32_MAX_ACTIVE);

    /* The cap must be forced below storage for this harness to bind; the build
     * target sets SPECTRAL_ARM32_ACTIVE_CAP small. Guard against an accidental
     * full-storage build that would make the test vacuous. */
    CHECK(cap < (uint32_t)SPECTRAL_ARM32_MAX_ACTIVE,
          "harness requires a forced-low cap (cap=%u must be < storage=%u)",
          cap, (unsigned)SPECTRAL_ARM32_MAX_ACTIVE);

    /* Overlap 2.5x the cap: all segments are live across the first block, so the
     * loader (which bounds the *simultaneously-live* count against storage, not
     * the cap) accepts them, and admission is what must throttle to the cap. */
    const uint32_t n_seg = cap * 2u + cap / 2u + 1u;
    const uint16_t length = 24000;          /* ~0.5 s at 48 kHz, fits uint16 */
    SpectralSegmentQ15* segs =
        (SpectralSegmentQ15*)calloc(n_seg, sizeof(SpectralSegmentQ15));
    if (!segs) { printf("  FAIL: oom\n"); return 1; }

    for (uint32_t i = 0; i < n_seg; i++) {
        /* Strictly-increasing starts (loader wants non-decreasing) that all still
         * overlap the first block, so every segment is eligible at out_pos=0. */
        segs[i].start = (uint32_t)i;
        segs[i].length = length;
        segs[i].freq_q88 = OMEGA_TO_Q88(2.0 * M_PI * 440.0 / (double)sr);
        segs[i].phase_q15 = 0;
        segs[i].amp_q15 = FLOAT_TO_Q15(0.25f);
        segs[i].da_q15 = 0;
    }

    /* Segment storage holds all n_seg; the ACTIVE arrays are sized by storage. */
    SpectralArm32Ctx ctx;
    SpectralSegmentQ15* pool =
        (SpectralSegmentQ15*)calloc(n_seg, sizeof(SpectralSegmentQ15));
    if (!pool) { printf("  FAIL: oom\n"); free(segs); return 1; }

    spectral_arm32_init(&ctx, pool, (uint16_t)n_seg, lut, sr);
    const uint32_t total = (uint32_t)length + n_seg + 256u;
    CHECK(spectral_arm32_load(&ctx, segs, (uint16_t)n_seg, total) == SPECTRAL_OK,
          "load(%u overlapping segs) should be OK (<= storage)", n_seg);

    /* Render a handful of blocks so admission runs and peak polyphony settles. */
    q15_t blk[256];
    for (uint32_t b = 0; b < 4u; b++) {
        (void)spectral_arm32_process(&ctx, blk, NULL, 256);
    }

    uint16_t peak = spectral_arm32_get_peak_active(&ctx);
    printf("  n_seg=%u overlapping, peak_active=%u\n", n_seg, (unsigned)peak);

    /* The core contract: never admit more than the cap. */
    CHECK(peak <= (uint16_t)cap,
          "peak_active (%u) must not exceed the admission cap (%u)",
          (unsigned)peak, cap);
    /* Non-vacuous: with 2.5x overlap the cap actually binds, so peak hits it
     * exactly. (Without the cap, peak would be n_seg=%u.) */
    CHECK(peak == (uint16_t)cap,
          "peak_active (%u) should bind exactly at the cap (%u); n_seg=%u",
          (unsigned)peak, cap, n_seg);

    free(pool);
    free(segs);

    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
