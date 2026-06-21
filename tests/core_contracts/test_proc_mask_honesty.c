/* test_proc_mask_honesty.c - the processing chain must never silently no-op a
 * requested stage. Regression gate for the honesty fixes:
 *   - W5: a RESERVED stage (parsed but unimplemented) fails loudly, not OK;
 *   - SD-5: the EXPERIMENTAL adaptive_track_density stage is compiled out unless
 *     SPECTRAL_PRECISE_PHASE=1, and when it is off (the default, shipped build)
 *     selecting it must fail loudly (BACKEND_UNAVAIL) rather than return OK while
 *     doing nothing;
 *   - mask = NONE is a genuine no-op (OK) -- the honest case, not a silent lie.
 *
 * "Honesty is not a build option": a stage the user asks for either runs or
 * fails loudly. This pins that contract through the real chain dispatch.
 *
 * Run: cmake --build build --target proc_mask_honesty_test
 *      && ctest --test-dir build -R proc_mask_honesty
 */
#include "spectral_processing_chain.h"
#include "spectral_common.h"
#include "spectral_error.h"
#include "spectral_config.h"   /* SPECTRAL_PRECISE_PHASE */

#include <stdio.h>

#include "../support/check.h"

static SpectralError apply_mask(SpectralProcessMask mask) {
    /* The stage bodies ignore the segment payload on the failure/no-op paths;
     * an empty (non-NULL) array satisfies the chain's boundary validation. */
    SegmentArray sa = SEGMENT_ARRAY_EMPTY;
    SpectralProcessReport report;
    return spectral_process_chain_apply(&sa, 48000, 256, 512, mask, &report);
}

int main(void) {
    printf("== processing-chain honesty (no silent no-op) ==\n");

    /* (1) NONE: nothing requested -> a true no-op succeeds. */
    SpectralError e_none = apply_mask(SPECTRAL_PROC_NONE);
    CHECK(e_none == SPECTRAL_OK,
          "mask=NONE must succeed as a no-op, got %d", (int)e_none);

    /* (2) RESERVED stage (johnston_1988 is parsed but has no implementation):
     * must fail loudly rather than silently return OK. */
    SpectralError e_res = apply_mask(SPECTRAL_PROC_JOHNSTON_1988);
    CHECK(e_res != SPECTRAL_OK,
          "reserved stage johnston_1988 must fail loudly, got OK (silent no-op)");

    /* (3) adaptive_track_density: experimental, gated on SPECTRAL_PRECISE_PHASE.
     * Off (default shipped build) it must fail loudly; on, it runs. */
    SpectralError e_atd = apply_mask(SPECTRAL_PROC_ADAPTIVE_TRACK_DENSITY);
#if SPECTRAL_PRECISE_PHASE
    CHECK(e_atd == SPECTRAL_OK,
          "adaptive_track_density should run under SPECTRAL_PRECISE_PHASE=1, got %d",
          (int)e_atd);
#else
    CHECK(e_atd == SPECTRAL_ERR_BACKEND_UNAVAIL,
          "adaptive_track_density must fail loudly (BACKEND_UNAVAIL=%d) when "
          "SPECTRAL_PRECISE_PHASE=0, got %d (silent no-op?)",
          (int)SPECTRAL_ERR_BACKEND_UNAVAIL, (int)e_atd);
#endif

    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
