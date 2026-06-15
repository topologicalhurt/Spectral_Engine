/* test_synth_hybrid_parity.c - F2b contract test for the hybrid IFFT fast path.
 *
 *   1. Eligibility: the classifier accepts a stationary in-domain sine partial
 *      and rejects chirp / cubic / amplitude-ramp / too-short / out-of-domain.
 *   2. Fast-path parity: a dense set of eligible segments rendered via
 *      spectral_synth_hybrid_try_render() matches the exact double oscillator
 *      sum in the deep interior (away from the COLA fade edges) at the F1 floor.
 *   3. Decline contract: non-sine timbre, non-identity stretch/pitch, an
 *      ineligible segment, or a sub-crossover count DECLINES with out[] left
 *      untouched, so the caller's synth_cpu fallback owns those renders.
 *
 * Run: cmake --build build --target synth_hybrid_parity_test
 *      && ctest --test-dir build -R synth_hybrid_parity
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "spectral_common.h"
#include "spectral_consts.h"
#include "spectral_synth_hybrid.h"

#include "../support/check.h"
#include "../support/xorshift_rng.h"

#define N_FFT   512u      /* must match the fast path's internal frame size */
#define TOTAL   8192u
#define N_SEG   16u

static double dbfs(double x) { return 20.0 * log10(x > 1e-300 ? x : 1e-300); }

/* Fill a stationary-sine fixture: distinct fractional bins, staggered onsets
 * (so phase0 = phase - omega*start is non-trivial), all spanning to the end. */
static void make_fixture(Segment* segs) {
    unsigned rng = 0xa5a5f00du;
    memset(segs, 0, N_SEG * sizeof(Segment));
    for (uint32_t i = 0; i < N_SEG; i++) {
        double bin = 12.3 + 14.0 * (double)i;                 /* 12.3 .. 222.3, in (1,255) */
        segs[i].start = (float)(i * 8);                       /* small staggered onsets */
        segs[i].length = (float)TOTAL;                        /* active through the end */
        segs[i].omega = (float)(bin * 2.0 * SPECTRAL_PI_D / (double)N_FFT);
        segs[i].phase = (float)(2.0 * SPECTRAL_PI_D * xorshift32_unit(&rng) - SPECTRAL_PI_D);
        segs[i].amp = (float)((0.1 + 0.9 * xorshift32_unit(&rng)) / N_SEG);
        segs[i].df = 0.0f;
        segs[i].da = 0.0f;
    }
}

static void test_eligibility(void) {
    printf("test_eligibility:\n");
    static Segment segs[N_SEG];
    make_fixture(segs);

    CHECK(spectral_synth_hybrid_segment_eligible(&segs[0], N_FFT) == 1, "stationary sine eligible");

    Segment bad = segs[0];
    bad.df = 1.0e-3f;
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "chirp rejected");
    bad = segs[0]; bad.length = 100.0f;
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "too-short rejected");
    bad = segs[0]; bad.da = 1.0e-2f;   /* da*length huge */
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "amplitude ramp rejected");
    bad = segs[0]; bad.omega = (float)(0.5 * 2.0 * SPECTRAL_PI_D / N_FFT);  /* bin 0.5 < 1 */
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "out-of-domain bin rejected");
    bad = segs[0]; spectral_segment_set_cubic(&bad, 1.0e-2f, 0.0f);
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "cubic phase rejected");
    bad = segs[0]; bad.amp = NAN;
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "non-finite amp rejected");
    bad = segs[0]; bad.phase = INFINITY;
    CHECK(spectral_synth_hybrid_segment_eligible(&bad, N_FFT) == 0, "non-finite phase rejected");
}

static void test_fast_path_parity(void) {
    printf("test_fast_path_parity:\n");
    static Segment segs[N_SEG];
    make_fixture(segs);
    SegmentArray sa = { segs, N_SEG, N_SEG };

    static float out[TOTAL];
    SpectralHybridResult r =
        spectral_synth_hybrid_try_render(sa, out, TOTAL, 1.0f, 1.0f, TIMBRE_SINE);
    CHECK(r == SPECTRAL_HYBRID_RENDERED, "dense eligible set must take the fast path");
    if (r != SPECTRAL_HYBRID_RENDERED) return;

    /* Deep interior: every segment active (t >= max start + one frame) and full
     * COLA coverage (t <= total - one frame). Compare to the exact oscillator
     * sum; phase0 = phase - omega*start. */
    size_t lo = (size_t)segs[N_SEG - 1].start + N_FFT;
    size_t hi = TOTAL - N_FFT;
    double worst = 0.0, sumsq = 0.0;
    size_t cnt = 0;
    for (size_t t = lo; t < hi; t++) {
        double want = 0.0;
        for (uint32_t i = 0; i < N_SEG; i++) {
            double omega = (double)segs[i].omega;
            double phi0 = (double)segs[i].phase - omega * (double)segs[i].start;
            want += (double)segs[i].amp * cos(omega * (double)t + phi0);
        }
        double err = fabs((double)out[t] - want);
        if (err > worst) worst = err;
        sumsq += err * err;
        cnt++;
    }
    double rms = sqrt(sumsq / (double)cnt);
    printf("  deep-interior parity: max=%.2f dBFS rms=%.2f dBFS (%u partials)\n",
           dbfs(worst), dbfs(rms), (unsigned)N_SEG);
    CHECK(dbfs(worst) < -55.0, "fast-path max err must beat -55 dBFS (got %.2f)", dbfs(worst));
    CHECK(dbfs(rms) < -70.0, "fast-path rms err must beat -70 dBFS (got %.2f)", dbfs(rms));
}

/* out[] must be left untouched on every decline so the caller's fallback owns it. */
static void check_declines(SegmentArray sa, float* out, const char* what,
                           float stretch, float pitch, SpectralTimbre timbre) {
    for (size_t t = 0; t < TOTAL; t++) out[t] = -123.5f;
    SpectralHybridResult r =
        spectral_synth_hybrid_try_render(sa, out, TOTAL, stretch, pitch, timbre);
    CHECK(r == SPECTRAL_HYBRID_DECLINED, "%s must decline", what);
    int touched = 0;
    for (size_t t = 0; t < TOTAL && !touched; t++) touched = (out[t] != -123.5f);
    CHECK(!touched, "%s decline must leave out[] untouched", what);
}

static void test_decline_contract(void) {
    printf("test_decline_contract:\n");
    static Segment segs[N_SEG];
    static float out[TOTAL];

    make_fixture(segs);
    SegmentArray sa = { segs, N_SEG, N_SEG };
    check_declines(sa, out, "non-sine timbre", 1.0f, 1.0f, TIMBRE_SAW);
    check_declines(sa, out, "stretch != 1", 2.0f, 1.0f, TIMBRE_SINE);
    check_declines(sa, out, "pitch != 1", 1.0f, 2.0f, TIMBRE_SINE);

    make_fixture(segs);
    segs[5].df = 1.0e-3f;   /* one chirped segment -> all-or-decline */
    check_declines(sa, out, "one ineligible segment", 1.0f, 1.0f, TIMBRE_SINE);

    make_fixture(segs);
    SegmentArray few = { segs, 4u, N_SEG };   /* below the crossover */
    check_declines(few, out, "sub-crossover count", 1.0f, 1.0f, TIMBRE_SINE);
}

int main(void) {
    test_eligibility();
    test_fast_path_parity();
    test_decline_contract();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
