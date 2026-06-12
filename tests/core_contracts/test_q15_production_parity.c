/* test_q15_production_parity.c - opt-in Q15 vs float production parity (Q3b).
 *
 * Q3a (q15_compute_precision) measured the Q15 waveform-eval floor in isolation
 * on a synthetic phase sweep. This test pins the SAME bound on the REAL
 * production path: it renders representative segments through the production
 * dispatch timbre_synth_segment() twice —
 *
 *   float reference : Q15 disabled  (the shipping default domain)
 *   Q15 compute     : osc_set_q15_enable(OSC_Q15_BIT(timbre))
 *
 * — and asserts the per-timbre RMS error stays at the characterized Q15 floor.
 * This is the CI lock on the QTYPE_DOMAIN_PLAN.md §7 per-path sign-off: float
 * stays default, and a regression that breaks a Q15 path (or silently widens its
 * error) fails the build. The five timbres are the full Q15-capable set
 * (sine/saw/square/triangle/parabola, osc_q15_available); asin/quantized/pwm
 * have no Q15 path.
 *
 * Both paths carry the real fade-in/out envelope and amp ramp, so the rendered
 * error is the production-observable delta of enabling Q15, not a synthetic
 * waveform-only number. (Faded regions scale BOTH paths by the same envelope, so
 * the aggregate RMS sits at or just below the Q3a constant-amp floor.)
 *
 * Run: cmake --build build --target q15_production_parity_test \
 *      && ctest --test-dir build -R q15_production_parity
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "oscillator.h"
#include "spectral_synth_internal.h"
#include "spectral_common.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

#define MAX_LEN 2048

/* Per-timbre RMS-error budget (dBFS, rel full scale 1.0). The Q3a floor
 * (q15_compute_precision, amp 0.8) is the physical limit: sine -85.1, saw -91.5,
 * square -90.0, triangle -92.6, parabola -91.0. Budgets sit ~12-13 dB above that
 * floor. The production-measured number (printed below) actually lands BELOW the
 * Q3a floor — the fade envelopes scale both paths down in the ramp regions, so
 * aggregate RMS error drops — but the budget is pinned to the floor, not the
 * lucky segment shape, so it stays a stable cross-platform bound. Wide enough to
 * absorb segment/envelope variation, tight enough that a real Q15 regression
 * (wrong shift, broken LUT index, dropped saturation -> >= -60 dBFS) is caught
 * with room to spare. Verdict lives in the printed table; these are tripwires. */
typedef struct {
    SpectralTimbre timbre;
    const char*    name;
    float          width;
    double         budget_dbfs;
} Q15Timbre;

static Q15Timbre k_timbres[] = {
    { TIMBRE_SINE,     "sine",     0.0f, -72.0 },
    { TIMBRE_SAW,      "saw",      0.0f, -78.0 },
    { TIMBRE_SQUARE,   "square",   0.0f, -78.0 },
    { TIMBRE_TRIANGLE, "triangle", 0.0f, -80.0 },
    { TIMBRE_PARABOLA, "parabola", 0.0f, -78.0 },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

typedef struct {
    size_t length;
    float  alpha;   /* phase increment per sample (2*pi*f/fs) */
    float  beta;    /* chirp rate; c2 mirrors it, c3 = 0 */
    float  phase;
    float  amp;
    float  d_amp;
} Q15Segment;

/* Low / high / chirped / odd-length segments (same spread as osc_parity). */
static const Q15Segment k_segments[] = {
    /* len   alpha      beta     phase    amp    d_amp    */
    { 1000,  0.0576f,   0.0f,    0.30f,   0.80f, -2.0e-4f },  /* ~440 Hz @48k */
    {  512,  0.5236f,   0.0f,   -1.10f,   0.50f,  0.0f    },  /* ~4 kHz       */
    {  768,  0.2000f,   1.0e-6f, 0.00f,   0.90f,  1.0e-4f },  /* slight chirp */
    {  257,  0.1234f,   0.0f,    2.50f,   0.70f,  3.0e-4f },  /* odd length   */
};
#define N_SEGMENTS (sizeof(k_segments) / sizeof(k_segments[0]))

static void fill_lp(SegmentLoopParams* lp, const Q15Segment* s, float width) {
    memset(lp, 0, sizeof(*lp));
    lp->start_idx = 0;
    lp->length    = s->length;
    lp->alpha     = s->alpha;
    lp->beta      = s->beta;
    lp->c2        = s->beta;   /* no-linkage default: c2 == beta, c3 == 0 */
    lp->c3        = 0.0f;
    lp->phase     = s->phase;
    lp->amp       = s->amp;
    lp->d_amp     = s->d_amp;
    lp->width     = width;
    lp->valid     = 1;
}

static void test_q15_parity(void) {
    for (size_t t = 0; t < N_TIMBRES; t++) {
        double sumsq_diff = 0.0, sumsq_sig = 0.0, max_abs = 0.0;
        size_t n = 0;

        for (size_t si = 0; si < N_SEGMENTS; si++) {
            SegmentLoopParams lp;
            fill_lp(&lp, &k_segments[si], k_timbres[t].width);

            float buf_float[MAX_LEN];
            float buf_q15[MAX_LEN];
            memset(buf_float, 0, sizeof(buf_float));
            memset(buf_q15,   0, sizeof(buf_q15));

            /* Reference: the shipping default (Q15 off, float dispatch). */
            osc_set_q15_enable(0);
            timbre_synth_segment(buf_float, &lp, k_timbres[t].timbre);

            /* Opt-in Q15 (also builds the sine LUT for the sine path). */
            osc_set_q15_enable(OSC_Q15_BIT(k_timbres[t].timbre));
            timbre_synth_segment(buf_q15, &lp, k_timbres[t].timbre);
            osc_set_q15_enable(0);

            for (size_t j = 0; j < lp.length; j++) {
                double d = (double)buf_q15[j] - (double)buf_float[j];
                double ad = fabs(d);
                if (ad > max_abs) max_abs = ad;
                sumsq_diff += d * d;
                sumsq_sig  += (double)buf_float[j] * (double)buf_float[j];
                n++;
            }
        }

        double rms_diff = (n > 0) ? sqrt(sumsq_diff / (double)n) : 0.0;
        double rms_sig  = (n > 0) ? sqrt(sumsq_sig  / (double)n) : 0.0;
        double rms_dbfs = (rms_diff > 0.0) ? 20.0 * log10(rms_diff) : -300.0;
        double rel_db   = (rms_diff > 0.0 && rms_sig > 0.0)
                          ? 20.0 * log10(rms_diff / rms_sig) : -300.0;

        printf("  %-9s RMS err = %.3e (%.1f dBFS, %.1f dB vs sig)  peak = %.3e  [budget %.1f]\n",
               k_timbres[t].name, rms_diff, rms_dbfs, rel_db, max_abs,
               k_timbres[t].budget_dbfs);

        CHECK(rms_dbfs <= k_timbres[t].budget_dbfs,
              "%s Q15 RMS error %.1f dBFS exceeds budget %.1f dBFS",
              k_timbres[t].name, rms_dbfs, k_timbres[t].budget_dbfs);
    }
}

int main(void) {
    printf("== opt-in Q15 vs float production parity (per-path dBFS) ==\n");
    test_q15_parity();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
