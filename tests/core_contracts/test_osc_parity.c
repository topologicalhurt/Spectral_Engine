/* test_osc_parity.c - SIMD-vs-scalar oscillator parity + RMS-drift contract.
 *
 * Pins the two equivalence guarantees the PASS200 SIMD-default re-baseline was
 * signed off against, so they are checked by CI instead of by hand:
 *
 *   (1) DRIFT GUARD  - per-sample, the SIMD CPU oscillator stays within a tiny
 *       absolute budget of the scalar reference for every SIMD-capable timbre.
 *       The only divergence source is FMA contraction: under -ffast-math the
 *       scalar phase Horner contracts to hardware FMA while the SIMD intrinsics
 *       emit separate mul+add, so the two differ by <= a few ULP per sample.
 *
 *   (2) RMS-dBFS GUARD - aggregated over all rendered samples, the RMS of the
 *       (SIMD - scalar) difference stays far below full scale and far below the
 *       signal itself. This is the automated form of a manual full-mix
 *       measurement (RMS error ~ -159 dBFS).
 *
 * Both paths are driven through the REAL production dispatch, timbre_synth_segment()
 * under osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR | OSC_DISPATCH_ALL_SIMD) - no
 * reimplementation of either kernel. Default oscillator quality (NAIVE) is left
 * untouched so the band-limited path is never entered.
 *
 * Run: cmake --build build --target osc_parity_test && ctest --test-dir build -R osc_parity
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

/* Budgets. Measured worst case on this tree (Apple Silicon, default build):
 *   per-sample max abs diff = 2.38e-7  (quantized; sine/saw/triangle/parabola
 *                                       ~6e-8 = 0.5 ULP from FMA contraction —
 *                                       sine now runs the minimax poly on both
 *                                       paths so it is no longer 0.0; square/pwm
 *                                       are pure comparisons -> 0.0)
 *   aggregate RMS diff      = -155 dBFS abs, -152 dB relative to signal
 * The margin over measured is intentionally wide (~80x abs, ~55 dB RMS): SIMD-vs-
 * scalar agreement is dominated by FMA-contraction behavior, which varies far more
 * across compiler / -ffp-contract / SIMDe-vs-native lane width than a polynomial's
 * error does. A real kernel regression (wrong op order, dropped term, broken lane)
 * lands at >= 1e-3 / >= -60 dBFS, so it is still caught with room to spare. */
#define BUDGET_ABS          1.0e-5    /* per-sample |simd - scalar|            */
#define BUDGET_RMS_DBFS     -100.0    /* 20*log10(rms_diff), rel full scale 1.0 */
#define BUDGET_RMS_REL_DB   -90.0     /* 20*log10(rms_diff / rms_signal)        */

#define MAX_LEN 2048

typedef struct {
    size_t length;
    float  alpha;   /* phase increment per sample (2*pi*f/fs) */
    float  beta;    /* chirp rate; c2 mirrors it, c3 = 0 (default no-linkage)  */
    float  phase;
    float  amp;
    float  d_amp;
    float  width;   /* quantized levels / pwm duty; ignored by other timbres   */
} ParitySegment;

/* Representative segments: a low and a high partial frequency, a chirped one, and
 * an odd length that forces the SIMD scalar-tail (length not a multiple of 4). */
static const ParitySegment k_segments[] = {
    /* len   alpha      beta     phase    amp    d_amp     width */
    { 1000,  0.0576f,   0.0f,    0.30f,   0.80f, -2.0e-4f, 8.0f  },  /* ~440 Hz @48k */
    {  512,  0.5236f,   0.0f,   -1.10f,   0.50f,  0.0f,    5.0f  },  /* ~4 kHz       */
    {  768,  0.2000f,   1.0e-6f, 0.00f,   0.90f,  1.0e-4f, 6.0f  },  /* slight chirp */
    {  257,  0.1234f,   0.0f,    2.50f,   0.70f,  3.0e-4f, 4.0f  },  /* odd length   */
};
#define N_SEGMENTS (sizeof(k_segments) / sizeof(k_segments[0]))

typedef struct { SpectralTimbre timbre; const char* name; float width; } ParityTimbre;

static const ParityTimbre k_timbres[] = {
    { TIMBRE_SINE,      "sine",      0.0f },
    { TIMBRE_SAW,       "saw",       0.0f },
    { TIMBRE_SQUARE,    "square",    0.0f },
    { TIMBRE_TRIANGLE,  "triangle",  0.0f },
    { TIMBRE_PARABOLA,  "parabola",  0.0f },
    { TIMBRE_QUANTIZED, "quantized", 8.0f },
    { TIMBRE_PWM,       "pwm",       0.5f },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

static void fill_lp(SegmentLoopParams* lp, const ParitySegment* s, float width) {
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

static void test_parity(void) {
    double agg_sumsq_diff = 0.0;   /* sum over ALL samples of (simd-scalar)^2 */
    double agg_sumsq_sig  = 0.0;   /* sum over ALL samples of scalar^2        */
    size_t agg_n          = 0;
    double agg_max_abs    = 0.0;

    for (size_t t = 0; t < N_TIMBRES; t++) {
        double tmax = 0.0;
        for (size_t si = 0; si < N_SEGMENTS; si++) {
            SegmentLoopParams lp;
            fill_lp(&lp, &k_segments[si], k_timbres[t].width);

            float buf_scalar[MAX_LEN];
            float buf_simd[MAX_LEN];
            memset(buf_scalar, 0, sizeof(buf_scalar));
            memset(buf_simd,   0, sizeof(buf_simd));

            osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR);
            timbre_synth_segment(buf_scalar, &lp, k_timbres[t].timbre);

            osc_set_dispatch(OSC_DISPATCH_ALL_SIMD);
            timbre_synth_segment(buf_simd, &lp, k_timbres[t].timbre);

            for (size_t j = 0; j < lp.length; j++) {
                double d = (double)buf_simd[j] - (double)buf_scalar[j];
                double ad = fabs(d);
                if (ad > tmax) tmax = ad;
                if (ad > agg_max_abs) agg_max_abs = ad;
                agg_sumsq_diff += d * d;
                agg_sumsq_sig  += (double)buf_scalar[j] * (double)buf_scalar[j];
                agg_n++;
            }
        }
        printf("  %-9s max abs diff = %.3e\n", k_timbres[t].name, tmax);
        CHECK(tmax <= BUDGET_ABS, "%s SIMD/scalar drift %.3e > budget %.1e",
              k_timbres[t].name, tmax, BUDGET_ABS);
    }

    double rms_diff = (agg_n > 0) ? sqrt(agg_sumsq_diff / (double)agg_n) : 0.0;
    double rms_sig  = (agg_n > 0) ? sqrt(agg_sumsq_sig  / (double)agg_n) : 0.0;
    double rms_dbfs   = (rms_diff > 0.0) ? 20.0 * log10(rms_diff) : -999.0;
    double rms_rel_db = (rms_diff > 0.0 && rms_sig > 0.0)
                        ? 20.0 * log10(rms_diff / rms_sig) : -999.0;

    printf("\n  aggregate samples = %zu\n", agg_n);
    printf("  max abs diff      = %.3e (budget %.1e)\n", agg_max_abs, BUDGET_ABS);
    printf("  RMS diff          = %.3e  (%.1f dBFS, budget %.1f)\n",
           rms_diff, rms_dbfs, (double)BUDGET_RMS_DBFS);
    printf("  RMS diff vs signal= %.1f dB (budget %.1f)\n",
           rms_rel_db, (double)BUDGET_RMS_REL_DB);

    CHECK(agg_max_abs <= BUDGET_ABS,
          "aggregate max abs drift %.3e > budget %.1e", agg_max_abs, BUDGET_ABS);
    CHECK(rms_dbfs <= BUDGET_RMS_DBFS,
          "RMS drift %.1f dBFS > budget %.1f dBFS", rms_dbfs, (double)BUDGET_RMS_DBFS);
    CHECK(rms_rel_db <= BUDGET_RMS_REL_DB,
          "RMS-vs-signal drift %.1f dB > budget %.1f dB", rms_rel_db, (double)BUDGET_RMS_REL_DB);
}

int main(void) {
    printf("== SIMD-vs-scalar oscillator parity (drift + RMS-dBFS) ==\n");
    test_parity();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
