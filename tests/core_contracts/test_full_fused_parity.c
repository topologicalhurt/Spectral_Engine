/* test_full_fused_parity.c
 *
 * D1 (ULTRAPLAN Phase D) — compiled full/fused analysis parity harness.
 *
 * Implements docs/core_audit/FULL_FUSED_PARITY_HARNESS.md as a real CTest
 * (not a string-grep): deterministic fixtures are run through the production
 * analyze_audio_with_path_mode() under SPECTRAL_ANALYSIS_PATH_FULL and
 * SPECTRAL_ANALYSIS_PATH_FUSED, the two segment arrays are sorted by
 * (start, omega, amp), compared field-by-field within documented tolerances,
 * per-field max errors are printed, and the process exits non-zero on failure.
 *
 * The two paths share the estimator/window policy and differ only in
 * scheduling/allocation shape (full materializes the whole STFT matrix; fused
 * streams it frame-by-frame), so for deterministic fixtures they must agree on
 * segment content to within FP-reordering tolerances.
 *
 * Run: cmake --build build --target full_fused_parity_test \
 *      && ctest --test-dir build -R full_fused_parity
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_analysis_internal.h"

/* Deterministic analysis parameters (kept small so both paths run fast). */
#define PAR_SR        48000
#define PAR_N_FFT     1024
#define PAR_HOP       256
#define PAR_DB_THRESH (-80.0f)
#define PAR_N_SAMPLES 16384u

/* Tolerances from FULL_FUSED_PARITY_HARNESS.md ("Suggested initial"). */
#define TOL_START_SAMPLES 1.0   /* start: <= 1 sample                     */
#define TOL_LENGTH_HOP    ((double)PAR_HOP) /* length: <= hop             */
#define TOL_OMEGA_REL     1e-4
#define TOL_OMEGA_ABS     1e-6
#define TOL_AMP_REL       1e-3
#define TOL_AMP_ABS       1e-5
#define TOL_DFDA_REL      1e-3
#define TOL_DFDA_ABS      1e-5
/* count: exact (a divergence is a real defect, surfaced — not silently
 * tolerated). If a justified one-segment edge case is observed it is to be
 * documented here and the bound relaxed, per the measure-first discipline. */

typedef void (*fixture_fn)(float* buf, size_t n);

static const double TWO_PI = 6.283185307179586476925287;

static double bin_freq(double bin) { return bin * (double)PAR_SR / (double)PAR_N_FFT; }

static void fx_silence(float* buf, size_t n) { memset(buf, 0, n * sizeof(float)); }

static void fx_sine_bin_centered(float* buf, size_t n)
{
    const double f = bin_freq(64.0);          /* exactly on bin 64 */
    for (size_t i = 0; i < n; i++)
        buf[i] = 0.5f * (float)sin(TWO_PI * f * (double)i / (double)PAR_SR);
}

static void fx_sine_off_bin(float* buf, size_t n)
{
    const double f = bin_freq(64.5);          /* halfway between bins */
    for (size_t i = 0; i < n; i++)
        buf[i] = 0.5f * (float)sin(TWO_PI * f * (double)i / (double)PAR_SR);
}

static void fx_two_tone(float* buf, size_t n)
{
    const double f1 = bin_freq(40.0);
    const double f2 = bin_freq(120.0);
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)PAR_SR;
        buf[i] = (float)(0.40 * sin(TWO_PI * f1 * t) + 0.30 * sin(TWO_PI * f2 * t));
    }
}

static void fx_chirp(float* buf, size_t n)
{
    const double f0 = 1000.0, f1 = 5000.0;
    const double dur = (double)n / (double)PAR_SR;
    const double k = (f1 - f0) / dur;          /* linear sweep rate */
    for (size_t i = 0; i < n; i++) {
        double t = (double)i / (double)PAR_SR;
        double phase = TWO_PI * (f0 * t + 0.5 * k * t * t);
        buf[i] = 0.4f * (float)sin(phase);
    }
}

static void fx_amp_ramp(float* buf, size_t n)
{
    const double f = bin_freq(64.0);
    for (size_t i = 0; i < n; i++) {
        double a = 0.8 * (double)i / (double)n; /* 0 -> 0.8 */
        buf[i] = (float)(a * sin(TWO_PI * f * (double)i / (double)PAR_SR));
    }
}

static int seg_cmp(const void* a, const void* b)
{
    const Segment* x = (const Segment*)a;
    const Segment* y = (const Segment*)b;
    if (x->start < y->start) return -1;
    if (x->start > y->start) return 1;
    if (x->omega < y->omega) return -1;
    if (x->omega > y->omega) return 1;
    if (x->amp   < y->amp)   return -1;
    if (x->amp   > y->amp)   return 1;
    return 0;
}

/* abs/rel tolerance: pass if |a-b| <= abs OR |a-b| <= rel*max(|a|,|b|). */
static int within(double a, double b, double rel, double abs_tol)
{
    double d = fabs(a - b);
    if (d <= abs_tol) return 1;
    double m = fmax(fabs(a), fabs(b));
    return d <= rel * m;
}

typedef struct {
    double start, length, omega, amp, df, da;
} FieldMax;

static void track_max(FieldMax* fm, const Segment* x, const Segment* y)
{
    fm->start  = fmax(fm->start,  fabs((double)x->start  - (double)y->start));
    fm->length = fmax(fm->length, fabs((double)x->length - (double)y->length));
    fm->omega  = fmax(fm->omega,  fabs((double)x->omega  - (double)y->omega));
    fm->amp    = fmax(fm->amp,    fabs((double)x->amp    - (double)y->amp));
    fm->df     = fmax(fm->df,     fabs((double)x->df     - (double)y->df));
    fm->da     = fmax(fm->da,     fabs((double)x->da     - (double)y->da));
}

static SegmentArray run_path(const float* audio, size_t n, SpectralAnalysisPathMode mode)
{
    double t_fft = 0.0, t_track = 0.0;
    return analyze_audio_with_path_mode(audio, n, PAR_SR, PAR_N_FFT, PAR_HOP,
                                        PAR_DB_THRESH, mode, &t_fft, &t_track);
}

static int compare_fixture(const char* name, fixture_fn gen, int expect_empty)
{
    float* buf = (float*)malloc(PAR_N_SAMPLES * sizeof(float));
    if (!buf) { fprintf(stderr, "  %-16s OOM\n", name); return 1; }
    gen(buf, PAR_N_SAMPLES);

    SegmentArray full  = run_path(buf, PAR_N_SAMPLES, SPECTRAL_ANALYSIS_PATH_FULL);
    SegmentArray fused = run_path(buf, PAR_N_SAMPLES, SPECTRAL_ANALYSIS_PATH_FUSED);
    free(buf);

    int fail = 0;
    int printed = 0;  /* distinct from the boolean verdict so the print cap works */

    /* For fixtures that must yield NO segments (silence), assert the absolute
     * count is 0 on BOTH paths — equality alone would pass if both paths
     * spuriously emitted the same nonzero count. */
    if (expect_empty && (full.count != 0u || fused.count != 0u)) {
        fprintf(stderr, "  %-16s FAIL expected 0 segments: full=%u fused=%u\n",
                name, full.count, fused.count);
        fail = 1;
    }

    if (full.count != fused.count) {
        fprintf(stderr, "  %-16s FAIL count: full=%u fused=%u\n",
                name, full.count, fused.count);
        fail = 1;
    }

    if (full.count && fused.count) {
        qsort(full.segs,  full.count,  sizeof(Segment), seg_cmp);
        qsort(fused.segs, fused.count, sizeof(Segment), seg_cmp);
    }

    uint32_t m = full.count < fused.count ? full.count : fused.count;
    FieldMax fm = {0};
    for (uint32_t i = 0; i < m; i++) {
        const Segment* a = &full.segs[i];
        const Segment* b = &fused.segs[i];
        track_max(&fm, a, b);
        if (!within(a->start,  b->start,  0.0,           TOL_START_SAMPLES) ||
            !within(a->length, b->length, 0.0,           TOL_LENGTH_HOP)    ||
            !within(a->omega,  b->omega,  TOL_OMEGA_REL, TOL_OMEGA_ABS)     ||
            !within(a->amp,    b->amp,    TOL_AMP_REL,   TOL_AMP_ABS)       ||
            !within(a->df,     b->df,     TOL_DFDA_REL,  TOL_DFDA_ABS)      ||
            !within(a->da,     b->da,     TOL_DFDA_REL,  TOL_DFDA_ABS)) {
            if (printed++ < 32) {
                fprintf(stderr,
                        "  %-16s FAIL seg[%u]: "
                        "start %.6f/%.6f omega %.9f/%.9f amp %.9f/%.9f "
                        "len %.3f/%.3f df %.9f/%.9f da %.9f/%.9f\n",
                        name, i, a->start, b->start, a->omega, b->omega,
                        a->amp, b->amp, a->length, b->length,
                        a->df, b->df, a->da, b->da);
            }
            fail = 1;
        }
    }

    printf("  %-16s count=%u  max|d|: start=%.3e len=%.3e omega=%.3e amp=%.3e df=%.3e da=%.3e  %s\n",
           name, full.count, fm.start, fm.length, fm.omega, fm.amp, fm.df, fm.da,
           fail ? "FAIL" : "ok");

    spectral_segment_array_free(&full);
    spectral_segment_array_free(&fused);
    return fail;
}

int main(void)
{
    struct { const char* name; fixture_fn fn; int expect_empty; } fixtures[] = {
        { "silence",     fx_silence,           1 },
        { "sine_bin",    fx_sine_bin_centered, 0 },
        { "sine_offbin", fx_sine_off_bin,      0 },
        { "two_tone",    fx_two_tone,          0 },
        { "chirp",       fx_chirp,             0 },
        { "amp_ramp",    fx_amp_ramp,          0 },
    };
    const size_t nfix = sizeof(fixtures) / sizeof(fixtures[0]);

    printf("full/fused analysis parity (sr=%d n_fft=%d hop=%d db=%.0f n=%u)\n",
           PAR_SR, PAR_N_FFT, PAR_HOP, (double)PAR_DB_THRESH, PAR_N_SAMPLES);

    int fail = 0;
    for (size_t i = 0; i < nfix; i++)
        fail |= compare_fixture(fixtures[i].name, fixtures[i].fn, fixtures[i].expect_empty);

    if (fail) {
        fprintf(stderr, "full_fused_parity: FAILED\n");
        return 1;
    }
    printf("full_fused_parity: all %zu fixtures within tolerance\n", nfix);
    return 0;
}
