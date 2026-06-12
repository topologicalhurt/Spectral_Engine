/* test_q15_compute_precision.c - Q15-compute vs float precision characterization (Q3a).
 *
 * Q3 of QTYPE_DOMAIN_PLAN.md adds an OPT-IN Q15 *compute* domain for throughput-
 * bound L1 oscillator paths. The plan's open question (§6) is which timbres clear
 * the 15-bit precision bar — computing audio in Q15 spends ~60 dB of the headroom
 * the float oscillator earns (measured -155 dBFS vs double truth), so each candidate path needs a
 * measured dBFS justification BEFORE it is wired into production. This harness is
 * that measurement. It changes NO production output: it renders the L1 waveforms
 * two ways and reports the per-timbre error, nothing more.
 *
 *   float reference :  spectral_osc_<t>(rads, w)        (the -155 dBFS truth)
 *   Q15 compute     :  q15_wave_<t>(phase_q15)           (pure fixed-point eval)
 *
 * SCOPE — this isolates Q15 WAVEFORM-EVALUATION precision. Both paths are driven
 * from the SAME phase (float `rads` mapped once to a Q15 phase), so the number is
 * the cost of evaluating the waveform in Q15, NOT the cost of an integer NCO's
 * phase quantization. NCO phase resolution is an orthogonal axis (a frequency-
 * resolution question, well understood) and is deliberately out of scope here; a
 * real Q15 production oscillator would carry both, but the per-path GO/NO-GO bar
 * the plan asks about is dominated by waveform-eval precision, which is what we
 * pin. The five timbres measured are the algebraically-clean / LUT throughput-
 * bound set; quantized/pwm/asin (width-/transcendental-shaped) are not Q15-compute
 * candidates and are excluded.
 *
 * The sine reference uses a LOCAL full-scale (Q15_MAX) sine LUT, not the
 * production spectral_lut_init_sine() table: production deliberately scales to
 * SPECTRAL_LUT_AMP_SCALE (32700) for linear-interp-overflow headroom, a ~-0.02 dB
 * intentional GAIN offset that is trivially correctable and would otherwise swamp
 * (~-54 dBFS) the quantization floor we are trying to read. Removing that gain
 * isolates the true noise floor.
 *
 * The CHECK budget is a loose CORRECTNESS sanity bound (catches a gross impl bug),
 * NOT the precision verdict — the printed dBFS table is the verdict, to be read by
 * the maintainer for the per-path promotion decision (QTYPE_DOMAIN_PLAN.md §7).
 *
 * Run: cmake --build build --target q15_compute_precision_test \
 *      && ctest --test-dir build -R q15_compute_precision
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "spectral_consts.h"
#include "spectral_osc_formulas.h"
#include "spectral_lut.h"
#include "spectral_q15.h"
#include "spectral_osc_q15.h"   /* the production Q15 evaluators under test */

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

/* Correctness sanity floor: any correct Q15 waveform eval beats this by 20+ dB.
 * This is a tripwire for gross bugs, not the precision bar — see the file header. */
#define SANITY_FLOOR_DBFS (-60.0)

/* Full-scale sine LUT (peak Q15_MAX, gain-matched to float — not the production
 * 32700 headroom). Built by the production helper so test == ship. */
static q15_t g_sine_lut[SPECTRAL_OSC_LUT_SIZE + 1u];

/* Thin forwarders to the production Q15 evaluators (spectral_osc_q15.h). The
 * formula math lives ONCE, in the header production also uses; these only adapt
 * the sine signature (LUT capture) to the uniform table below, so what this
 * test measures is exactly what production ships. */
static q15_t q15_wave_sine(q15_t pq)     { return spectral_osc_q15_sine(pq, g_sine_lut); }
static q15_t q15_wave_saw(q15_t pq)      { return spectral_osc_q15_saw(pq); }
static q15_t q15_wave_square(q15_t pq)   { return spectral_osc_q15_square(pq); }
static q15_t q15_wave_triangle(q15_t pq) { return spectral_osc_q15_triangle(pq); }
static q15_t q15_wave_parabola(q15_t pq) { return spectral_osc_q15_parabola(pq); }

typedef float  (*FloatWaveFn)(float rads, float width);
typedef q15_t  (*Q15WaveFn)(q15_t pq);

typedef struct { const char* name; FloatWaveFn fref; Q15WaveFn fq15; } Timbre;

static const Timbre k_timbres[] = {
    { "sine",     spectral_osc_sine,     q15_wave_sine     },
    { "saw",      spectral_osc_saw,      q15_wave_saw      },
    { "square",   spectral_osc_square,   q15_wave_square   },
    { "triangle", spectral_osc_triangle, q15_wave_triangle },
    { "parabola", spectral_osc_parabola, q15_wave_parabola },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

/* A spread of partials: low, mid, high, and an irrational ratio to avoid the
 * phase repeating on a short period. omega = 2*pi*f/fs, fs = 48000. */
static const float k_freqs_hz[] = { 110.0f, 440.0f, 2637.0f, 9000.123f };
#define N_FREQS (sizeof(k_freqs_hz) / sizeof(k_freqs_hz[0]))

#define N_SAMPLES 8192
#define SAMPLE_RATE 48000.0f
#define TEST_AMP 0.8f            /* below 1.0 so the Q15 amp cap is not the story */

static double measure_timbre(const Timbre* t) {
    q15_t amp_q15 = FLOAT_TO_Q15(TEST_AMP);
    double agg_sq = 0.0, agg_peak = 0.0;
    size_t n = 0;

    for (size_t fi = 0; fi < N_FREQS; fi++) {
        float omega = SPECTRAL_TWO_PI * k_freqs_hz[fi] / SAMPLE_RATE;
        float phase = 0.31f;     /* arbitrary non-zero start */
        for (size_t j = 0; j < N_SAMPLES; j++, phase += omega) {
            float rads = spectral_normalize_phase(phase);

            float ref = TEST_AMP * t->fref(rads, 0.0f);

            q15_t pq  = spectral_osc_q15_phase_from_rads(rads);
            q15_t wq  = t->fq15(pq);
            q15_t sq  = spectral_mul_q15(wq, amp_q15);
            float test = Q15_TO_FLOAT(sq);

            double e = (double)test - (double)ref;
            agg_sq += e * e;
            double ae = fabs(e);
            if (ae > agg_peak) agg_peak = ae;
            n++;
        }
    }

    double rms = sqrt(agg_sq / (double)n);
    double dbfs = (rms > 0.0) ? 20.0 * log10(rms) : -300.0;
    printf("  %-9s  RMS err = %.3e (%.1f dBFS)   peak err = %.3e\n",
           t->name, rms, dbfs, agg_peak);
    return dbfs;
}

int main(void) {
    spectral_osc_q15_init_sine_lut(g_sine_lut);
    printf("== Q15-compute vs float precision (waveform-eval, same phase) ==\n");
    printf("   ref = float L0 formula; test = pure-Q15 eval; amp = %.2f, fs = %.0f\n\n",
           TEST_AMP, SAMPLE_RATE);

    for (size_t i = 0; i < N_TIMBRES; i++) {
        double dbfs = measure_timbre(&k_timbres[i]);
        CHECK(dbfs <= SANITY_FLOOR_DBFS, "%s Q15 error %.1f dBFS worse than sanity floor %.1f",
              k_timbres[i].name, dbfs, SANITY_FLOOR_DBFS);
    }

    printf("\n  (sanity floor %.1f dBFS is a gross-bug tripwire, not the precision verdict;\n"
           "   the table above is the per-path GO/NO-GO data for QTYPE plan Q3.)\n",
           SANITY_FLOOR_DBFS);
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
