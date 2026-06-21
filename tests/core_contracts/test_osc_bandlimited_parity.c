/* test_osc_bandlimited_parity.c - the ADDITIVE band-limited oscillator must equal the
 * analytic Nyquist-capped Fourier series for each algebraic timbre.
 *
 * spectral_osc_bandlimited_synth_segment(ADDITIVE) builds each waveform from a Chebyshev
 * recurrence for sin(k.theta)/cos(k.theta). This pins it against an INDEPENDENT
 * double-precision libm sum of the same truncated series, so it catches: a wrong/missing
 * harmonic-coefficient, a sin/cos (quadrature) swap, an off-by-one in the Nyquist harmonic
 * cap n_harm = floor(0.5/dt), and a regression in the per-timbre recurrence specialization
 * (each timbre advances only the sin- OR cos-recurrence it consumes). The module otherwise
 * has no coverage.
 *
 * Reference (double): theta = normalize_phase(phase0 + alpha*j); dt = |alpha|/2pi clamped
 * to [1e-6,0.5]; n_harm = clamp(floor(0.5/dt), 1, 512); then the standard series:
 *   saw      = (2/pi)  sum_{k=1..nh}      ((-1)^k / k)   sin(k.theta)
 *   square   = (4/pi)  sum_{k odd <=nh}   (1/k)          sin(k.theta)
 *   triangle = (8/pi^2)sum_{k odd <=nh}   (1/k^2)        cos(k.theta)
 *   parabola = 2/3 - (4/pi^2) sum_{k=1..nh} ((-1)^k/k^2) cos(k.theta)
 * scaled by amp_at(amp0,d_amp,j) * fade_envelope(j) exactly as the engine does.
 *
 * Run: cmake --build build --target osc_bandlimited_parity_test
 *      && ctest --test-dir build -R osc_bandlimited_parity
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "spectral_common.h"
#include "spectral_consts.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"     /* spectral_normalize_phase (inline) */
#include "spectral_osc_bandlimited.h"
#include "spectral_segment_math.h"
#include "spectral_synth_internal.h"   /* SegmentLoopParams */

#include "../support/check.h"

#define N 600u   /* > 2*SPECTRAL_FADE_SAMPLES_ACTIVE so there is a no-fade sustain region */

/* Analytic Nyquist-capped series in double; mirrors the engine's harmonic cap exactly. */
static double ref_wave(SpectralTimbre timbre, double theta, int n_harm) {
    double s = 0.0;
    if (timbre == TIMBRE_SINE) return sin(theta);
    for (int k = 1; k <= n_harm; k++) {
        double fk = (double)k;
        double sign = (k & 1) ? -1.0 : 1.0;   /* (-1)^k */
        switch (timbre) {
        case TIMBRE_SAW:      s += (sign / fk) * sin(fk * theta); break;
        case TIMBRE_SQUARE:   if (k & 1) s += (1.0 / fk) * sin(fk * theta); break;
        case TIMBRE_TRIANGLE: if (k & 1) s += (1.0 / (fk * fk)) * cos(fk * theta); break;
        case TIMBRE_PARABOLA: s += (sign / (fk * fk)) * cos(fk * theta); break;
        default: break;
        }
    }
    switch (timbre) {
    case TIMBRE_SAW:      return (2.0 / M_PI) * s;
    case TIMBRE_SQUARE:   return (4.0 / M_PI) * s;
    case TIMBRE_TRIANGLE: return (8.0 / (M_PI * M_PI)) * s;
    case TIMBRE_PARABOLA: return (2.0 / 3.0) - (4.0 / (M_PI * M_PI)) * s;
    default:              return 0.0;
    }
}

static int n_harm_for(float alpha) {
    float dt = fabsf(alpha) * SPECTRAL_INV_TWO_PI;
    if (!(dt > 1e-6f)) dt = 1e-6f;
    else if (dt > 0.5f) dt = 0.5f;
    int nh = (int)floorf(0.5f / dt);
    if (nh < 1) nh = 1;
    else if (nh > 512) nh = 512;
    return nh;
}

static void test_timbre(SpectralTimbre timbre, const char* name, float alpha, double tol) {
    SegmentLoopParams lp;
    memset(&lp, 0, sizeof lp);
    lp.start_idx = 0;
    lp.length = N;
    lp.alpha = alpha;          /* phase increment per sample (stationary: beta/c2/c3 = 0) */
    lp.phase = 0.37f;
    lp.amp = 0.8f;
    lp.d_amp = 0.0f;
    lp.valid = 1;

    static float dst[N];
    memset(dst, 0, sizeof dst);
    int ok = spectral_osc_bandlimited_synth_segment(dst, &lp, timbre,
                                                    SPECTRAL_OSC_QUALITY_ADDITIVE);
    CHECK(ok == 1, "%s: additive must render", name);
    if (!ok) return;

    FadeParams fp = fade_params_init(N, SPECTRAL_FADE_SAMPLES_ACTIVE);
    int n_harm = n_harm_for(alpha);
    double worst = 0.0;
    for (size_t j = 0; j < N; j++) {
        float pj = spectral_segment_phase_at_cubic_f32(lp.phase, lp.alpha, 0.0f, 0.0f, (float)j);
        double theta = (double)spectral_normalize_phase(pj);
        double wave = ref_wave(timbre, theta, n_harm);
        double amp = (double)(spectral_segment_amp_at_f32(lp.amp, lp.d_amp, (float)j)
                              * fade_envelope(j, &fp, N));
        double want = amp * wave;
        double d = fabs((double)dst[j] - want);
        if (d > worst) worst = d;
    }
    printf("  %-9s alpha=%.4f n_harm=%d max|engine-analytic|=%.3e\n", name, alpha, n_harm, worst);
    CHECK(worst < tol, "%s additive must track the analytic series (max %.3e >= %.1e)",
          name, worst, tol);
}

int main(void) {
    printf("test_osc_bandlimited_additive_parity:\n");
    /* alpha = 0.10 rad/sample -> n_harm ~ 31 (top harmonic just below Nyquist). The Chebyshev
     * recurrence drifts vs the double libm sum over the harmonics, so the tolerance is loose
     * enough for that float drift but tight enough to catch a wrong coefficient (~O(1)),
     * a sin/cos swap, or an off-by-one in the harmonic cap. */
    const float alpha = 0.10f;
    const double tol = 2.0e-3;
    test_timbre(TIMBRE_SAW,      "saw",      alpha, tol);
    test_timbre(TIMBRE_SQUARE,   "square",   alpha, tol);
    test_timbre(TIMBRE_TRIANGLE, "triangle", alpha, tol);
    test_timbre(TIMBRE_PARABOLA, "parabola", alpha, tol);
    test_timbre(TIMBRE_SINE,     "sine",     alpha, 1.0e-5);
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
