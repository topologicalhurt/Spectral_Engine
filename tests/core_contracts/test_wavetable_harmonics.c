/* test_wavetable_harmonics.c - wavetable frequency-domain bridge (RENDERER_ABSTRACTION_PLAN.md 3a).
 *
 * Pins spectral_wavetable_harmonics (DFT extraction) + spectral_wavetable_expand (harmonic ->
 * additive IFFT partials) against a synthetic table with KNOWN harmonics, so the freq-domain
 * representation is provably consistent with the table's own samples:
 *   - extraction recovers the planted harmonic amplitudes + phases (and no spurious harmonics);
 *   - the harmonic series reconstructs the table samples (round-trip RMS tiny);
 *   - expansion places harmonic k at bin k*f0 with amp*amp_k and phase k*phase0+phase_k, and
 *     truncates at Nyquist.
 * Fail-on-bug: perturb the extraction scaling/phase or the expansion mapping and a check fails.
 */
#include "spectral_wavetable_harmonics.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TWO_PI 6.283185307179586476925286766559

static int g_fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { fprintf(stderr, "FAIL: %s\n", (msg)); g_fails++; } } while (0)
static int approx(double a, double b, double tol) { return fabs(a - b) <= tol; }

int main(void)
{
    const size_t N = (size_t)SPECTRAL_WAVETABLE_SIZE;
    /* planted harmonics: k=1 (A=1.0, P=0.3) and k=3 (A=0.5, P=1.1) */
    const double A1 = 1.0, P1 = 0.3, A3 = 0.5, P3 = 1.1;
    SpectralWavetable* t = (SpectralWavetable*)calloc(1, sizeof(SpectralWavetable));
    float amp_k[17], phase_k[17];
    size_t n_harm, n, k;
    double rms = 0.0;
    SpectralIfftPartial out[64];
    size_t count;

    if (!t) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    for (n = 0; n < N; n++) {
        double th = (double)n / (double)N;
        t->samples[n] = (float)(A1 * cos(TWO_PI * 1.0 * th + P1) +
                                A3 * cos(TWO_PI * 3.0 * th + P3));
    }
    t->samples[N] = t->samples[0];

    /* extraction recovers the planted harmonics, nothing else */
    n_harm = spectral_wavetable_harmonics(t, amp_k, phase_k, 16);
    CHECK(n_harm == 17, "harmonic count = 17 (k=0..16)");
    CHECK(approx(amp_k[0], 0.0, 1e-3), "no DC");
    CHECK(approx(amp_k[1], A1, 1e-3) && approx(phase_k[1], P1, 1e-3), "harmonic 1 amp+phase");
    CHECK(approx(amp_k[2], 0.0, 1e-3), "harmonic 2 absent");
    CHECK(approx(amp_k[3], A3, 1e-3) && approx(phase_k[3], P3, 1e-3), "harmonic 3 amp+phase");
    for (k = 4; k <= 16; k++) CHECK(approx(amp_k[k], 0.0, 1e-3), "harmonics 4..16 absent");

    /* round-trip: the harmonic series reconstructs the table samples */
    for (n = 0; n < N; n++) {
        double th = (double)n / (double)N;
        double r = (double)amp_k[0];
        double d;
        for (k = 1; k < n_harm; k++) r += (double)amp_k[k] * cos(TWO_PI * (double)k * th + (double)phase_k[k]);
        d = r - (double)t->samples[n];
        rms += d * d;
    }
    rms = sqrt(rms / (double)N);
    CHECK(rms < 1e-3, "round-trip reconstruction RMS < 1e-3");

    /* expansion: harmonic k -> bin k*f0, amp*amp_k, phase k*phase0+phase_k */
    count = spectral_wavetable_expand(amp_k, phase_k, n_harm,
                                      /*f0_bin*/10.0f, /*amp*/2.0f, /*phase0*/0.5f,
                                      /*n_fft*/512u, out, 64);
    CHECK(count == 16, "expand deposits k=1..16 (all below Nyquist bin 256)");
    CHECK(approx(out[0].bin, 10.0, 1e-4) && approx(out[0].amp, 2.0 * A1, 1e-4) &&
          approx(out[0].phase0, 1.0 * 0.5 + P1, 1e-4), "expanded harmonic 1");
    CHECK(approx(out[2].bin, 30.0, 1e-4) && approx(out[2].amp, 2.0 * A3, 1e-4) &&
          approx(out[2].phase0, 3.0 * 0.5 + P3, 1e-4), "expanded harmonic 3");

    /* Nyquist truncation: f0_bin=20, n_fft=512 -> bin 256; k*20<256 keeps k=1..12 */
    count = spectral_wavetable_expand(amp_k, phase_k, n_harm, 20.0f, 1.0f, 0.0f, 512u, out, 64);
    CHECK(count == 12, "Nyquist truncation keeps k=1..12 at f0_bin=20");

    free(t);
    if (g_fails == 0) fprintf(stderr, "PASS: wavetable harmonics bridge\n");
    return g_fails ? 1 : 0;
}
