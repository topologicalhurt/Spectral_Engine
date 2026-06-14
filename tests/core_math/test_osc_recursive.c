/* test_osc_recursive.c - Coupled-form Q31 oscillator: fixed-point SNR/stability contract.
 *
 * Asserts the recursive oscillator that replaces the LUT gather stays within an SNR budget
 * vs exact sin over a frequency sweep and a multi-second sustain (the worst case for drift),
 * with per-block renorm. This is the make-or-break the magic-circle form failed (see plan):
 * if the polynomial/scaling regresses, or renorm breaks, SNR collapses and this fails.
 */
#include "spectral_osc_q31.h"
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FS        48000.0
#define NSEC      8.0          /* 8 s sustained partial */
#define BLOCK     256          /* renorm cadence = synth block size */
#define SNR_BUDGET 72.0        /* dB; coupled+renorm measured 83-128, LUT bar 75-86 */
#define DRIFT_MAX  1.0e-3      /* max |amp-1| over the run */

#include "../support/check.h"

static void test_freq(double freq) {
    double w = 2.0 * M_PI * freq / FS, phi = 0.31;
    SpectralCoupledOsc o;
    q31_t cos_w, sin_w;
    spectral_coupled_init(&o, w, phi, &cos_w, &sin_w);

    long n = (long)(FS * NSEC);
    double sig_pow = 0.0, err_pow = 0.0, max_drift = 0.0;
    for (long i = 0; i < n; i++) {
        q31_t sq = spectral_coupled_step(&o, cos_w, sin_w);
        if ((i % BLOCK) == (BLOCK - 1)) spectral_coupled_renorm(&o);
        double sf  = (double)sq / 2147483647.0;
        double cf  = (double)o.c / 2147483647.0;
        double ref = sin(w * (double)(i + 1) + phi);
        double e   = sf - ref;
        sig_pow += ref * ref;
        err_pow += e * e;
        double a = fabs(sqrt(cf * cf + sf * sf) - 1.0);
        if (a > max_drift) max_drift = a;
    }
    double snr = 10.0 * log10(sig_pow / (err_pow + 1e-30));
    printf("  %6.0f Hz: SNR %6.2f dB  drift %.2e\n", freq, snr, max_drift);
    CHECK(snr >= SNR_BUDGET, "%.0f Hz SNR %.2f < budget %.1f dB", freq, snr, SNR_BUDGET);
    CHECK(max_drift <= DRIFT_MAX, "%.0f Hz drift %.2e > %.1e", freq, max_drift, DRIFT_MAX);
}

int main(void) {
    printf("== coupled-form Q31 oscillator: SNR/stability over %.0f s, renorm/%d ==\n", NSEC, BLOCK);
    const double freqs[] = { 27.5, 55, 110, 440, 1000, 2000, 4000, 8000, 12000, 18000 };
    for (size_t i = 0; i < sizeof(freqs) / sizeof(freqs[0]); i++) test_freq(freqs[i]);
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
