/* test_subtractive_filter_render.c - subtractive source x filter integration
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3b-2)
 *
 * Composes the whole subtractive chain end-to-end and proves the output spectrum = source x H(f):
 *   1. a saw-like wavetable source (harmonics 1..8, amplitude 1/k) -> harmonics (3a-1)
 *   2. expand into additive partials at a fundamental
 *   3. apply a low-pass per-band filter envelope (3b-1)
 *   4. render the filtered partials through the real IFFT synth
 *   5. INDEPENDENTLY measure each harmonic's amplitude in the rendered output (single-bin DFT)
 *      and assert it equals source_amp_k * H(freq_k): passband harmonics survive, stopband
 *      harmonics are cut.
 *
 * Integer-bin fundamentals + TOTAL = 8*n_fft make the render float-exact and the single-bin DFT
 * leakage-free, so this isolates the FILTER + composition correctness (the IFFT approximation
 * floor itself is pinned separately by wavetable_render_parity at fractional bins). Fail-on-bug:
 * skip the filter apply and the stopband harmonics come back -> the attenuation checks fail.
 */
#include "spectral_wavetable_harmonics.h"
#include "spectral_filter_envelope.h"
#include "spectral_synth_ifft.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TWO_PI 6.283185307179586476925286766559
#define N_FFT 512u
#define TOTAL (8u * N_FFT)   /* integer periods for every integer bin -> leakage-free measurement */
#define SR 48000.0f

static int g_fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); g_fails++; } } while (0)

/* amplitude of the cosine component at fractional `bin` over the output, via a single-bin DFT */
static double measure_amp(const float* out, size_t total, double bin)
{
    double ac = 0.0, as = 0.0;
    size_t i;
    for (i = 0; i < total; i++) {
        double a = TWO_PI * bin * (double)i / (double)N_FFT;
        ac += (double)out[i] * cos(a);
        as += (double)out[i] * sin(a);
    }
    ac *= 2.0 / (double)total;
    as *= 2.0 / (double)total;
    return sqrt(ac * ac + as * as);
}

int main(void)
{
    const size_t N = (size_t)SPECTRAL_WAVETABLE_SIZE;
    SpectralWavetable* t = (SpectralWavetable*)calloc(1, sizeof(SpectralWavetable));
    float amp_k[9], phase_k[9];
    size_t n_harm, n, np, k;
    const float f0_bin = 6.0f;           /* integer -> harmonics at integer bins 6,12,...,48 */
    SpectralIfftPartial parts[64];
    float src_amp[9];                    /* unfiltered partial amplitudes, indexed by harmonic k */
    SpectralFilterEnvelope env = {0};
    SpectralIfftSynth* s;
    float* out;
    float bin_to_hz = SR / (float)N_FFT;

    if (!t) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    /* saw-like single-cycle table: harmonics 1..8 with 1/k amplitude */
    for (n = 0; n < N; n++) {
        double th = (double)n / (double)N, acc = 0.0;
        for (k = 1; k <= 8; k++) acc += (1.0 / (double)k) * cos(TWO_PI * (double)k * th);
        t->samples[n] = (float)acc;
    }
    t->samples[N] = t->samples[0];

    n_harm = spectral_wavetable_harmonics(t, amp_k, phase_k, 8);

    /* expand the source (unfiltered); record the source amplitude per harmonic */
    np = spectral_wavetable_expand(amp_k, phase_k, n_harm, f0_bin, 1.0f, 0.0f, N_FFT, parts, 64);
    CHECK(np == 8, "expanded 8 source harmonics");
    for (k = 0; k < np; k++) src_amp[k + 1] = parts[k].amp;   /* parts[0] = harmonic 1 */

    /* low-pass: flat to 2000 Hz, ramp to zero at 4000 Hz */
    env.freq_hz[0] = 0.0f;    env.gain[0] = 1.0f;
    env.freq_hz[1] = 2000.0f; env.gain[1] = 1.0f;
    env.freq_hz[2] = 4000.0f; env.gain[2] = 0.0f;
    env.n_points = 3;

    /* apply the filter, then render the filtered partials */
    spectral_filter_envelope_apply(&env, parts, np, SR, N_FFT);
    s = spectral_ifft_synth_create(N_FFT);
    out = (float*)malloc(TOTAL * sizeof(float));
    if (!s || !out) { fprintf(stderr, "FAIL: setup\n"); return 1; }
    CHECK(spectral_ifft_synth_render(s, parts, np, out, TOTAL) == 0,
          "ifft render accepted every filtered partial (all within the renderer domain)");

    /* measure each harmonic in the output, assert == source_amp_k * H(freq_k) */
    for (k = 1; k <= np; k++) {
        double bin = (double)k * (double)f0_bin;
        double freq = bin * (double)bin_to_hz;
        double expect = (double)src_amp[k] * (double)spectral_filter_envelope_gain(&env, (float)freq);
        double measured = measure_amp(out, TOTAL, bin);
        char msg[96];
        snprintf(msg, sizeof(msg), "harmonic %zu: measured %.4f == source*H %.4f (%.0f Hz)",
                 k, measured, expect, freq);
        CHECK(fabs(measured - expect) < 5e-3, msg);
    }

    /* the filter must have an EFFECT (not a no-op): passband survives, stopband is cut */
    CHECK(measure_amp(out, TOTAL, 1.0 * f0_bin) > 0.9, "passband harmonic 1 (562 Hz) survives");
    CHECK(measure_amp(out, TOTAL, 8.0 * f0_bin) < 0.01, "stopband harmonic 8 (4500 Hz) is cut");

    spectral_ifft_synth_destroy(s);
    free(out);
    free(t);
    if (g_fails == 0) fprintf(stderr, "PASS: subtractive source x filter render\n");
    return g_fails ? 1 : 0;
}
