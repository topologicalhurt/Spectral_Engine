/* test_wavetable_render_parity.c - wavetable freq-domain render parity
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3a-2)
 *
 * End-to-end: render a small wavetable SCENE through the IFFT domain — extract the table's
 * harmonics (3a-1 spectral_wavetable_harmonics), expand each wavetable partial into additive
 * partials (spectral_wavetable_expand), render them via the real spectral_ifft_synth_render —
 * and assert it matches the EXACT band-limited additive sum (the correct wavetable tone) at the
 * IFFT approximation floor.
 *
 * Combined with the 3a-1 round-trip (harmonics reconstruct the table samples), this proves the
 * freq-domain wavetable render reproduces the existing table's band-limited tone. A direct
 * comparison to the raw linear-interp table read (spectral_wavetable_lookup) is deliberately NOT
 * used: that read aliases + images above the table Nyquist, whereas the freq-domain render is
 * band-limited — they legitimately differ, so the additive sum is the correct reference (the same
 * one the existing ifft_synth_parity test uses for the sine path).
 */
#include "spectral_wavetable_harmonics.h"
#include "spectral_synth_ifft.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TWO_PI 6.283185307179586476925286766559
#define N_FFT 512u
#define TOTAL 4096u

static int g_fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); g_fails++; } } while (0)

int main(void)
{
    const size_t N = (size_t)SPECTRAL_WAVETABLE_SIZE;
    SpectralWavetable* t = (SpectralWavetable*)calloc(1, sizeof(SpectralWavetable));
    float amp_k[33], phase_k[33];
    size_t n_harm, n, i, k, p, np = 0;
    /* two wavetable partials at FRACTIONAL fundamentals (so harmonics land on fractional bins —
     * the realistic case that exercises the IFFT motif-truncation floor; integer bins are
     * periodic in the frame and reconstruct float-exact, which would not test the approximation) */
    const float f0_bin[2] = { 8.37f, 13.74f };
    const float p_amp[2]  = { 0.8f, 0.5f };
    const float p_ph[2]   = { 0.0f, 1.0f };
    SpectralIfftPartial parts[256];
    SpectralIfftSynth* s;
    float* outf;
    double max_err = 0.0, sse = 0.0, rms, rms_dbfs;

    if (!t) { fprintf(stderr, "FAIL: alloc table\n"); return 1; }

    /* a saw-like single-cycle table: harmonics 1..6 with 1/k amplitude */
    for (n = 0; n < N; n++) {
        double th = (double)n / (double)N, acc = 0.0;
        for (k = 1; k <= 6; k++) acc += (1.0 / (double)k) * cos(TWO_PI * (double)k * th);
        t->samples[n] = (float)acc;
    }
    t->samples[N] = t->samples[0];

    n_harm = spectral_wavetable_harmonics(t, amp_k, phase_k, 8);

    /* expand both wavetable partials into one additive partial set */
    for (p = 0; p < 2; p++)
        np += spectral_wavetable_expand(amp_k, phase_k, n_harm, f0_bin[p], p_amp[p], p_ph[p],
                                        N_FFT, parts + np, 256 - np);
    CHECK(np > 0, "expansion produced partials");

    /* freq-domain render via the real IFFT synth */
    s = spectral_ifft_synth_create(N_FFT);
    CHECK(s != NULL, "ifft synth create");
    outf = (float*)malloc(TOTAL * sizeof(float));
    if (!s || !outf) { fprintf(stderr, "FAIL: setup\n"); return 1; }
    CHECK(spectral_ifft_synth_render(s, parts, np, outf, TOTAL) == 0,
          "ifft render accepted every expanded partial (all within the renderer domain)");

    /* exact band-limited additive reference = the deposited partials' cosine sum */
    for (i = 0; i < TOTAL; i++) {
        double ref = 0.0, e;
        for (k = 0; k < np; k++)
            ref += (double)parts[k].amp *
                   cos(TWO_PI * (double)parts[k].bin * (double)i / (double)N_FFT + (double)parts[k].phase0);
        e = (double)outf[i] - ref;
        if (fabs(e) > max_err) max_err = fabs(e);
        sse += e * e;
    }
    rms = sqrt(sse / (double)TOTAL);
    rms_dbfs = 20.0 * log10(rms + 1e-30);
    fprintf(stderr, "  wavetable IFFT render vs additive sum: max=%.3e rms=%.3e (%.1f dBFS), partials=%zu\n",
            max_err, rms, rms_dbfs, np);

    /* IFFT fractional-bin motif-truncation floor: measured -63.1 dBFS RMS (this is the IFFT
     * approximation, the reason the path rides the F3 golden). Frozen at -55 dBFS = measured +
     * ~8 dB headroom for vDSP-vs-FFTW variance across platforms. */
    CHECK(rms_dbfs < -55.0, "render parity RMS below the -55 dBFS floor");

    spectral_ifft_synth_destroy(s);
    free(outf);
    free(t);
    if (g_fails == 0) fprintf(stderr, "PASS: wavetable render parity\n");
    return g_fails ? 1 : 0;
}
