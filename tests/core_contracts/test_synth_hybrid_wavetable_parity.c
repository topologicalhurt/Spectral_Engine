/* test_synth_hybrid_wavetable_parity.c - opt-in wavetable IFFT fast path
 * (RENDERER_ABSTRACTION_PLAN.md, Stage 3a-3)
 *
 * Mirrors test_synth_hybrid_parity.c for the wavetable analog. Asserts:
 *   1. an eligible dense wavetable scene RENDERS via the IFFT fast path;
 *   2. deep-interior parity vs the exact band-limited additive sum of the expanded harmonics
 *      (the ideal the IFFT targets — NOT the aliasing table read, which it deliberately differs
 *      from; that acceptance is the F3 judgment);
 *   3. the decline contract: ineligible inputs return DECLINED with out[] left byte-untouched.
 */
#include "spectral_synth_hybrid_wavetable.h"
#include "spectral_wavetable_harmonics.h"
#include "spectral_synth_ifft.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TWO_PI 6.283185307179586476925286766559
#define N_FFT  512u
#define TOTAL  4096u
#define N_SEG  8u

static int g_fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", (m)); g_fails++; } } while (0)

int main(void)
{
    const size_t N = (size_t)SPECTRAL_WAVETABLE_SIZE;
    SpectralWavetable* table = (SpectralWavetable*)calloc(1, sizeof(SpectralWavetable));
    Segment segs[N_SEG];
    /* FRACTIONAL fundamentals so the harmonics land on fractional bins — the realistic case
     * that exercises the IFFT motif-truncation floor (integer bins are frame-periodic -> exact). */
    const float bins[N_SEG] = { 10.3f, 12.7f, 14.1f, 16.9f, 18.4f, 20.6f, 22.2f, 24.8f };
    SegmentArray sa;
    float* out;
    float amp_k[33], phase_k[33];
    size_t n_harm, i, t, p, np = 0;
    SpectralIfftPartial ref[256];
    double max_e = 0.0, sse = 0.0, rms, rms_db;
    size_t cnt = 0;
    float small[64];
    int untouched;
    Segment segs2[N_SEG];
    SegmentArray sa2;

    if (!table) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    /* saw-like single-cycle table: harmonics 1..8 with 1/k */
    for (i = 0; i < N; i++) {
        double th = (double)i / (double)N, a = 0.0; int k;
        for (k = 1; k <= 8; k++) a += (1.0 / (double)k) * cos(TWO_PI * (double)k * th);
        table->samples[i] = (float)a;
    }
    table->samples[N] = table->samples[0];
    table->valid = 1;
    table->timbre_id = 0;

    /* 8 full-duration stationary partials at distinct in-domain fundamentals (active over the
     * whole buffer + a frame margin so the deep interior is fully COLA-covered) */
    memset(segs, 0, sizeof(segs));
    for (i = 0; i < N_SEG; i++) {
        segs[i].start  = 0.0f;
        segs[i].length = (float)(TOTAL + 2u * N_FFT);
        segs[i].omega  = (float)(TWO_PI * (double)bins[i] / (double)N_FFT);
        segs[i].phase  = (float)(0.13 * (double)i);
        segs[i].amp    = 0.4f;
    }
    sa.segs = segs; sa.count = N_SEG;
    out = (float*)malloc(TOTAL * sizeof(float));
    if (!out) { fprintf(stderr, "FAIL: alloc\n"); return 1; }

    /* 1. eligible scene renders */
    CHECK(spectral_synth_hybrid_try_render_wavetable(sa, table, out, TOTAL, 1.0f, 0.0f)
          == SPECTRAL_HYBRID_RENDERED, "eligible dense wavetable scene renders via IFFT");

    /* 2. reference = additive sum of the SAME expanded partials (full duration -> constant set) */
    n_harm = spectral_wavetable_harmonics(table, amp_k, phase_k, 32);
    for (i = 0; i < N_SEG; i++) {
        double omega = (double)segs[i].omega;
        double f0_bin = omega * (double)N_FFT / TWO_PI;
        double base = (double)segs[i].phase - omega * floor((double)segs[i].start);
        base -= TWO_PI * floor(base / TWO_PI + 0.5);
        np += spectral_wavetable_expand(amp_k, phase_k, n_harm, (float)f0_bin, segs[i].amp,
                                        (float)base, N_FFT, ref + np, 256 - np);
    }
    for (t = 2u * N_FFT; t < TOTAL - 2u * N_FFT; t++) {
        double r = 0.0, e;
        for (p = 0; p < np; p++)
            r += (double)ref[p].amp * cos(TWO_PI * (double)ref[p].bin * (double)t / (double)N_FFT
                                          + (double)ref[p].phase0);
        e = (double)out[t] - r;
        if (fabs(e) > max_e) max_e = fabs(e);
        sse += e * e; cnt++;
    }
    rms = sqrt(sse / (double)cnt);
    rms_db = 20.0 * log10(rms + 1e-30);
    fprintf(stderr, "  wavetable hybrid vs additive sum (interior): max=%.3e rms=%.3e (%.1f dBFS) partials=%zu\n",
            max_e, rms, rms_db, np);
    /* measured -60.1 dBFS RMS (fractional bins, 123 partials — the IFFT motif-truncation floor);
     * gate at -52 dBFS = measured + ~8 dB headroom for vDSP-vs-FFTW variance. */
    CHECK(rms_db < -52.0, "deep-interior render parity below the -52 dBFS floor");

    /* 3. decline contract: out[] left strictly untouched (sentinel -123.5) */
    for (i = 0; i < 64; i++) small[i] = -123.5f;
    CHECK(spectral_synth_hybrid_try_render_wavetable(sa, table, small, 64, 1.0f, 3.0f)
          == SPECTRAL_HYBRID_DECLINED, "non-identity pitch declines");
    untouched = 1;
    for (i = 0; i < 64; i++) if (small[i] != -123.5f) untouched = 0;
    CHECK(untouched, "decline leaves out[] byte-untouched");
    CHECK(spectral_synth_hybrid_try_render_wavetable(sa, NULL, small, 64, 1.0f, 0.0f)
          == SPECTRAL_HYBRID_DECLINED, "null table declines");

    memcpy(segs2, segs, sizeof(segs));
    segs2[3].df = 1.0e-3f;                       /* a chirp -> ineligible -> all-or-decline */
    sa2.segs = segs2; sa2.count = N_SEG;
    CHECK(spectral_synth_hybrid_try_render_wavetable(sa2, table, small, 64, 1.0f, 0.0f)
          == SPECTRAL_HYBRID_DECLINED, "a single chirp segment declines the whole set");

    free(out); free(table);
    if (g_fails == 0) fprintf(stderr, "PASS: wavetable hybrid parity\n");
    return g_fails ? 1 : 0;
}
