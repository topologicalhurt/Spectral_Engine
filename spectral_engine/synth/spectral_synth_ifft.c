/* spectral_synth_ifft.c - IFFT synthesis frame builder + OLA renderer.
 *
 * Backend-independent half: the F1-validated construction (see the contract
 * header and IFFT_SYNTHESIS_PLAN.md). The motif table is computed once at
 * create() from the same periodic-Hann window the OLA uses, in double, then
 * stored float — the F1 harness measured interpolation error negligible at
 * O=16 and float storage sits ~25 dB below the K=8 truncation floor.
 */
#include "spectral_synth_ifft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_consts.h"

/* F1-measured operating point (IFFT_SYNTHESIS_PLAN error budget): K=8 taps
 * per side -> -55 dBFS frame / -83 dBFS stream RMS floor; O=16 oversampling
 * is within 1 dB of O=64. K is an explicit F2/F3 knob: raising it buys
 * ~8 dB per +4 taps at ~2K complex MACs per partial. */
#define IFFT_MOTIF_K 8
#define IFFT_MOTIF_O 16

struct SpectralIfftSynth {
    size_t n_fft;
    size_t hop;
    SpectralIfftBackend* backend;
    float* motif;     /* 2*K*O+1 samples of the centered window spectrum */
    float* window;    /* periodic Hann, n_fft (kept for tests/inspection) */
    float* re;        /* half-spectrum scratch, n_fft/2 */
    float* im;
    float* frame;     /* time-domain frame scratch, n_fft */
};

static int is_pow2(size_t v) { return v && (v & (v - 1)) == 0; }

SpectralIfftSynth* spectral_ifft_synth_create(size_t n_fft) {
    if (!is_pow2(n_fft) || n_fft < 64) return NULL;

    SpectralIfftSynth* s = (SpectralIfftSynth*)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->n_fft = n_fft;
    s->hop = n_fft / 2;

    size_t mlen = 2u * IFFT_MOTIF_K * IFFT_MOTIF_O + 1u;
    s->backend = spectral_ifft_backend_create(n_fft);
    s->motif = (float*)malloc(mlen * sizeof(float));
    s->window = (float*)malloc(n_fft * sizeof(float));
    s->re = (float*)calloc(n_fft / 2, sizeof(float));
    s->im = (float*)calloc(n_fft / 2, sizeof(float));
    s->frame = (float*)malloc(n_fft * sizeof(float));
    if (!s->backend || !s->motif || !s->window || !s->re || !s->im || !s->frame) {
        spectral_ifft_synth_destroy(s);
        return NULL;
    }

    for (size_t n = 0; n < n_fft; n++) {
        s->window[n] = (float)(0.5 - 0.5 * cos(2.0 * SPECTRAL_PI_D * (double)n / (double)n_fft));
    }

    /* M(d) = sum_m v[m] cos(2 pi d m / N), v = window circularly centered at 0.
     * Double accumulation; the table is tiny (257 floats at K=8/O=16). */
    for (size_t i = 0; i < mlen; i++) {
        double d = ((double)i - (double)(IFFT_MOTIF_K * IFFT_MOTIF_O)) / IFFT_MOTIF_O;
        double acc = 0.0;
        for (long m = -(long)n_fft / 2; m < (long)n_fft / 2; m++) {
            double v = 0.5 - 0.5 * cos(2.0 * SPECTRAL_PI_D
                                       * (double)((m + (long)n_fft + (long)n_fft / 2) % (long)n_fft)
                                       / (double)n_fft);
            acc += v * cos(2.0 * SPECTRAL_PI_D * d * (double)m / (double)n_fft);
        }
        s->motif[i] = (float)acc;
    }
    return s;
}

void spectral_ifft_synth_destroy(SpectralIfftSynth* s) {
    if (!s) return;
    spectral_ifft_backend_destroy(s->backend);
    free(s->motif);
    free(s->window);
    free(s->re);
    free(s->im);
    free(s->frame);
    free(s);
}

size_t spectral_ifft_synth_n_fft(const SpectralIfftSynth* s) {
    return s ? s->n_fft : 0;
}

static inline float motif_eval(const float* t, float d) {
    float x = (d + (float)IFFT_MOTIF_K) * (float)IFFT_MOTIF_O;
    int i0 = (int)x;
    int len = 2 * IFFT_MOTIF_K * IFFT_MOTIF_O + 1;
    if (i0 < 0) i0 = 0;
    if (i0 >= len - 1) i0 = len - 2;
    float f = x - (float)i0;
    return t[i0] * (1.0f - f) + t[i0 + 1] * f;
}

/* Place one partial into the packed half-spectrum (Hermitian implicit). */
static void place_partial(SpectralIfftSynth* s, float bin, float amp, float phi_c) {
    float cr = 0.5f * amp * cosf(phi_c);
    float ci = 0.5f * amp * sinf(phi_c);
    int k0 = (int)floorf(bin);
    int half = (int)(s->n_fft / 2);
    for (int tap = -IFFT_MOTIF_K + 1; tap <= IFFT_MOTIF_K; tap++) {
        int k = k0 + tap;
        if (k <= 0 || k >= half) continue;     /* DC/Nyquist kept clear */
        float m = motif_eval(s->motif, (float)k - bin);
        float sgn = (k & 1) ? -1.0f : 1.0f;    /* (-1)^k centering twiddle */
        s->re[k] += sgn * m * cr;
        s->im[k] += sgn * m * ci;
    }
}

int spectral_ifft_synth_render(SpectralIfftSynth* s,
                               const SpectralIfftPartial* partials, size_t n,
                               float* out, size_t total) {
    if (!s || (!partials && n > 0) || !out || total == 0) return 1;

    /* Validate at the boundary, ONCE: the float->int floor in place_partial
     * is int-conversion UB on garbage bins (the defect class the Segment
     * width bound closes elsewhere), and the contract domain is the header's
     * bin in (1, n_fft/2 - 1). The negated comparisons also reject NaN. */
    const size_t half = s->n_fft / 2;
    const float bin_max = (float)half - 1.0f;
    for (size_t p = 0; p < n; p++) {
        if (!(partials[p].bin > 1.0f) || !(partials[p].bin < bin_max) ||
            !isfinite(partials[p].amp) || !isfinite(partials[p].phase0)) {
            return 1;
        }
    }
    memset(out, 0, total * sizeof(float));

    /* Frames at m*hop for m = -1.. so every output sample has full COLA
     * coverage (frame m covers [m*hop, m*hop + n_fft)). */
    long m_first = -1;
    long m_last = (long)((total - 1) / s->hop);
    for (long m = m_first; m <= m_last; m++) {
        double center = (double)m * (double)s->hop + (double)s->n_fft / 2.0;

        /* Full half-spectrum clear per frame, though only the K-bin
         * neighborhood of each partial gets written: at N=512 this is 2 KB
         * per frame against ~2K-tap placement work per partial, and the path
         * still measured 7.5x over the oscillator loop — not the bottleneck.
         * Revisit with a dirty-region clear in the ARM port, where the
         * memory system is the constraint. */
        memset(s->re, 0, (s->n_fft / 2) * sizeof(float));
        memset(s->im, 0, (s->n_fft / 2) * sizeof(float));
        for (size_t p = 0; p < n; p++) {
            double omega = 2.0 * SPECTRAL_PI_D * (double)partials[p].bin / (double)s->n_fft;
            double phi = fmod(omega * center + (double)partials[p].phase0, 2.0 * SPECTRAL_PI_D);
            place_partial(s, partials[p].bin, partials[p].amp, (float)phi);
        }

        spectral_ifft_backend_inverse(s->backend, s->re, s->im, s->frame);

        long base = m * (long)s->hop;
        size_t n0 = (base < 0) ? (size_t)(-base) : 0u;
        for (size_t i = n0; i < s->n_fft; i++) {
            size_t t = (size_t)(base + (long)i);
            if (t >= total) break;
            out[t] += s->frame[i];
        }
    }
    return 0;
}
