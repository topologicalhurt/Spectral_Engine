/* ifft_synth_sweep.c - F1 math harness for IFFT (Rodet-Depalle) synthesis.
 *
 * Validates the frame-builder MATH before any engine integration
 * (IFFT_SYNTHESIS_PLAN.md F1) and measures the error-budget floors the plan
 * gates on. No engine dependencies on purpose: a local O(N^2) DFT is the
 * reference transform (the engine's FFT backends come in at F2 behind the
 * contract header).
 *
 * Construction under test (derivation in the plan):
 *   v[n] = w[(n + N/2) mod N]            circularly centered periodic Hann
 *   M(d) = sum_m v[m] cos(2*pi*d*m/N)    REAL even motif (v real-even)
 *   F[k] = (A/2) e^{j phi_c} (-1)^k M(k - b)   for k within K bins of b,
 *          plus the Hermitian mirror; b = fractional bin, phi_c = phase at
 *          frame center. iDFT(F) == w[n] * A cos(omega (n - N/2) + phi_c)
 *          up to motif truncation (K) + table interpolation (O) error.
 * OLA at hop N/2 with periodic Hann == COLA gain 1, so stationary partials
 * reconstruct exactly up to the same floors.
 *
 * Sections (parsed by the pytest driver):
 *   FRAME  k=<K> o=<O> max_err_dbfs=...   frame-level parity vs time domain
 *   STREAM k=<K> o=<O> partials=<P> max_err_dbfs=... rms_err_dbfs=...
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef IFFT_N
#define IFFT_N 512
#endif
#define HOP (IFFT_N / 2)
#define PI 3.14159265358979323846

/* xorshift for deterministic partial sets */
static unsigned rng_state = 0x9e3779b9u;
static double rng01(void) {
    unsigned x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return (double)(x & 0xffffffu) / 16777216.0;
}

static double win[IFFT_N];
static void make_window(void) {
    for (int n = 0; n < IFFT_N; n++)
        win[n] = 0.5 * (1.0 - cos(2.0 * PI * n / IFFT_N));   /* periodic Hann */
}

/* Oversampled real motif: M(d) for d in [-K, +K] step 1/O.
 * v[m] = w[(m + N/2) mod N]; circular index m in (-N/2, N/2]. */
static double* motif_build(int K, int O) {
    int len = 2 * K * O + 1;
    double* t = (double*)malloc((size_t)len * sizeof(double));
    for (int i = 0; i < len; i++) {
        double d = (double)(i - K * O) / O;
        double acc = 0.0;
        for (int m = -IFFT_N / 2; m < IFFT_N / 2; m++) {
            double v = win[(m + IFFT_N + IFFT_N / 2) % IFFT_N];
            acc += v * cos(2.0 * PI * d * m / IFFT_N);
        }
        t[i] = acc;
    }
    return t;
}

static double motif_eval(const double* t, int K, int O, double d) {
    /* linear interp; caller guarantees |d| <= K */
    double x = (d + K) * O;
    int i0 = (int)x;
    int len = 2 * K * O + 1;
    if (i0 >= len - 1) i0 = len - 2;
    double f = x - i0;
    return t[i0] * (1.0 - f) + t[i0 + 1] * f;
}

/* Add one partial into the complex spectrum (re/im arrays, length N). */
static void place_partial(double* re, double* im, const double* motif,
                          int K, int O, double amp, double bin, double phi_c) {
    double cr = 0.5 * amp * cos(phi_c);
    double ci = 0.5 * amp * sin(phi_c);
    int k0 = (int)floor(bin);
    for (int tap = -K + 1; tap <= K; tap++) {
        int k = k0 + tap;
        if (k <= 0 || k >= IFFT_N / 2) continue;   /* keep clear of DC/Nyquist */
        double m = motif_eval(motif, K, O, (double)k - bin);
        double sgn = (k & 1) ? -1.0 : 1.0;          /* (-1)^k centering twiddle */
        double vr = sgn * m * cr;
        double vi = sgn * m * ci;
        re[k] += vr;             im[k] += vi;       /* +b component */
        re[IFFT_N - k] += vr;    im[IFFT_N - k] -= vi;   /* Hermitian mirror */
    }
}

/* Reference inverse DFT (O(N^2)), real output. */
static void idft(const double* re, const double* im, double* out) {
    for (int n = 0; n < IFFT_N; n++) {
        double acc = 0.0;
        for (int k = 0; k < IFFT_N; k++) {
            double a = 2.0 * PI * (double)k * n / IFFT_N;
            acc += re[k] * cos(a) - im[k] * sin(a);
        }
        out[n] = acc / IFFT_N;
    }
}

static double dbfs(double x) { return 20.0 * log10(x > 1e-300 ? x : 1e-300); }

/* (a) frame-level parity: one partial, sweep fractional bins. */
static void frame_test(int K, int O) {
    const double* motif = motif_build(K, O);
    double re[IFFT_N], im[IFFT_N], frame[IFFT_N];
    double worst = 0.0;

    for (int c = 0; c < 24; c++) {
        double bin = 20.0 + 200.0 * rng01();
        double amp = 0.1 + 0.9 * rng01();
        double phi = 2.0 * PI * rng01() - PI;
        double omega = 2.0 * PI * bin / IFFT_N;

        memset(re, 0, sizeof re); memset(im, 0, sizeof im);
        place_partial(re, im, motif, K, O, amp, bin, phi);
        idft(re, im, frame);

        for (int n = 0; n < IFFT_N; n++) {
            double want = win[n] * amp * cos(omega * ((double)n - IFFT_N / 2.0) + phi);
            double err = fabs(frame[n] - want);
            if (err > worst) worst = err;
        }
    }
    printf("FRAME k=%d o=%d max_err_dbfs=%.2f\n", K, O, dbfs(worst));
    free((void*)motif);
}

/* (b) stream-level parity: P stationary partials, OLA across frames,
 * compared to the exact oscillator sum on the interior samples. */
static void stream_test(int K, int O, int P) {
    const int n_frames = 16;
    const int total = (n_frames - 1) * HOP + IFFT_N;
    const double* motif = motif_build(K, O);
    double* out = (double*)calloc((size_t)total, sizeof(double));
    double re[IFFT_N], im[IFFT_N], frame[IFFT_N];
    double binv[64], ampv[64], phiv[64];

    rng_state = 0xc0ffee11u;
    for (int p = 0; p < P; p++) {
        binv[p] = 8.0 + 230.0 * rng01();
        ampv[p] = (0.05 + 0.95 * rng01()) / P;     /* headroom */
        phiv[p] = 2.0 * PI * rng01() - PI;
    }

    for (int mfr = 0; mfr < n_frames; mfr++) {
        double center = (double)mfr * HOP + IFFT_N / 2.0;
        memset(re, 0, sizeof re); memset(im, 0, sizeof im);
        for (int p = 0; p < P; p++) {
            double omega = 2.0 * PI * binv[p] / IFFT_N;
            double phi_c = fmod(omega * center + phiv[p], 2.0 * PI);
            place_partial(re, im, motif, K, O, ampv[p], binv[p], phi_c);
        }
        idft(re, im, frame);
        for (int n = 0; n < IFFT_N; n++) out[mfr * HOP + n] += frame[n];
    }

    /* Interior region: full COLA coverage (skip the first/last half frame). */
    double worst = 0.0, sumsq = 0.0;
    int n0 = IFFT_N, n1 = total - IFFT_N, count = 0;
    for (int t = n0; t < n1; t++) {
        double want = 0.0;
        for (int p = 0; p < P; p++) {
            double omega = 2.0 * PI * binv[p] / IFFT_N;
            want += ampv[p] * cos(omega * t + phiv[p]);
        }
        double err = fabs(out[t] - want);
        if (err > worst) worst = err;
        sumsq += err * err;
        count++;
    }
    printf("STREAM k=%d o=%d partials=%d max_err_dbfs=%.2f rms_err_dbfs=%.2f\n",
           K, O, P, dbfs(worst), dbfs(sqrt(sumsq / count)));
    free(out);
    free((void*)motif);
}

int main(void) {
    make_window();
    printf("IFFTSWEEP n=%d hop=%d\n", IFFT_N, HOP);

    /* COLA sanity: periodic Hann at 50%% must sum to exactly 1. */
    double cmin = 1e9, cmax = -1e9;
    for (int n = 0; n < HOP; n++) {
        double s = win[n] + win[n + HOP];
        if (s < cmin) cmin = s;
        if (s > cmax) cmax = s;
    }
    printf("COLA min=%.12f max=%.12f\n", cmin, cmax);

    static const int KS[] = {4, 8, 12};
    static const int OS[] = {16, 64};
    for (size_t ki = 0; ki < sizeof KS / sizeof *KS; ki++)
        for (size_t oi = 0; oi < sizeof OS / sizeof *OS; oi++) {
            rng_state = 0x9e3779b9u;
            frame_test(KS[ki], OS[oi]);
        }
    stream_test(8, 64, 1);
    stream_test(8, 64, 16);
    stream_test(8, 64, 64);
    stream_test(12, 64, 16);
    return 0;
}
