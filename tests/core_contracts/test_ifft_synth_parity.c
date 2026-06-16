/* test_ifft_synth_parity.c - F2 contract test for the IFFT synthesis path.
 *
 * Three rungs (IFFT_SYNTHESIS_PLAN parity ladder, now over the REAL float
 * path with the build's live iFFT port — vDSP on Apple, portable radix-2
 * elsewhere):
 *   1. Backend contract: inverse of a random Hermitian half-spectrum matches
 *      a double-precision O(N^2) reference iDFT (catches packing/scale bugs
 *      in either port immediately).
 *   2. Stream parity: 64 dense stationary partials rendered via
 *      spectral_ifft_synth_render() vs the exact double oscillator sum —
 *      must hit the F1-measured floors (-55 dBFS frame class at K=8) with
 *      float headroom: max <= -60 dBFS... see asserts (F1 floors -68 max /
 *      -83 RMS in double; float port budgeted ~8 dB).
 *   3. Determinism: two renders are byte-identical.
 *   4. Dynamic path (the F2b router foundation): a constant per-frame set
 *      reproduces the static render BIT-EXACT; time-localized activity keeps
 *      energy to the active frames; out-of-domain partials a frame returns are
 *      skipped, not rendered.
 * Also prints a MEASURED throughput comparison vs a plain float oscillator
 * loop (informational; perf gating lives in the S4 bench world, not ctest).
 *
 * Run: cmake --build build --target ifft_synth_parity_test
 *      && ctest --test-dir build -R ifft_synth_parity
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "spectral_consts.h"
#include "spectral_synth_ifft.h"

#define N_FFT 512u
#define TOTAL 8192u
#define N_PARTIALS 64u

#include "../support/check.h"

#include "../support/xorshift_rng.h"

static unsigned rng_state = 0x5eed5eedu;

static double dbfs(double x) { return 20.0 * log10(x > 1e-300 ? x : 1e-300); }

static uint64_t now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* Rung 1: the port's inverse vs a double reference iDFT. */
static void test_backend_contract(void) {
    printf("test_backend_contract:\n");
    SpectralIfftBackend* b = spectral_ifft_backend_create(N_FFT);
    CHECK(b != NULL, "backend create");
    if (!b) return;

    float re[N_FFT / 2], im[N_FFT / 2], out[N_FFT];
    double fre[N_FFT], fim[N_FFT];
    memset(fre, 0, sizeof fre); memset(fim, 0, sizeof fim);

    re[0] = (float)(xorshift32_unit(&rng_state) - 0.5);              /* DC */
    im[0] = (float)(xorshift32_unit(&rng_state) - 0.5);              /* Nyquist (packed) */
    fre[0] = re[0];
    fre[N_FFT / 2] = im[0];
    for (size_t k = 1; k < N_FFT / 2; k++) {
        re[k] = (float)(xorshift32_unit(&rng_state) - 0.5);
        im[k] = (float)(xorshift32_unit(&rng_state) - 0.5);
        fre[k] = re[k];           fim[k] = im[k];
        fre[N_FFT - k] = re[k];   fim[N_FFT - k] = -im[k];   /* Hermitian */
    }

    spectral_ifft_backend_inverse(b, re, im, out);

    double worst = 0.0;
    for (size_t n = 0; n < N_FFT; n++) {
        double acc = 0.0;
        for (size_t k = 0; k < N_FFT; k++) {
            double a = 2.0 * SPECTRAL_PI_D * (double)k * (double)n / (double)N_FFT;
            acc += fre[k] * cos(a) - fim[k] * sin(a);
        }
        acc /= (double)N_FFT;
        double err = fabs((double)out[n] - acc);
        if (err > worst) worst = err;
    }
    printf("  backend max err vs reference iDFT: %.3e\n", worst);
    CHECK(worst < 1e-4, "port inverse must match the textbook iDFT (got %.3e)", worst);
    spectral_ifft_backend_destroy(b);
}

/* Rungs 2+3: stream parity + determinism + measured throughput. */
static void test_stream_parity(void) {
    printf("test_stream_parity:\n");
    SpectralIfftSynth* s = spectral_ifft_synth_create(N_FFT);
    CHECK(s != NULL, "synth create");
    if (!s) return;

    static SpectralIfftPartial parts[N_PARTIALS];
    rng_state = 0xc0ffee11u;
    for (size_t p = 0; p < N_PARTIALS; p++) {
        parts[p].bin = (float)(10.0 + 230.0 * xorshift32_unit(&rng_state));
        parts[p].amp = (float)((0.05 + 0.95 * xorshift32_unit(&rng_state)) / N_PARTIALS);
        parts[p].phase0 = (float)(2.0 * SPECTRAL_PI_D * xorshift32_unit(&rng_state) - SPECTRAL_PI_D);
    }

    static float out[TOTAL], out2[TOTAL];

    /* Boundary contract: a partial outside bin in (1, n_fft/2-1), or with a
     * non-finite field, must be REJECTED — the bin floor-cast would be
     * int-conversion UB on garbage, and an out-of-domain partial would
     * silently render nothing of itself. */
    {
        SpectralIfftPartial bad = parts[0];
        bad.bin = (float)N_FFT / 2.0f;
        CHECK(spectral_ifft_synth_render(s, &bad, 1, out, TOTAL) != 0,
              "render must reject bin at/above n_fft/2-1");
        bad.bin = 0.5f;
        CHECK(spectral_ifft_synth_render(s, &bad, 1, out, TOTAL) != 0,
              "render must reject bin at/below 1");
        bad = parts[0];
        bad.bin = NAN;
        CHECK(spectral_ifft_synth_render(s, &bad, 1, out, TOTAL) != 0,
              "render must reject NaN bin");
        bad = parts[0];
        bad.amp = INFINITY;
        CHECK(spectral_ifft_synth_render(s, &bad, 1, out, TOTAL) != 0,
              "render must reject non-finite amp");
        bad = parts[0];
        bad.phase0 = 1.0e20f;   /* finite but huge -> the (long long) reduction would overflow */
        CHECK(spectral_ifft_synth_render(s, &bad, 1, out, TOTAL) != 0,
              "render must reject out-of-range phase0 (cast-UB guard)");
    }

    uint64_t t0 = now_ns();
    CHECK(spectral_ifft_synth_render(s, parts, N_PARTIALS, out, TOTAL) == 0, "render rc");
    uint64_t t_ifft = now_ns() - t0;

    /* Exact reference: double oscillator sum. */
    double worst = 0.0, sumsq = 0.0;
    for (size_t t = 0; t < TOTAL; t++) {
        double want = 0.0;
        for (size_t p = 0; p < N_PARTIALS; p++) {
            double omega = 2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT;
            want += (double)parts[p].amp * cos(omega * (double)t + (double)parts[p].phase0);
        }
        double err = fabs((double)out[t] - want);
        if (err > worst) worst = err;
        sumsq += err * err;
    }
    double rms = sqrt(sumsq / TOTAL);
    printf("  stream parity: max=%.2f dBFS rms=%.2f dBFS (%u partials)\n",
           dbfs(worst), dbfs(rms), (unsigned)N_PARTIALS);
    CHECK(dbfs(worst) < -60.0, "stream max err must beat -60 dBFS (got %.2f)", dbfs(worst));
    CHECK(dbfs(rms) < -75.0, "stream rms err must beat -75 dBFS (got %.2f)", dbfs(rms));

    /* Determinism. */
    CHECK(spectral_ifft_synth_render(s, parts, N_PARTIALS, out2, TOTAL) == 0, "render rc 2");
    CHECK(memcmp(out, out2, sizeof out) == 0, "render must be deterministic");

    /* MEASURED throughput vs a plain float oscillator loop (info only). */
    {
        static float ref[TOTAL];
        uint64_t t1 = now_ns();
        memset(ref, 0, sizeof ref);
        for (size_t p = 0; p < N_PARTIALS; p++) {
            float omega = (float)(2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT);
            float ph = parts[p].phase0;
            for (size_t t = 0; t < TOTAL; t++) {
                ref[t] += parts[p].amp * cosf(omega * (float)t + ph);
            }
        }
        uint64_t t_osc = now_ns() - t1;

        /* Volatile sink AFTER timing: every element feeds the reduction, so
         * the rendered buffer stays live without perturbing the timed loop. */
        {
            static volatile float sink;
            float acc = 0.0f;
            for (size_t t = 0; t < TOTAL; t++) acc += ref[t];
            sink = acc;
            (void)sink;
        }
        printf("  MEASURED: ifft %.2f ms vs naive-osc %.2f ms -> %.1fx "
               "(info; not a gate)\n",
               t_ifft / 1e6, t_osc / 1e6,
               t_ifft > 0 ? (double)t_osc / (double)t_ifft : 0.0);
    }

    spectral_ifft_synth_destroy(s);
}

/* --- dynamic (per-frame-activity) render path, the F2b router foundation --- */

typedef struct { const SpectralIfftPartial* parts; size_t n; } ConstSetCtx;
static size_t const_set_fn(void* vctx, double center, SpectralIfftPartial* out, size_t cap) {
    (void)center;
    const ConstSetCtx* c = (const ConstSetCtx*)vctx;
    size_t n = c->n < cap ? c->n : cap;
    for (size_t i = 0; i < n; i++) out[i] = c->parts[i];
    return n;
}

typedef struct { SpectralIfftPartial part; double t_lo, t_hi; } GatedCtx;
static size_t gated_fn(void* vctx, double center, SpectralIfftPartial* out, size_t cap) {
    const GatedCtx* c = (const GatedCtx*)vctx;
    if (cap == 0 || center < c->t_lo || center > c->t_hi) return 0;
    out[0] = c->part;
    return 1;
}

typedef struct { SpectralIfftPartial good; } SkipCtx;
static size_t skip_fn(void* vctx, double center, SpectralIfftPartial* out, size_t cap) {
    (void)center;
    const SkipCtx* c = (const SkipCtx*)vctx;
    if (cap < 2) return 0;
    out[0] = c->good;
    out[1] = c->good;
    out[1].bin = 1.0e9f;   /* out of (1, n_fft/2-1) domain -> must be skipped */
    return 2;
}

/* Rung 4: the dynamic path equals the static path on a constant set (bit-exact),
 * localizes energy to active frames, and skips out-of-domain partials. */
static void test_dynamic_render(void) {
    printf("test_dynamic_render:\n");
    SpectralIfftSynth* s = spectral_ifft_synth_create(N_FFT);
    CHECK(s != NULL, "synth create (dynamic)");
    if (!s) return;

    static SpectralIfftPartial parts[N_PARTIALS];
    rng_state = 0xc0ffee11u;   /* same seed/params as the static stream test */
    for (size_t p = 0; p < N_PARTIALS; p++) {
        parts[p].bin = (float)(10.0 + 230.0 * xorshift32_unit(&rng_state));
        parts[p].amp = (float)((0.05 + 0.95 * xorshift32_unit(&rng_state)) / N_PARTIALS);
        parts[p].phase0 = (float)(2.0 * SPECTRAL_PI_D * xorshift32_unit(&rng_state) - SPECTRAL_PI_D);
    }

    static float ds[TOTAL], st[TOTAL];

    /* 1. Constant per-frame set == static render, BIT-EXACT: the dynamic path
     *    shares the framing/OLA/phase math; only the partial source differs. */
    ConstSetCtx cc = { parts, N_PARTIALS };
    CHECK(spectral_ifft_synth_render_dynamic(s, const_set_fn, &cc, ds, TOTAL) == 0, "dynamic rc");
    CHECK(spectral_ifft_synth_render(s, parts, N_PARTIALS, st, TOTAL) == 0, "static rc");
    CHECK(memcmp(ds, st, sizeof ds) == 0,
          "dynamic w/ constant set must equal static render bit-for-bit");

    /* 2. Time-localized: one partial live only in the middle third. Frames whose
     *    center lands in [t_lo, t_hi] place it; the rest place nothing, so energy
     *    concentrates in the middle and the far edges (beyond a frame of OLA
     *    spill) stay exactly silent. */
    GatedCtx gc;
    gc.part = parts[0];
    gc.part.amp = 0.5f;
    gc.t_lo = (double)TOTAL / 3.0;
    gc.t_hi = 2.0 * (double)TOTAL / 3.0;
    CHECK(spectral_ifft_synth_render_dynamic(s, gated_fn, &gc, ds, TOTAL) == 0, "gated rc");

    double mid_sumsq = 0.0, edge_max = 0.0;
    const size_t guard = N_FFT;   /* one frame of OLA spill past the gate */
    for (size_t t = 0; t < TOTAL; t++) {
        if (t >= (size_t)gc.t_lo + guard && t < (size_t)gc.t_hi - guard) {
            mid_sumsq += (double)ds[t] * (double)ds[t];
        } else if (t + guard < (size_t)gc.t_lo || t > (size_t)gc.t_hi + guard) {
            double a = fabs((double)ds[t]);
            if (a > edge_max) edge_max = a;
        }
    }
    double mid_rms = sqrt(mid_sumsq / TOTAL);
    printf("  gated: mid rms=%.2f dBFS far-edge max=%.3e\n", dbfs(mid_rms), edge_max);
    CHECK(mid_rms > 1.0e-3, "active middle must carry energy (rms %.3e)", mid_rms);
    CHECK(edge_max < 1.0e-4, "inactive far edges must stay silent (max %.3e)", edge_max);

    /* 3. A frame returning an out-of-domain partial: it must be SKIPPED, never
     *    rendered or fatal. [good, bad] every frame == static render of [good]. */
    SkipCtx sc;
    sc.good = parts[3];
    CHECK(spectral_ifft_synth_render_dynamic(s, skip_fn, &sc, ds, TOTAL) == 0, "skip rc");
    CHECK(spectral_ifft_synth_render(s, &sc.good, 1, st, TOTAL) == 0, "static single rc");
    CHECK(memcmp(ds, st, sizeof ds) == 0,
          "dynamic must skip out-of-domain partials (== static of the valid one)");

    spectral_ifft_synth_destroy(s);
}

/* Rung 5: exact-integer bins (frac=0) exercise place_partial_cri's per-tap boundary
 * fallback — the hoisted interior fast path gates on frac>0 because an integer bin
 * would otherwise index one past the motif table on the last tap. Verify they render
 * correctly (a regression that dropped the frac>0 guard would read past the table). */
static void test_integer_bins(void) {
    printf("test_integer_bins:\n");
    SpectralIfftSynth* s = spectral_ifft_synth_create(N_FFT);
    CHECK(s != NULL, "synth create (int bins)");
    if (!s) return;

    static SpectralIfftPartial parts[8];
    for (size_t p = 0; p < 8; p++) {
        parts[p].bin = (float)(16 + 24 * (int)p);   /* exact integers 16..184, all interior */
        parts[p].amp = 0.1f;
        parts[p].phase0 = (float)(0.3 * (double)p);
    }
    static float out[TOTAL];
    CHECK(spectral_ifft_synth_render(s, parts, 8, out, TOTAL) == 0, "int-bin render rc");

    double worst = 0.0;
    for (size_t t = N_FFT; t < TOTAL - N_FFT; t++) {
        double want = 0.0;
        for (size_t p = 0; p < 8; p++) {
            double omega = 2.0 * SPECTRAL_PI_D * (double)parts[p].bin / (double)N_FFT;
            want += (double)parts[p].amp * cos(omega * (double)t + (double)parts[p].phase0);
        }
        double err = fabs((double)out[t] - want);
        if (err > worst) worst = err;
    }
    printf("  integer-bin max err: %.2f dBFS\n", dbfs(worst));
    CHECK(dbfs(worst) < -55.0, "integer-bin render must beat -55 dBFS (got %.2f)", dbfs(worst));
    spectral_ifft_synth_destroy(s);
}

/* Rung 6: static-allocation init (the embedded F4 path) — a caller-pool synth renders
 * BIT-IDENTICALLY to a malloc'd create() synth; undersized pool + null backend are
 * rejected; destroy() on an init pool is a no-op (the caller owns + frees it). */
static void test_init_inplace(void) {
    printf("test_init_inplace:\n");
    size_t bytes = spectral_ifft_synth_pool_bytes(N_FFT);
    CHECK(bytes > 0, "pool_bytes > 0");
    CHECK(spectral_ifft_synth_pool_bytes(63) == 0, "non-pow2 -> 0 bytes");

    void* pool = malloc(bytes);
    SpectralIfftBackend* b = spectral_ifft_backend_create(N_FFT);
    CHECK(pool && b, "pool + backend alloc");
    if (!pool || !b) return;

    CHECK(spectral_ifft_synth_init(pool, bytes - 1, N_FFT, b) == NULL, "undersized pool rejected");
    CHECK(spectral_ifft_synth_init(pool, bytes, N_FFT, NULL) == NULL, "null backend rejected");

    /* A backend built for a different n_fft must be rejected (else the FFT reads/writes
     * past the synth's re/im/frame scratch). */
    SpectralIfftBackend* bmis = spectral_ifft_backend_create(N_FFT * 2u);
    CHECK(bmis != NULL, "mismatch backend alloc");
    CHECK(spectral_ifft_synth_init(pool, bytes, N_FFT, bmis) == NULL, "backend n_fft mismatch rejected");
    spectral_ifft_backend_destroy(bmis);

    SpectralIfftSynth* si = spectral_ifft_synth_init(pool, bytes, N_FFT, b);
    SpectralIfftSynth* sc = spectral_ifft_synth_create(N_FFT);
    CHECK(si && sc, "init + create");
    if (si && sc) {
        static SpectralIfftPartial parts[N_PARTIALS];
        rng_state = 0xc0ffee11u;
        for (size_t p = 0; p < N_PARTIALS; p++) {
            parts[p].bin = (float)(10.0 + 230.0 * xorshift32_unit(&rng_state));
            parts[p].amp = (float)((0.05 + 0.95 * xorshift32_unit(&rng_state)) / N_PARTIALS);
            parts[p].phase0 = (float)(2.0 * SPECTRAL_PI_D * xorshift32_unit(&rng_state) - SPECTRAL_PI_D);
        }
        static float oi[TOTAL], oc[TOTAL];
        CHECK(spectral_ifft_synth_render(si, parts, N_PARTIALS, oi, TOTAL) == 0, "init render rc");
        CHECK(spectral_ifft_synth_render(sc, parts, N_PARTIALS, oc, TOTAL) == 0, "create render rc");
        CHECK(memcmp(oi, oc, sizeof oi) == 0, "init pool render == create render (bit-identical)");
    }
    spectral_ifft_synth_destroy(sc);
    spectral_ifft_synth_destroy(si);   /* no-op: init pool is caller-owned */
    spectral_ifft_backend_destroy(b);
    free(pool);
}

int main(void) {
    test_backend_contract();
    test_stream_parity();
    test_dynamic_render();
    test_integer_bins();
    test_init_inplace();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
