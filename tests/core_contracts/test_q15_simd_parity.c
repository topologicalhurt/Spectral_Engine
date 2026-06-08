/* test_q15_simd_parity.c - packed 8xQ15 SIMD vs scalar Q15 parity (Q5c).
 *
 * Q5b locked the SCALAR Q15 path (integer-NCO phase) against float in
 * q15_production_parity. Q5c adds the packed 8xQ15 SIMD twin (osc_simd_q15_segment).
 * This test pins the SIMD kernel against its scalar Q15 ORACLE: it renders the same
 * representative segments through the production dispatch timbre_synth_segment()
 * twice with Q15 enabled, toggling only the float scalar/SIMD axis --
 *
 *   scalar Q15 : osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR)  -> synth_segment_q15
 *   SIMD   Q15 : osc_set_dispatch(OSC_DISPATCH_ALL_SIMD)    -> osc_simd_q15_segment
 *
 * -- and asserts the per-timbre RMS error stays at the Q15-eval LSB floor. The SIMD
 * eval matches the scalar spectral_osc_q15_* evaluators to <=1 LSB everywhere (the
 * triangle double-subtract and the pq pre-clamp keep the saturating corners bounded),
 * so any regression that breaks a lane op, the widen, or the amp ramp (>> 1 LSB) is
 * caught with room to spare. The five timbres are osc_simd_q15_available()'s set
 * (saw/square/triangle/parabola + sine). Sine's SIMD path is a serial LUT gather --
 * bit-identical to scalar spectral_osc_q15_sine -- so its only SIMD-vs-scalar delta
 * is the <=1 LSB vec-phase rounding, the same floor as the algebraic timbres.
 *
 * Run: cmake --build build --target q15_simd_parity_test \
 *      && ctest --test-dir build -R q15_simd_parity
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "oscillator.h"
#include "oscillator_dispatch.h"
#include "spectral_synth_internal.h"
#include "spectral_common.h"
#include "spectral_osc_q15.h"   /* scalar Q15 evaluators -- the SIMD oracle */
#include "simde/x86/sse2.h"

/* Test-only handle on the static-inline packed SIMD eval (oscillator_simd.c built
 * with SPECTRAL_EXPOSE_Q15_PACK8_FOR_TEST). */
simde__m128i spectral_q15_pack8_eval_for_test(simde__m128i pq, SpectralTimbre timbre,
                                              const q15_t* lut);

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

#define MAX_LEN 2048

/* Per-timbre RMS-error budget (dBFS, rel full scale 1.0). The SIMD vs scalar Q15
 * delta is sub-LSB (a 1-LSB Q15 step is ~-90 dBFS; only a handful of saturating-
 * corner samples differ), so the measured RMS lands far below these tripwires.
 * Budgets sit well above the floor but far below any real-bug threshold (a dropped
 * saturation or wrong shift shows up at >= -60 dBFS). Verdict is the printed table. */
typedef struct {
    SpectralTimbre timbre;
    const char*    name;
    double         budget_dbfs;
} Q15Timbre;

static Q15Timbre k_timbres[] = {
    { TIMBRE_SINE,     "sine",     -84.0 },
    { TIMBRE_SAW,      "saw",      -84.0 },
    { TIMBRE_SQUARE,   "square",   -84.0 },
    { TIMBRE_TRIANGLE, "triangle", -84.0 },
    { TIMBRE_PARABOLA, "parabola", -84.0 },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

typedef struct {
    size_t length;
    float  alpha;
    float  beta;
    float  c2;     /* rad/sample^2; no-linkage default == beta */
    float  c3;     /* rad/sample^3; 0 on the default quadratic model (SCOPE boundary) */
    float  phase;
    float  amp;
    float  d_amp;
} Q15Segment;

/* Low / high / chirped / odd-length segments (same spread as q15_production_parity), PLUS a
 * deliberately aggressive cubic segment (c3 != 0). The first four (c3 == 0) exercise the
 * vectorized uint32 8-wide phase (osc_simd_q15_segment's use_vec branch); the cubic one
 * exercises the scalar pack8 FALLBACK that production takes when c3 != 0. Both branches must
 * land within budget. The cubic c3 is large enough that a BROKEN scope gate (vec wrongly
 * applied to c3 != 0) would drift the phase tens of LSB and blow the -84 dBFS budget -- so
 * this row is the CI lock on the c3==0 scope boundary, not just fallback coverage. */
static const Q15Segment k_segments[] = {
    /* len   alpha      beta     c2        c3        phase    amp    d_amp    */
    { 1000,  0.0576f,   0.0f,    0.0f,     0.0f,     0.30f,   0.80f, -2.0e-4f },  /* ~440 Hz @48k */
    {  512,  0.5236f,   0.0f,    0.0f,     0.0f,    -1.10f,   0.50f,  0.0f    },  /* ~4 kHz       */
    {  768,  0.2000f,   1.0e-6f, 1.0e-6f,  0.0f,     0.00f,   0.90f,  1.0e-4f },  /* slight chirp */
    {  257,  0.1234f,   0.0f,    0.0f,     0.0f,     2.50f,   0.70f,  3.0e-4f },  /* odd length   */
    { 2000,  0.0262f,   5.0e-5f, 5.0e-5f,  3.0e-7f, -0.70f,   0.85f,  1.0e-4f },  /* cubic stress (fallback) */
};
#define N_SEGMENTS (sizeof(k_segments) / sizeof(k_segments[0]))

static void fill_lp(SegmentLoopParams* lp, const Q15Segment* s) {
    memset(lp, 0, sizeof(*lp));
    lp->start_idx = 0;
    lp->length    = s->length;
    lp->alpha     = s->alpha;
    lp->beta      = s->beta;
    lp->c2        = s->c2;
    lp->c3        = s->c3;
    lp->phase     = s->phase;
    lp->amp       = s->amp;
    lp->d_amp     = s->d_amp;
    lp->width     = 0.0f;
    lp->valid     = 1;
}

static void render(SegmentLoopParams* lp, SpectralTimbre timbre, OscDispatchWord dispatch, float* dst) {
    memset(dst, 0, sizeof(float) * MAX_LEN);
    osc_set_dispatch(dispatch);
    osc_set_q15_enable(OSC_Q15_BIT(timbre));
    timbre_synth_segment(dst, lp, timbre);
    osc_set_q15_enable(0);
}

static void test_q15_simd_parity(void) {
    for (size_t t = 0; t < N_TIMBRES; t++) {
        double sumsq_diff = 0.0, sumsq_sig = 0.0, max_abs = 0.0;
        size_t n = 0;

        for (size_t si = 0; si < N_SEGMENTS; si++) {
            SegmentLoopParams lp;
            fill_lp(&lp, &k_segments[si]);

            float buf_scalar[MAX_LEN];
            float buf_simd[MAX_LEN];
            render(&lp, k_timbres[t].timbre, OSC_DISPATCH_ALL_SCALAR, buf_scalar);
            render(&lp, k_timbres[t].timbre, OSC_DISPATCH_ALL_SIMD,   buf_simd);

            for (size_t j = 0; j < lp.length; j++) {
                double d = (double)buf_simd[j] - (double)buf_scalar[j];
                double ad = fabs(d);
                if (ad > max_abs) max_abs = ad;
                sumsq_diff += d * d;
                sumsq_sig  += (double)buf_scalar[j] * (double)buf_scalar[j];
                n++;
            }
        }

        double rms_diff = (n > 0) ? sqrt(sumsq_diff / (double)n) : 0.0;
        double rms_sig  = (n > 0) ? sqrt(sumsq_sig  / (double)n) : 0.0;
        double rms_dbfs = (rms_diff > 0.0) ? 20.0 * log10(rms_diff) : -300.0;
        double rel_db   = (rms_diff > 0.0 && rms_sig > 0.0)
                          ? 20.0 * log10(rms_diff / rms_sig) : -300.0;

        printf("  %-9s RMS err = %.3e (%.1f dBFS, %.1f dB vs sig)  peak = %.3e  [budget %.1f]\n",
               k_timbres[t].name, rms_diff, rms_dbfs, rel_db, max_abs,
               k_timbres[t].budget_dbfs);

        CHECK(rms_dbfs <= k_timbres[t].budget_dbfs,
              "%s SIMD-vs-scalar Q15 RMS error %.1f dBFS exceeds budget %.1f dBFS",
              k_timbres[t].name, rms_dbfs, k_timbres[t].budget_dbfs);
    }
}

/* EXHAUSTIVE bit-exact lock: in the default exact mode the packed 8xQ15 SIMD eval must
 * equal the scalar Q15 evaluator for EVERY phase index pq in [-32768, 32767], and the
 * saturating phase = -pi corner (pq == Q15_MIN) must equal the float reference --
 * triangle = -1.0 (Q15_MIN), parabola = 0. This pins the SIMD int16 overflow-corner fix;
 * under SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY the corner is allowed to drift 1 LSB. */
static void test_q15_simd_exhaustive_parity(void) {
    q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1u];
    spectral_osc_q15_init_sine_lut(lut);

    const SpectralTimbre timbres[] = { TIMBRE_SAW, TIMBRE_SQUARE, TIMBRE_TRIANGLE, TIMBRE_PARABOLA };
    const char* names[] = { "saw", "square", "triangle", "parabola" };

    for (size_t t = 0; t < 4; t++) {
        long mism = 0;
        for (int p = Q15_MIN; p <= Q15_MAX; p++) {
            q15_t pq = (q15_t)p;
            q15_t scalar;
            switch (timbres[t]) {
            case TIMBRE_SAW:      scalar = spectral_osc_q15_saw(pq);      break;
            case TIMBRE_SQUARE:   scalar = spectral_osc_q15_square(pq);   break;
            case TIMBRE_TRIANGLE: scalar = spectral_osc_q15_triangle(pq); break;
            default:              scalar = spectral_osc_q15_parabola(pq); break;
            }
            simde__m128i r = spectral_q15_pack8_eval_for_test(simde_mm_set1_epi16(pq), timbres[t], lut);
            q15_t simd = (q15_t)simde_mm_extract_epi16(r, 0);
            if (simd != scalar) {
                if (mism < 3) printf("  %s pq=%d : SIMD %d != scalar %d\n", names[t], pq, simd, scalar);
                mism++;
            }
        }
        CHECK(mism == 0, "%s: %ld of 65536 pq diverge SIMD-vs-scalar (exact mode must be bit-identical)",
              names[t], mism);
        if (mism == 0) printf("  %-9s 65536/65536 pq bit-identical (SIMD == scalar)\n", names[t]);
    }

    /* The corner the fix exists for: must equal the float reference, not the cheap clamp. */
    CHECK(spectral_osc_q15_triangle(Q15_MIN) == Q15_MIN,
          "triangle(-pi) must be -1.0 (Q15_MIN=%d), got %d", Q15_MIN, spectral_osc_q15_triangle(Q15_MIN));
    CHECK(spectral_osc_q15_parabola(Q15_MIN) == 0,
          "parabola(-pi) must be 0, got %d", spectral_osc_q15_parabola(Q15_MIN));
}

int main(void) {
    printf("== packed 8xQ15 SIMD vs scalar Q15 parity (per-path dBFS) ==\n");
    test_q15_simd_parity();
    printf("== exhaustive SIMD-vs-scalar Q15 eval parity (all 65536 pq) ==\n");
    test_q15_simd_exhaustive_parity();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
