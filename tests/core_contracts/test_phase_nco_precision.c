/* test_phase_nco_precision.c - integer-NCO cubic phase precision characterization (Q5a).
 *
 * QTYPE_DOMAIN_PLAN.md Q5 takes up the integer-NCO phase axis: the prerequisite for
 * double-lane Q15 packing (8xint16 in a 128-bit register) to ever beat float-SIMD.
 * Measurement (bench_q15_pack8) proved a Q15 SIMD kernel over a FLOAT phase
 * accumulator cannot win -- the
 * per-lane float->Q15 conversion cancels the 2x lane-density advantage before the eval
 * starts. The fix is to derive the Q15 phase index from an INTEGER accumulator with no
 * float on the hot path: spectral_phase_nco.h (a uint64 cubic forward-difference NCO,
 * full circle == 2^64, top 16 bits == the signed Q15 index the osc_q15 evaluators read).
 *
 * Before any kernel is wired (Q5b/Q5c), this harness answers the measure-first gate:
 * does the integer cubic phase track the float reference closely enough that switching
 * the Q15 path's phase source from float-cubic to integer-NCO costs nothing audible?
 *
 * Two measurements, both changing NO production output (header-only inlines + this TU):
 *
 *   Part 1 - PHASE-INDEX DRIFT (timbre-independent).  For worst-case alpha/c2/c3/length
 *   segments, step the integer NCO and the production float-cubic path in parallel and
 *   compare each one's resulting Q15 phase index to a DOUBLE-precision truth index.
 *   Reports max |index error| in LSBs for both.  Expectation: the integer NCO holds
 *   sub-LSB (its forward differences are EXACT integer adds; the only error is the
 *   one-time fixed-point init quantization of D1/D2/D3), while the float-cubic path
 *   drifts as total phase grows (float32 has ~8e-5 rad resolution near 700 rad).  This
 *   is the mechanistic "why integer phase is the right axis" number.
 *
 *   Part 2 - PHASE-SOURCE SWAP COST (per timbre, dBFS).  Drive the SAME Q15 waveform
 *   evaluator two ways -- once from the integer-NCO index, once from the float-cubic
 *   index (the CURRENTLY-SHIPPING opt-in Q15 path, oscillator.c synth_segment_q15) --
 *   and report the RMS difference in dBFS.  This is exactly Q5b's question: swapping the
 *   Q15 path's phase source must not degrade the already-characterized Q15-eval floor
 *   (test_q15_compute_precision: sine -85.1, saw -91.5, square -90.0, triangle -92.6,
 *   parabola -91.0 dBFS).
 *   The CHECK is a loose gross-bug tripwire; the printed table is the Q5b GO/NO-GO
 *   verdict (measure-first / decline-on-data discipline -- if integer phase
 *   drifts, we learn it HERE, before any kernel).
 *
 * Run: cmake --build build --target phase_nco_precision_test \
 *      && ctest --test-dir build -R phase_nco_precision
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "spectral_consts.h"
#include "spectral_osc_formulas.h"   /* float L0 osc + spectral_normalize_phase */
#include "spectral_lut.h"
#include "spectral_q15.h"
#include "spectral_osc_q15.h"        /* production Q15 evaluators + phase_from_rads */
#include "spectral_segment_math.h"   /* spectral_segment_phase_at_cubic_f32 (float path) */
#include "spectral_phase_nco.h"      /* the integer-NCO primitive under test */
#include "spectral_phase_nco8.h"     /* the vectorized 8-wide uint32 NCO (Bv) under test */

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

/* Correctness sanity floor: any sane phase source clears this by 20+ dB. Tripwire for a
 * gross NCO bug (wrong scale, dropped term), NOT the precision bar -- the table is. */
#define SANITY_FLOOR_DBFS (-55.0)

/* The integer NCO must hold the Q15 index to within a few LSB of the double truth; the
 * measured worst is 2 LSB (near-Nyquist) and is deterministic (fixed-point + double, no
 * float eval), so 3 is a tripwire with margin -- a genuinely broken NCO (a dropped cubic
 * term, wrong scale) drifts dozens-to-hundreds of LSB, not 3. The float-cubic path is NOT
 * gated: it is ALLOWED to drift (5 LSB near Nyquist) -- that drift is the axis motivation. */
#define NCO_INDEX_LSB_GATE 3

/* Vectorized uint32 8-wide NCO (Bv) gates -- conditional on c3, because production SCOPES
 * the vec NCO to c3==0 segments (osc_simd_q15_segment uses it only when lp->c3==0.0f, else
 * falls back to the shipped scalar pack8). uint32 lanes keep 16 fractional bits vs the
 * scalar uint64's 48: with c3==0 (linear/quad) the narrowing is sub-LSB and the vec index
 * holds the SAME few-LSB floor as the scalar NCO, so the SHIPPED path is gated TIGHT (==
 * NCO_INDEX_LSB_GATE). With c3!=0 the narrowed third difference drifts tens of LSB over a
 * long segment -- which is WHY production declines vec there; that path is gated only by a
 * loose gross-bug tripwire (the printed table documents the scope boundary, not a budget). */
#define VEC_NCO_C3ZERO_LSB_GATE  NCO_INDEX_LSB_GATE   /* shipped path: must match scalar floor */
#define VEC_NCO_C3NZ_LSB_GATE    128                  /* fallback path: tripwire, informational */

#define SAMPLE_RATE 48000.0
#define TEST_AMP 0.8f
#define OMEGA_HZ(f) ((float)(SPECTRAL_TWO_PI * (double)(f) / SAMPLE_RATE))

/* Full-scale (Q15_MAX) sine LUT, gain-matched to float -- built by the production helper
 * so test == ship (see spectral_osc_q15.h on why not the 32700 headroom table). */
static q15_t g_sine_lut[SPECTRAL_OSC_LUT_SIZE + 1u];

/* Worst-case cubic-phase segments. c2 (rad/sample^2) and c3 (rad/sample^3) are MQ-scale:
 * tiny per-step but their c3*t^3 term accumulates tens-to-hundreds of radians over the
 * segment -- which is exactly the regime where (a) float phase loses resolution and
 * (b) the NCO's sub-LSB third difference D3=6*c3 must not round to zero. */
typedef struct {
    const char* name;
    float phase0;   /* rad */
    float alpha;    /* rad/sample (== omega at t=0) */
    float c2;       /* rad/sample^2 */
    float c3;       /* rad/sample^3 */
    int   n;        /* segment length, samples */
} Seg;

static const Seg k_segs[] = {
    /* name           phase0   alpha             c2        c3        n     */
    { "linear-mid",   0.31f,   OMEGA_HZ(440.0),  0.0f,     0.0f,     2048 },
    { "near-nyquist", 0.31f,   OMEGA_HZ(22000.0),0.0f,     0.0f,     2048 },
    { "quad-chirp",   0.10f,   OMEGA_HZ(200.0),  1.0e-4f,  0.0f,     1024 },
    { "cubic-chirp", -0.70f,   OMEGA_HZ(200.0),  5.0e-5f,  3.0e-8f,  1024 },
    { "long-cubic",   0.00f,   OMEGA_HZ(150.0),  2.0e-5f,  1.0e-8f,  4096 },
    /* Mirrors the q15_simd_parity cubic-stress row: confirms that segment's vec drift is
     * large enough (tens of LSB) that a broken c3==0 scope gate would blow its dBFS budget. */
    { "parity-cubic",-0.70f,   0.0262f,          5.0e-5f,  3.0e-7f,  2000 },
};
#define N_SEGS (sizeof(k_segs) / sizeof(k_segs[0]))

/* Shortest signed circular distance between two Q15 phase indices, in LSBs (the index is
 * a full-turn uint16 alias, so 0 and 65535 are adjacent). */
static int circ_lsb(int16_t a, int16_t b) {
    int d = (int)(uint16_t)((uint16_t)a - (uint16_t)b);  /* 0..65535 */
    if (d >= 32768) d -= 65536;                          /* -32768..32767 */
    return d < 0 ? -d : d;
}

/* Double-precision truth index: round-to-nearest of the exact cubic phase, full circle
 * == 65536 (the same convention the NCO's top-16-bits extraction uses). */
static int16_t truth_index(const Seg* s, int t) {
    double td = (double)t;
    double theta = (double)s->phase0
                 + td * ((double)s->alpha + td * ((double)s->c2 + td * (double)s->c3));
    double turns = theta * 0.15915494309189535;          /* / 2pi, double */
    long long q  = llround(turns * 65536.0);
    return (int16_t)(uint16_t)(q & 0xFFFFLL);
}

/* Production float-cubic Q15 index: float Horner phase -> normalize -> phase_from_rads.
 * This is the index the SHIPPING opt-in Q15 path computes today (synth_segment_q15). */
static q15_t float_cubic_index(const Seg* s, int t) {
    float pf   = spectral_segment_phase_at_cubic_f32(s->phase0, s->alpha, s->c2, s->c3, (float)t);
    float rads = spectral_normalize_phase(pf);
    return spectral_osc_q15_phase_from_rads(rads);
}

typedef q15_t (*Q15WaveFn)(q15_t pq);
static q15_t wave_sine(q15_t pq)     { return spectral_osc_q15_sine(pq, g_sine_lut); }
static q15_t wave_saw(q15_t pq)      { return spectral_osc_q15_saw(pq); }
static q15_t wave_square(q15_t pq)   { return spectral_osc_q15_square(pq); }
static q15_t wave_triangle(q15_t pq) { return spectral_osc_q15_triangle(pq); }
static q15_t wave_parabola(q15_t pq) { return spectral_osc_q15_parabola(pq); }

typedef struct { const char* name; Q15WaveFn fq15; } Timbre;
static const Timbre k_timbres[] = {
    { "sine",     wave_sine     },
    { "saw",      wave_saw      },
    { "square",   wave_square   },
    { "triangle", wave_triangle },
    { "parabola", wave_parabola },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

/* Part 1 - phase-index drift (timbre-independent). Returns the worst NCO LSB error over
 * all segments (for the gate); prints a per-segment row for NCO and float-cubic. */
static int part1_index_drift(void) {
    int worst_nco = 0;
    printf("== Part 1: phase-index drift vs double truth (LSBs, lower is tighter) ==\n");
    printf("  %-13s  %6s  %10s  %12s\n", "segment", "len", "nco-max", "float-max");
    for (size_t si = 0; si < N_SEGS; si++) {
        const Seg* s = &k_segs[si];
        SpectralPhaseNco nco;
        spectral_phase_nco_init(&nco, s->phase0, s->alpha, s->c2, s->c3);
        int nco_max = 0, flt_max = 0;
        for (int t = 0; t < s->n; t++) {
            int16_t truth = truth_index(s, t);
            int16_t inco  = spectral_phase_nco_step(&nco);   /* reads then advances */
            int16_t iflt  = float_cubic_index(s, t);
            int en = circ_lsb(inco, truth);
            int ef = circ_lsb(iflt, truth);
            if (en > nco_max) nco_max = en;
            if (ef > flt_max) flt_max = ef;
        }
        if (nco_max > worst_nco) worst_nco = nco_max;
        printf("  %-13s  %6d  %10d  %12d\n", s->name, s->n, nco_max, flt_max);
        CHECK(nco_max <= NCO_INDEX_LSB_GATE,
              "%s integer-NCO index drift %d LSB exceeds gate %d",
              s->name, nco_max, NCO_INDEX_LSB_GATE);
    }
    printf("\n");
    return worst_nco;
}

#if defined(OSC_SIMD_GENERIC)
/* Part 3 - vectorized uint32 8-wide NCO drift (Bv). Same worst-case segments as Part 1,
 * but stepping the packed 8-lane uint32 NCO (spectral_phase_nco8.h) seeded EXACTLY from
 * the scalar uint64 NCO at t=0. uint32 lanes keep 16 fractional bits where the scalar keeps
 * 48, so this is the measure-first gate for replacing the kernel's serial scalar phase with
 * the vectorized one. Reports, per segment, the worst lane index error vs the double truth
 * AND vs the scalar uint64 NCO (the shipping oracle). Linear/quad expected at the scalar
 * floor; the cubic segments put the narrowed third difference on trial. */
static int part3_vec_index_drift(void) {
    int worst_vt = 0;
    double worst_saw_dbfs = -300.0;
    q15_t amp_q15 = FLOAT_TO_Q15(TEST_AMP);
    printf("== Part 3: vectorized uint32 8-wide NCO drift (Bv) vs scalar oracle ==\n");
    printf("  %-13s  %6s  %12s  %14s  %16s\n",
           "segment", "len", "vec-vs-truth", "vec-vs-scalar", "saw eval dBFS");
    for (size_t si = 0; si < N_SEGS; si++) {
        const Seg* s = &k_segs[si];
        SpectralPhaseNco snco;
        spectral_phase_nco_init(&snco, s->phase0, s->alpha, s->c2, s->c3);
        SpectralPhaseNco8 v = spectral_phase_nco8_seed(&snco);  /* seeds from a copy at t=0 */
        int nblocks = s->n / 8;
        int vt_max = 0, vs_max = 0;
        double sumsq = 0.0; size_t n = 0;
        for (int blk = 0; blk < nblocks; blk++) {
            int16_t lane[8];
            simde_mm_storeu_si128((simde__m128i*)lane, spectral_phase_nco8_step(&v));
            for (int k = 0; k < 8; k++) {
                int t = blk * 8 + k;
                int16_t sidx  = spectral_phase_nco_step(&snco);  /* lockstep scalar oracle */
                int16_t truth = truth_index(s, t);
                int evt = circ_lsb(lane[k], truth);
                int evs = circ_lsb(lane[k], sidx);
                if (evt > vt_max) vt_max = evt;
                if (evs > vs_max) vs_max = evs;
                /* Audio impact on the steepest algebraic timbre (saw = saturating negate,
                 * so its eval error tracks the raw phase-index error 1:1 -- the worst case). */
                float vf = Q15_TO_FLOAT(spectral_mul_q15(wave_saw(lane[k]), amp_q15));
                float sf = Q15_TO_FLOAT(spectral_mul_q15(wave_saw(sidx),    amp_q15));
                double e = (double)vf - (double)sf;
                sumsq += e * e; n++;
            }
        }
        double rms  = (n > 0) ? sqrt(sumsq / (double)n) : 0.0;
        double dbfs = (rms > 0.0) ? 20.0 * log10(rms) : -300.0;
        if (vt_max > worst_vt) worst_vt = vt_max;
        if (dbfs > worst_saw_dbfs) worst_saw_dbfs = dbfs;
        int is_shipped = (s->c3 == 0.0f);   /* production uses vec only on c3==0 segments */
        int gate = is_shipped ? VEC_NCO_C3ZERO_LSB_GATE : VEC_NCO_C3NZ_LSB_GATE;
        printf("  %-13s  %6d  %12d  %14d  %14.1f   %s\n", s->name, s->n, vt_max, vs_max, dbfs,
               is_shipped ? "[vec-shipped]" : "[scalar-fallback]");
        CHECK(vt_max <= gate,
              "%s vectorized-NCO index drift %d LSB exceeds %s gate %d",
              s->name, vt_max, is_shipped ? "shipped(c3==0)" : "fallback(c3!=0)", gate);
    }
    printf("  (saw eval dBFS = RMS of vec-vs-scalar phase through the steepest timbre, amp %.2f;\n"
           "   worst-segment saw impact = %.1f dBFS)\n\n", TEST_AMP, worst_saw_dbfs);
    return worst_vt;
}
#endif /* OSC_SIMD_GENERIC */

/* Part 2 - phase-source swap cost (per timbre, dBFS). ref = float-cubic index -> Q15 eval
 * (the shipping Q15 path); test = integer-NCO index -> Q15 eval. Same evaluator both
 * sides, so the dBFS is purely the cost of swapping the phase source. */
static double part2_swap_cost(const Timbre* tb) {
    q15_t amp_q15 = FLOAT_TO_Q15(TEST_AMP);
    double agg_sq = 0.0, agg_peak = 0.0;
    size_t n = 0;
    for (size_t si = 0; si < N_SEGS; si++) {
        const Seg* s = &k_segs[si];
        SpectralPhaseNco nco;
        spectral_phase_nco_init(&nco, s->phase0, s->alpha, s->c2, s->c3);
        for (int t = 0; t < s->n; t++) {
            q15_t inco = spectral_phase_nco_step(&nco);
            q15_t iflt = float_cubic_index(s, t);

            float test = Q15_TO_FLOAT(spectral_mul_q15(tb->fq15(inco), amp_q15));
            float ref  = Q15_TO_FLOAT(spectral_mul_q15(tb->fq15(iflt), amp_q15));

            double e = (double)test - (double)ref;
            agg_sq += e * e;
            double ae = fabs(e);
            if (ae > agg_peak) agg_peak = ae;
            n++;
        }
    }
    double rms  = sqrt(agg_sq / (double)n);
    double dbfs = (rms > 0.0) ? 20.0 * log10(rms) : -300.0;
    printf("  %-9s  RMS diff = %.3e (%.1f dBFS)   peak diff = %.3e\n",
           tb->name, rms, dbfs, agg_peak);
    return dbfs;
}

int main(void) {
    spectral_osc_q15_init_sine_lut(g_sine_lut);

    int worst_nco = part1_index_drift();

#if defined(OSC_SIMD_GENERIC)
    int worst_vec = part3_vec_index_drift();
#endif

    printf("== Part 2: phase-source swap cost (int-NCO vs float-cubic, same Q15 eval) ==\n");
    printf("   ref = shipping float-cubic Q15 path; test = integer-NCO Q15 path; amp = %.2f\n\n",
           TEST_AMP);
    for (size_t i = 0; i < N_TIMBRES; i++) {
        double dbfs = part2_swap_cost(&k_timbres[i]);
        CHECK(dbfs <= SANITY_FLOOR_DBFS, "%s swap cost %.1f dBFS worse than sanity floor %.1f",
              k_timbres[i].name, dbfs, SANITY_FLOOR_DBFS);
    }

    printf("\n  worst integer-NCO index drift across all segments: %d LSB (gate %d)\n",
           worst_nco, NCO_INDEX_LSB_GATE);
#if defined(OSC_SIMD_GENERIC)
    printf("  worst vectorized-NCO (Bv) index drift across all segments: %d LSB "
           "(shipped c3==0 gate %d; fallback c3!=0 tripwire %d)\n",
           worst_vec, VEC_NCO_C3ZERO_LSB_GATE, VEC_NCO_C3NZ_LSB_GATE);
#endif
    printf("  (the LSB and dBFS tables are the Q5b GO/NO-GO data for QTYPE plan Q5;\n"
           "   the sanity floor %.1f dBFS is a gross-bug tripwire, not the verdict.)\n",
           SANITY_FLOOR_DBFS);
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
