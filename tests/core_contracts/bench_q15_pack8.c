/* bench_q15_pack8.c - Q5c measure-first probe: can packed 8xQ15 beat 4-wide float-SIMD?
 *
 * NOT a ctest (timing is machine-dependent): a standalone EXCLUDE_FROM_ALL probe
 * that settles the Q5c GO/NO-GO on data (QTYPE_DOMAIN_PLAN.md), exactly as
 * bench_q15_throughput settled Q3b. A Q15 SIMD kernel over FLOAT phase was
 * declined on data: the per-lane float->Q15 conversion cancelled the
 * lane-density win. Q5a/Q5b removed that precondition: phase is now an integer NCO. This probe drives the kernel as hard
 * as it can be driven on desktop and isolates where the time goes, so the GO/NO-GO
 * is made on a *tuned* kernel, not a strawman.
 *
 * Variants (all render the SAME linear sustain segment, accumulate into a buffer;
 * an opaque noinline sink forces every store live so DCE can't inflate the win):
 *
 *   A  prod-fsimd   production timbre_synth_segment(), ALL_SIMD (the real shipping
 *                   path; carries its full fade/dispatch/call cost).
 *   A8 lean-favx8   hand-rolled lean 8-wide float kernel (phase+eval+amp+acc) --
 *                   the FAIR yardstick: same lane count + leanness as pack8, so
 *                   B/A8 isolates Q15 density from production overhead.
 *   B  pack8-f      packed 8xQ15: scalar int-NCO x8 -> pack -> 8-wide Q15 eval
 *                   -> Q15->float (1-op sign-extend) -> vectorized float amp ->
 *                   float accumulate. The realizable desktop kernel, tuned.
 *   Bv pack8-f-vph  same, but phase from a VECTORIZED 8-wide cubic NCO (uint32
 *                   lanes, stride-8 forward differences) -- squeezes serial phase.
 *   B0 pack8-f-noph B with phase removed (constant index) -- the convert+amp+
 *                   accumulate FLOOR (what no phase trick can remove).
 *
 * Decision reads:
 *   B/A8, Bv/A8 : does a tuned desktop pack8 beat a fair lean 8-wide float kernel?
 *   Bv/B        : does vectorizing the phase help (is phase the bottleneck)?
 *   B0          : the irreducible eval+convert+accumulate floor.
 *
 * Run: cmake --build build --target bench_q15_pack8 && build/bench_q15_pack8
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "oscillator.h"
#include "oscillator_dispatch.h"        /* pulls in simde/x86/sse2.h */
#include "spectral_synth_internal.h"
#include "spectral_q.h"
#include "spectral_lut.h"
#include "spectral_osc_q15.h"
#include "spectral_phase_nco.h"

#if defined(OSC_SIMD_GENERIC)
#include "simde/x86/ssse3.h"            /* abs_epi16 / mulhrs_epi16 */
#include "simde/x86/sse4.1.h"           /* cvtepi16_epi32 / packus_epi32 / blendv */
#include "simde/x86/avx.h"              /* __m256 lean 8-wide float baseline */
#endif

#define SEG_LEN  2048   /* sustain-dominated single segment (multiple of 8) */
#define ITERS    20000  /* ~40M samples/measurement -> stable timing */

typedef struct { SpectralTimbre timbre; const char* name; } BenchTimbre;
static const BenchTimbre k_timbres[] = {
    { TIMBRE_SINE,     "sine"     },
    { TIMBRE_SAW,      "saw"      },
    { TIMBRE_SQUARE,   "square"   },
    { TIMBRE_TRIANGLE, "triangle" },
    { TIMBRE_PARABOLA, "parabola" },
};
#define N_TIMBRES (sizeof(k_timbres) / sizeof(k_timbres[0]))

/* Segment shape shared by all variants. */
#define SEG_ALPHA   0.0576f    /* ~440 Hz @ 48k (rad/sample) */
#define SEG_PHASE0  0.30f
#define SEG_AMP0    0.80f
#define SEG_DAMP   (-2.0e-4f)

static void fill_lp(SegmentLoopParams* lp) {
    memset(lp, 0, sizeof(*lp));
    lp->start_idx = 0;
    lp->length    = SEG_LEN;
    lp->alpha     = SEG_ALPHA;
    lp->phase     = SEG_PHASE0;
    lp->amp       = SEG_AMP0;
    lp->d_amp     = SEG_DAMP;
    lp->valid     = 1;
}

/* timing helper: ns/sample given elapsed ns. */
static inline double ns_per_sample(double ns) {
    return ns / ((double)ITERS * (double)SEG_LEN);
}

/* Opaque sink: takes the buffer POINTER so the compiler cannot dead-store-eliminate
 * the writes (it can't prove this noinline fn doesn't read every element). Without
 * this, the fully-inlined pack8/float variants would have their stores elided
 * (only buf[mid] is read), inflating the win vs the non-inlined production call. */
__attribute__((noinline)) static double consume_buf(const float* b) {
    return (double)b[SEG_LEN / 2] + (double)b[0] + (double)b[SEG_LEN - 1];
}

/* ---- variant A: production 4-wide float-SIMD (shipping reference) ---- */
static double bench_prod_fsimd4(SpectralTimbre timbre, double* sink) {
    SegmentLoopParams lp; fill_lp(&lp);
    float buf[SEG_LEN];
    osc_set_dispatch(OSC_DISPATCH_ALL_SIMD);
    osc_set_q15_enable(0);
    for (int it = 0; it < 64; it++) { memset(buf, 0, sizeof(buf)); timbre_synth_segment(buf, &lp, timbre); }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    double acc = 0.0;
    for (int it = 0; it < ITERS; it++) {
        memset(buf, 0, sizeof(buf));
        timbre_synth_segment(buf, &lp, timbre);
        acc += consume_buf(buf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    *sink += acc;
    return ns_per_sample((double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec));
}

#if defined(OSC_SIMD_GENERIC)

/* ===== lean float-AVX-8 baseline (A''): the FAIR same-width float kernel ====
 * 8-wide float (matches pack8's 8 samples/iter and AVX2 register width). This is
 * what a lean float kernel costs at the same lane count -- the honest yardstick
 * for "does Q15 density beat float", isolating density from production overhead. */
static double bench_lean_favx8(SpectralTimbre timbre, const q15_t* lut, double* sink) {
    float buf[SEG_LEN];
    const float dn = SEG_ALPHA / 3.14159265358979f;
    const simde__m256 absmask = simde_mm256_castsi256_ps(simde_mm256_set1_epi32(0x7FFFFFFF));
    const simde__m256 one  = simde_mm256_set1_ps(1.0f);
    const simde__m256 two  = simde_mm256_set1_ps(2.0f);
    const simde__m256 zero = simde_mm256_setzero_ps();
    const simde__m256 negone = simde_mm256_set1_ps(-1.0f);
    const simde__m256 nstep = simde_mm256_set1_ps(8.0f * dn);
    const simde__m256 ampstep = simde_mm256_set1_ps(8.0f * SEG_DAMP);

    for (int warm = 0; warm < 2; warm++) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    double chk = 0.0;
    int iters = warm ? ITERS : 64;
    for (int it = 0; it < iters; it++) {
        memset(buf, 0, sizeof(buf));
        simde__m256 n = simde_mm256_set_ps(SEG_PHASE0+7*dn, SEG_PHASE0+6*dn, SEG_PHASE0+5*dn,
                                           SEG_PHASE0+4*dn, SEG_PHASE0+3*dn, SEG_PHASE0+2*dn,
                                           SEG_PHASE0+dn, SEG_PHASE0);
        simde__m256 amp = simde_mm256_set_ps(SEG_AMP0+7*SEG_DAMP, SEG_AMP0+6*SEG_DAMP,
                                             SEG_AMP0+5*SEG_DAMP, SEG_AMP0+4*SEG_DAMP,
                                             SEG_AMP0+3*SEG_DAMP, SEG_AMP0+2*SEG_DAMP,
                                             SEG_AMP0+SEG_DAMP, SEG_AMP0);
        for (int j = 0; j < SEG_LEN; j += 8) {
            simde__m256 ge = simde_mm256_cmp_ps(n, one, SIMDE_CMP_GE_OQ);
            n = simde_mm256_sub_ps(n, simde_mm256_and_ps(ge, two));
            simde__m256 wave;
            switch (timbre) {
                case TIMBRE_SAW:   wave = simde_mm256_sub_ps(zero, n); break;
                case TIMBRE_SQUARE: {
                    simde__m256 gt = simde_mm256_cmp_ps(n, zero, SIMDE_CMP_GT_OQ);
                    wave = simde_mm256_blendv_ps(negone, one, gt);
                } break;
                case TIMBRE_TRIANGLE:
                    wave = simde_mm256_sub_ps(one, simde_mm256_mul_ps(two, simde_mm256_and_ps(n, absmask)));
                    break;
                case TIMBRE_PARABOLA:
                    wave = simde_mm256_sub_ps(one, simde_mm256_mul_ps(n, n));
                    break;
                case TIMBRE_SINE:
                default: {
                    SIMDE_ALIGN_LIKE_32(simde__m256) float nn[8], wv[8];
                    simde_mm256_store_ps(nn, n);
                    for (int k = 0; k < 8; k++) {
                        q15_t pq = spectral_osc_q15_phase_from_rads(nn[k] * 3.14159265358979f);
                        wv[k] = Q15_TO_FLOAT(spectral_lut_sin((uq16_t)(uint16_t)pq, lut));
                    }
                    wave = simde_mm256_load_ps(wv);
                } break;
            }
            simde__m256 b = simde_mm256_loadu_ps(&buf[j]);
            b = simde_mm256_add_ps(b, simde_mm256_mul_ps(amp, wave));
            simde_mm256_storeu_ps(&buf[j], b);
            n   = simde_mm256_add_ps(n, nstep);
            amp = simde_mm256_add_ps(amp, ampstep);
        }
        chk += consume_buf(buf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (warm) { *sink += chk;
        return ns_per_sample((double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec)); }
    }
    return 0.0;
}

/* ===== packed 8-wide Q15 waveform eval (SIMD twin of osc_q15_eval) ========= */
static inline simde__m128i pack8_eval(simde__m128i pq, SpectralTimbre timbre, const q15_t* lut) {
    const simde__m128i zero = simde_mm_setzero_si128();
    switch (timbre) {
        case TIMBRE_SAW:   return simde_mm_subs_epi16(zero, pq);
        case TIMBRE_SQUARE: {
            simde__m128i gt = simde_mm_cmpgt_epi16(pq, zero);
            return simde_mm_or_si128(simde_mm_and_si128(gt, simde_mm_set1_epi16(Q15_MAX)),
                                     simde_mm_andnot_si128(gt, simde_mm_set1_epi16(Q15_MIN)));
        }
        case TIMBRE_TRIANGLE: {
            simde__m128i a2 = simde_mm_adds_epi16(simde_mm_abs_epi16(pq), simde_mm_abs_epi16(pq));
            return simde_mm_subs_epi16(simde_mm_set1_epi16(Q15_MAX), a2);
        }
        case TIMBRE_PARABOLA:
            return simde_mm_subs_epi16(simde_mm_set1_epi16(Q15_MAX), simde_mm_mulhrs_epi16(pq, pq));
        case TIMBRE_SINE:
        default: {
            q15_t idx[8], out[8];
            simde_mm_storeu_si128((simde__m128i*)idx, pq);
            for (int k = 0; k < 8; k++) out[k] = spectral_lut_sin((uq16_t)(uint16_t)idx[k], lut);
            return simde_mm_loadu_si128((const simde__m128i*)out);
        }
    }
}

/* widen 8xint16 Q15 -> 8 floats (scaled), 1-op sign-extend via cvtepi16_epi32. */
static inline void q15x8_to_floatx8(simde__m128i wq, simde__m128 inv,
                                    simde__m128* lo, simde__m128* hi) {
    simde__m128i e_lo = simde_mm_cvtepi16_epi32(wq);                       /* lanes 0..3 */
    simde__m128i e_hi = simde_mm_cvtepi16_epi32(simde_mm_srli_si128(wq, 8)); /* lanes 4..7 */
    *lo = simde_mm_mul_ps(simde_mm_cvtepi32_ps(e_lo), inv);
    *hi = simde_mm_mul_ps(simde_mm_cvtepi32_ps(e_hi), inv);
}

/* scalar int-NCO x8 -> packed 8xQ15 phase index (the realizable serial phase). */
static inline simde__m128i nco_pack8(SpectralPhaseNco* nco) {
    q15_t idx[8];
    for (int k = 0; k < 8; k++) idx[k] = spectral_phase_nco_step(nco);
    return simde_mm_loadu_si128((const simde__m128i*)idx);
}

/* ===== vectorized 8-wide cubic NCO (uint32 lanes, stride-8 fwd differences) =
 * 8 phase lanes in uint32 (top 16 bits = Q15 index, low 16 = fractional). Each
 * lane runs an independent stride-8 cubic forward-difference chain. The init
 * (cold, float boundary) seeds from the scalar uint64 NCO so it carries the cubic
 * c3 term; narrowing to uint32 keeps only 16 fractional bits -- enough for the
 * realistic chirp here, and this is a THROUGHPUT ceiling probe (production would
 * re-validate precision or keep uint64 lanes). 2 regs (lo=lanes0..3, hi=4..7). */
typedef struct { simde__m128i acc_lo, acc_hi, d1_lo, d1_hi, d2_lo, d2_hi, d3; } Nco8;

static void nco8_init(Nco8* v, float phase0, float alpha, float c2, float c3) {
    SpectralPhaseNco s; spectral_phase_nco_init(&s, phase0, alpha, c2, c3);
    uint64_t th[25];
    for (int i = 0; i < 25; i++) { th[i] = s.acc; s.acc += s.d1; s.d1 += s.d2; s.d2 += s.d3; }
    uint32_t acc[8], d1[8], d2[8];
    for (int k = 0; k < 8; k++) {
        acc[k] = (uint32_t)(th[k] >> 32);
        d1[k]  = (uint32_t)((th[k + 8]  - th[k]) >> 32);
        d2[k]  = (uint32_t)((th[k + 16] - 2*th[k + 8] + th[k]) >> 32);
    }
    uint32_t d3 = (uint32_t)((th[24] - 3*th[16] + 3*th[8] - th[0]) >> 32);
    v->acc_lo = simde_mm_loadu_si128((const simde__m128i*)&acc[0]);
    v->acc_hi = simde_mm_loadu_si128((const simde__m128i*)&acc[4]);
    v->d1_lo  = simde_mm_loadu_si128((const simde__m128i*)&d1[0]);
    v->d1_hi  = simde_mm_loadu_si128((const simde__m128i*)&d1[4]);
    v->d2_lo  = simde_mm_loadu_si128((const simde__m128i*)&d2[0]);
    v->d2_hi  = simde_mm_loadu_si128((const simde__m128i*)&d2[4]);
    v->d3     = simde_mm_set1_epi32((int32_t)d3);
}

// SPECTRAL_Q_DOMAIN BEGIN
static inline simde__m128i nco8_step(Nco8* v) {
    const simde__m128i bias = simde_mm_set1_epi32(0x00008000); /* half-LSB round */
    simde__m128i i_lo = simde_mm_srli_epi32(simde_mm_add_epi32(v->acc_lo, bias), 16);
    simde__m128i i_hi = simde_mm_srli_epi32(simde_mm_add_epi32(v->acc_hi, bias), 16);
    simde__m128i idx  = simde_mm_packus_epi32(i_lo, i_hi);     /* 8x uint16 index */
    v->acc_lo = simde_mm_add_epi32(v->acc_lo, v->d1_lo);
    v->acc_hi = simde_mm_add_epi32(v->acc_hi, v->d1_hi);
    v->d1_lo  = simde_mm_add_epi32(v->d1_lo, v->d2_lo);
    v->d1_hi  = simde_mm_add_epi32(v->d1_hi, v->d2_hi);
    v->d2_lo  = simde_mm_add_epi32(v->d2_lo, v->d3);
    v->d2_hi  = simde_mm_add_epi32(v->d2_hi, v->d3);
    return idx;
}
// SPECTRAL_Q_DOMAIN END

/* ===== pack8 desktop kernel: phase mode 0=scalar NCO, 1=vectorized, 2=none == */
static double bench_pack8_f(SpectralTimbre timbre, const q15_t* lut, int phase_mode, double* sink) {
    float buf[SEG_LEN];
    const simde__m128 inv = simde_mm_set1_ps(Q15_TO_FLOAT((q15_t)1));
    const simde__m128i fixed_idx = simde_mm_set_epi16(7000,6000,5000,4000,3000,2000,1000,0);

    for (int warm = 0; warm < 2; warm++) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    double chk = 0.0;
    int iters = warm ? ITERS : 64;
    for (int it = 0; it < iters; it++) {
        memset(buf, 0, sizeof(buf));
        SpectralPhaseNco nco; spectral_phase_nco_init(&nco, SEG_PHASE0, SEG_ALPHA, 0.0f, 0.0f);
        Nco8 v8; nco8_init(&v8, SEG_PHASE0, SEG_ALPHA, 0.0f, 0.0f);
        simde__m128 amp = simde_mm_set_ps(SEG_AMP0 + 3*SEG_DAMP, SEG_AMP0 + 2*SEG_DAMP,
                                          SEG_AMP0 + SEG_DAMP, SEG_AMP0);
        simde__m128 amp_hi = simde_mm_add_ps(amp, simde_mm_set1_ps(4.0f * SEG_DAMP));
        const simde__m128 ampstep = simde_mm_set1_ps(8.0f * SEG_DAMP);
        for (int j = 0; j < SEG_LEN; j += 8) {
            simde__m128i pq = (phase_mode == 0) ? nco_pack8(&nco)
                            : (phase_mode == 1) ? nco8_step(&v8)
                                                : fixed_idx;
            simde__m128i wq = pack8_eval(pq, timbre, lut);
            simde__m128 wf_lo, wf_hi;
            q15x8_to_floatx8(wq, inv, &wf_lo, &wf_hi);
            simde__m128 b_lo = simde_mm_loadu_ps(&buf[j]);
            simde__m128 b_hi = simde_mm_loadu_ps(&buf[j + 4]);
            b_lo = simde_mm_add_ps(b_lo, simde_mm_mul_ps(amp,    wf_lo));
            b_hi = simde_mm_add_ps(b_hi, simde_mm_mul_ps(amp_hi, wf_hi));
            simde_mm_storeu_ps(&buf[j],     b_lo);
            simde_mm_storeu_ps(&buf[j + 4], b_hi);
            amp    = simde_mm_add_ps(amp,    ampstep);
            amp_hi = simde_mm_add_ps(amp_hi, ampstep);
        }
        chk += consume_buf(buf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (warm) {
        *sink += chk;
        return ns_per_sample((double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec));
    }
    }
    return 0.0;
}

/* ===== B2 experiment: amp ramp + scale folded into Q15 (mulhrs) ============
 * The shipping kernel (bench_pack8_f / osc_simd_q15_segment) widens Q15->float
 * THEN does amp*wave + accumulate in float -- 2 float muls per 8 samples plus a
 * float amp-ramp carry. This variant folds the amp into ONE 8-wide mulhrs in Q15
 * BEFORE the widen, so the only float left is the (unavoidable, float-dst) load/
 * add/store accumulate. Question: does keeping the scale in Q close the gap to 2x?
 * Cost: the amp ramp is now an int16 carry (ampq += astep), so d_amp quantizes to
 * a whole Q15 LSB per block -- qamp_precision() measures that drift vs the float
 * amp. Phase = vec NCO (the shipping c3==0 path), so Bq/Bv is the apples-to-apples
 * throughput read. amp assumed in [0,1] here (Q15-representable); a production amp
 * >1.0 would overflow Q15 and is part of the decline-on-data risk. */
static inline simde__m128i qamp_init8(void) {
    return simde_mm_set_epi16(
        (int16_t)lrintf((SEG_AMP0 + 7*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 6*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 5*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 4*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 3*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 2*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 1*SEG_DAMP) * 32768.0f),
        (int16_t)lrintf((SEG_AMP0 + 0*SEG_DAMP) * 32768.0f));
}

static double bench_pack8_qamp(SpectralTimbre timbre, const q15_t* lut, double* sink) {
    float buf[SEG_LEN];
    const simde__m128 inv = simde_mm_set1_ps(Q15_TO_FLOAT((q15_t)1));
    const simde__m128i astepv = simde_mm_set1_epi16((int16_t)lrintf(8.0f * SEG_DAMP * 32768.0f));
    for (int warm = 0; warm < 2; warm++) {
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    double chk = 0.0;
    int iters = warm ? ITERS : 64;
    for (int it = 0; it < iters; it++) {
        memset(buf, 0, sizeof(buf));
        Nco8 v8; nco8_init(&v8, SEG_PHASE0, SEG_ALPHA, 0.0f, 0.0f);
        simde__m128i ampq = qamp_init8();
        for (int j = 0; j < SEG_LEN; j += 8) {
            simde__m128i wq = pack8_eval(nco8_step(&v8), timbre, lut);
            simde__m128i sq = simde_mm_mulhrs_epi16(wq, ampq);  /* amp*wave in Q15, 8-wide */
            simde__m128 wf_lo, wf_hi;
            q15x8_to_floatx8(sq, inv, &wf_lo, &wf_hi);
            simde_mm_storeu_ps(&buf[j],     simde_mm_add_ps(simde_mm_loadu_ps(&buf[j]),     wf_lo));
            simde_mm_storeu_ps(&buf[j + 4], simde_mm_add_ps(simde_mm_loadu_ps(&buf[j + 4]), wf_hi));
            ampq = simde_mm_adds_epi16(ampq, astepv);
        }
        chk += consume_buf(buf);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (warm) { *sink += chk;
        return ns_per_sample((double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec)); }
    }
    return 0.0;
}

/* B2 precision: Q15-amp vs float-amp (SAME Q15 wave + vec phase) over one segment.
 * Isolates the cost of quantizing the amp ramp to Q15 -> max abs diff in dBFS. */
static void qamp_precision(SpectralTimbre timbre, const q15_t* lut,
                           double* max_dbfs, double* rms_dbfs) {
    float bf[SEG_LEN], bq[SEG_LEN];
    memset(bf, 0, sizeof(bf)); memset(bq, 0, sizeof(bq));
    const simde__m128 inv = simde_mm_set1_ps(Q15_TO_FLOAT((q15_t)1));
    {   /* float-amp reference (matches bench_pack8_f vec path) */
        Nco8 v8; nco8_init(&v8, SEG_PHASE0, SEG_ALPHA, 0.0f, 0.0f);
        simde__m128 amp = simde_mm_set_ps(SEG_AMP0 + 3*SEG_DAMP, SEG_AMP0 + 2*SEG_DAMP,
                                          SEG_AMP0 + SEG_DAMP, SEG_AMP0);
        simde__m128 amp_hi = simde_mm_add_ps(amp, simde_mm_set1_ps(4.0f * SEG_DAMP));
        const simde__m128 ampstep = simde_mm_set1_ps(8.0f * SEG_DAMP);
        for (int j = 0; j < SEG_LEN; j += 8) {
            simde__m128 lo, hi; q15x8_to_floatx8(pack8_eval(nco8_step(&v8), timbre, lut), inv, &lo, &hi);
            simde_mm_storeu_ps(&bf[j],     simde_mm_mul_ps(amp,    lo));
            simde_mm_storeu_ps(&bf[j + 4], simde_mm_mul_ps(amp_hi, hi));
            amp = simde_mm_add_ps(amp, ampstep); amp_hi = simde_mm_add_ps(amp_hi, ampstep);
        }
    }
    {   /* Q15-amp */
        Nco8 v8; nco8_init(&v8, SEG_PHASE0, SEG_ALPHA, 0.0f, 0.0f);
        simde__m128i ampq = qamp_init8();
        const simde__m128i astepv = simde_mm_set1_epi16((int16_t)lrintf(8.0f * SEG_DAMP * 32768.0f));
        for (int j = 0; j < SEG_LEN; j += 8) {
            simde__m128i sq = simde_mm_mulhrs_epi16(pack8_eval(nco8_step(&v8), timbre, lut), ampq);
            simde__m128 lo, hi; q15x8_to_floatx8(sq, inv, &lo, &hi);
            simde_mm_storeu_ps(&bq[j], lo); simde_mm_storeu_ps(&bq[j + 4], hi);
            ampq = simde_mm_adds_epi16(ampq, astepv);
        }
    }
    double ma = 0.0, ss_d = 0.0, ss_s = 0.0;
    for (int i = 0; i < SEG_LEN; i++) {
        double d = (double)bq[i] - (double)bf[i];
        if (fabs(d) > ma) ma = fabs(d);
        ss_d += d * d; ss_s += (double)bf[i] * (double)bf[i];
    }
    double rms_d = sqrt(ss_d / SEG_LEN);
    *max_dbfs = (ma > 0.0) ? 20.0 * log10(ma) : -300.0;
    *rms_dbfs = (rms_d > 0.0) ? 20.0 * log10(rms_d) : -300.0;
    (void)ss_s;
}
#endif /* OSC_SIMD_GENERIC */

int main(void) {
    double sink = 0.0;
#if !defined(OSC_SIMD_GENERIC)
    printf("bench_q15_pack8: SIMDe generic SIMD not available on this target; skipping.\n");
    return 0;
#else
    static q15_t sine_lut[SPECTRAL_OSC_LUT_SIZE + 1];
    spectral_osc_q15_init_sine_lut(sine_lut);

    printf("== Q5c pack8 probe (ns/sample, %d-sample seg, %d iters) ==\n", SEG_LEN, ITERS);
    printf("  A=prod-fsimd  A8=lean-favx8(FAIR 8-wide float)  B=pack8(scalar ph)  Bv=pack8(vec ph)  B0=pack8(no ph)\n");
    printf("  %-9s %9s %9s %9s %9s %9s   %8s %8s\n",
           "timbre", "A prod", "A8 lean8", "B scal", "Bv vec", "B0 noph",
           "B/A8", "Bv/A8");
    for (size_t t = 0; t < N_TIMBRES; t++) {
        SpectralTimbre tb = k_timbres[t].timbre;
        double a   = bench_prod_fsimd4(tb, &sink);
        double a8  = bench_lean_favx8(tb, sine_lut, &sink);
        double b   = bench_pack8_f(tb, sine_lut, 0, &sink);
        double bv  = bench_pack8_f(tb, sine_lut, 1, &sink);
        double b0  = bench_pack8_f(tb, sine_lut, 2, &sink);
        printf("  %-9s %9.3f %9.3f %9.3f %9.3f %9.3f   %7.2fx %7.2fx\n",
               k_timbres[t].name, a, a8, b, bv, b0, b/a8, bv/a8);
    }
    printf("(ratios < 1.00x = pack8 faster than the FAIR lean 8-wide float kernel; sink=%.3e)\n", sink);

    /* ---- B2 experiment: amp ramp + scale folded into Q15 (mulhrs) ---- */
    printf("\n== B2: Q15-amp (mulhrs) vs float-amp pack8 [both vec phase] ==\n");
    printf("  Bv = shipping float-amp (vec ph); Bq = amp folded into Q15. Bq/Bv < 1.00x = faster.\n");
    printf("  %-9s %9s %9s   %8s   %10s %10s\n",
           "timbre", "Bv float", "Bq Q15", "Bq/Bv", "amp maxErr", "amp rmsErr");
    for (size_t t = 0; t < N_TIMBRES; t++) {
        SpectralTimbre tb = k_timbres[t].timbre;
        double bv = bench_pack8_f(tb, sine_lut, 1, &sink);
        double bq = bench_pack8_qamp(tb, sine_lut, &sink);
        double maxe, rmse; qamp_precision(tb, sine_lut, &maxe, &rmse);
        printf("  %-9s %9.3f %9.3f   %7.2fx   %7.1f dB %7.1f dB\n",
               k_timbres[t].name, bv, bq, bq/bv, maxe, rmse);
    }
    printf("(amp err = Q15-amp vs float-amp over the segment; sink=%.3e)\n", sink);
    return 0;
#endif
}
