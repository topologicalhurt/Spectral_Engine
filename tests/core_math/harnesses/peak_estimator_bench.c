/* peak_estimator_bench.c - local peak-estimator speed/error harness
 *
 * Example:
 *   cc -O3 -std=c11 -I spectral_engine/core -I spectral_engine/analysis \
 *      spectral_engine/analysis/spectral_peak_estimator.c \
 *      spectral_engine/core/spectral_fast_math.c \
 *      spectral_engine/core/spectral_windows.c \
 *      tests/core_math/harnesses/peak_estimator_bench.c -lm -o /tmp/peak_estimator_bench
 *
 * On macOS, add: -framework Accelerate
 *
 * Optional approximation profile:
 *   -DSPECTRAL_ENABLE_APPROX_PEAK_LOG=1
 *   -DSPECTRAL_ENABLE_APPROX_TRIG=1
 *   -DSPECTRAL_ENABLE_APPROX_INV_SQRT=1
 */
#include "spectral_peak_estimator.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SPECTRAL_PEAK_BENCH_CASES
#define SPECTRAL_PEAK_BENCH_CASES 256u
#endif
#ifndef SPECTRAL_PEAK_BENCH_BLOCKS
#define SPECTRAL_PEAK_BENCH_BLOCKS 64u
#endif
#ifndef SPECTRAL_PEAK_BENCH_ITERS
#define SPECTRAL_PEAK_BENCH_ITERS 4096u
#endif

typedef struct BenchCase {
    float magsq[3];
    float phase[3];
    float next_max_magsq;
    int best_next_bin;
} BenchCase;

typedef struct BenchComplex {
    float re;
    float im;
} BenchComplex;

typedef struct BenchExpected {
    float offset;
    float amp;
    int fallback;
} BenchExpected;

static volatile float spectral_peak_bench_sink = 0.0f;

static uint64_t bench_now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t bench_cycle_counter(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_ia32_rdtsc();
#elif defined(__aarch64__)
    uint64_t ticks = 0u;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(ticks));
    return ticks;
#else
    return 0u;
#endif
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static float clamp_offset(float p) {
    if (p > 0.5f) return 0.5f;
    if (p < -0.5f) return -0.5f;
    return p;
}

static float ref_log_parabolic(const BenchCase* c) {
    float left = fmaxf(c->magsq[0], SPECTRAL_TRACK_LOG_FLOOR);
    float center = fmaxf(c->magsq[1], SPECTRAL_TRACK_LOG_FLOOR);
    float right = fmaxf(c->magsq[2], SPECTRAL_TRACK_LOG_FLOOR);
    float log_l = logf(left);
    float log_c = logf(center);
    float log_r = logf(right);
    float denom = log_l - 2.0f * log_c + log_r;
    if (!isfinite(denom) || fabsf(denom) < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) return 0.0f;
    return clamp_offset(0.5f * (log_l - log_r) / denom);
}

static int ref_complex(const BenchCase* c, int idx, BenchComplex* out) {
    float mag = 0.0f;
    if (!c || !out || idx < 0 || idx > 2 || c->magsq[idx] < 0.0f ||
        !isfinite(c->magsq[idx]) || !isfinite(c->phase[idx])) {
        return 0;
    }
    mag = sqrtf(c->magsq[idx]);
    out->re = mag * cosf(c->phase[idx]);
    out->im = mag * sinf(c->phase[idx]);
    return isfinite(out->re) && isfinite(out->im);
}

static int ref_jacobsen(const BenchCase* c, float* out) {
    BenchComplex xm, x0, xp;
    float num_re = 0.0f, num_im = 0.0f;
    float den_re = 0.0f, den_im = 0.0f;
    float den_mag2 = 0.0f;
    if (!ref_complex(c, 0, &xm) || !ref_complex(c, 1, &x0) || !ref_complex(c, 2, &xp)) return 0;
    num_re = xm.re - xp.re;
    num_im = xm.im - xp.im;
    den_re = 2.0f * x0.re - xm.re - xp.re;
    den_im = 2.0f * x0.im - xm.im - xp.im;
    den_mag2 = den_re * den_re + den_im * den_im;
    if (!(den_mag2 > 1.0e-30f) || !isfinite(den_mag2)) return 0;
    *out = clamp_offset((num_re * den_re + num_im * den_im) / den_mag2);
    return isfinite(*out);
}

static int ref_mag_parabolic(const BenchCase* c, float* out) {
    float left = sqrtf(c->magsq[0]);
    float center = sqrtf(c->magsq[1]);
    float right = sqrtf(c->magsq[2]);
    float denom = left - 2.0f * center + right;
    if (!isfinite(denom) || fabsf(denom) < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) {
        *out = 0.0f;
        return 1;
    }
    *out = clamp_offset(0.5f * (left - right) / denom);
    return isfinite(*out);
}

static float ref_quinn_tau(float x) {
    const float sqrt_6 = 2.449489742783178f;
    const float sqrt_2_over_3 = 0.816496580927726f;
    float a = 3.0f * x * x + 6.0f * x + 1.0f;
    float b_num = x + 1.0f - sqrt_2_over_3;
    float b_den = x + 1.0f + sqrt_2_over_3;
    float b_ratio = b_num / b_den;
    if (x < 0.0f || !isfinite(x) || a <= 0.0f ||
        b_num <= 0.0f || b_den <= 0.0f || b_ratio <= 0.0f) {
        return 0.0f;
    }
    return 0.25f * logf(a) - (sqrt_6 / 24.0f) * logf(b_ratio);
}

static int ref_quinn_second(const BenchCase* c, float* out) {
    BenchComplex xm, x0, xp;
    float mag0 = c->magsq[1];
    float ap = 0.0f, am = 0.0f;
    float den_p = 0.0f, den_m = 0.0f;
    float dp = 0.0f, dm = 0.0f;
    if (!ref_complex(c, 0, &xm) || !ref_complex(c, 1, &x0) || !ref_complex(c, 2, &xp)) return 0;
    if (!(mag0 > 0.0f) || !isfinite(mag0)) return 0;
    ap = (xp.re * x0.re + xp.im * x0.im) / mag0;
    am = (xm.re * x0.re + xm.im * x0.im) / mag0;
    den_p = 1.0f - ap;
    den_m = 1.0f - am;
    if (!isfinite(ap) || !isfinite(am) || fabsf(den_p) < 1.0e-12f || fabsf(den_m) < 1.0e-12f) return 0;
    dp = -ap / den_p;
    dm =  am / den_m;
    if (!isfinite(dp) || !isfinite(dm)) return 0;
    *out = clamp_offset(0.5f * (dp + dm) + ref_quinn_tau(dp * dp) - ref_quinn_tau(dm * dm));
    return isfinite(*out);
}

static float ref_offset(SpectralPeakEstimatorType type, const BenchCase* c, int* fallback) {
    float offset = 0.0f;
    *fallback = 0;
    switch (type) {
        case SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC:
            if (ref_mag_parabolic(c, &offset)) return offset;
            break;
        case SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX:
            if (ref_jacobsen(c, &offset)) return offset;
            break;
        case SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX:
            if (ref_jacobsen(c, &offset)) {
                return clamp_offset(offset * spectral_peak_candan_correction_for_n_freqs(3u));
            }
            break;
        case SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND:
            if (ref_quinn_second(c, &offset)) return offset;
            break;
        case SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC:
        case SPECTRAL_PEAK_ESTIMATOR_AUTO:
        case SPECTRAL_PEAK_ESTIMATOR_COUNT:
        default:
            return ref_log_parabolic(c);
    }
    *fallback = 1;
    return ref_log_parabolic(c);
}

static void fill_cases(BenchCase* cases, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float a = (float)(i % 29u) / 29.0f;
        float b = (float)((i * 7u) % 31u) / 31.0f;
        float c = (float)((i * 11u) % 37u) / 37.0f;
        cases[i].magsq[0] = 1.0f + 2.0f * a;
        cases[i].magsq[1] = 7.0f + 3.0f * b;
        cases[i].magsq[2] = 1.25f + 2.5f * c;
        cases[i].phase[0] = -0.75f + 0.5f * a;
        cases[i].phase[1] = -0.10f + 0.8f * b;
        cases[i].phase[2] = 0.25f + 0.6f * c;
        cases[i].next_max_magsq = 4.0f + 5.0f * b;
        cases[i].best_next_bin = (int)(i % 3u);
    }
}

static void run_method(SpectralPeakEstimatorType type, const char* name,
                       const BenchCase* cases, size_t n_cases) {
    SpectralPeakEstimateInput inputs[SPECTRAL_PEAK_BENCH_CASES];
    BenchExpected expected[SPECTRAL_PEAK_BENCH_CASES];
    double ns_per_estimate[SPECTRAL_PEAK_BENCH_BLOCKS];
    double ticks_per_estimate[SPECTRAL_PEAK_BENCH_BLOCKS];
    double max_offset_error = 0.0;
    double max_amp_rel_error = 0.0;
    uint64_t fallbacks = 0u;
    uint64_t estimates = 0u;

    memset(inputs, 0, sizeof(inputs));
    memset(expected, 0, sizeof(expected));
    for (size_t i = 0; i < n_cases; i++) {
        const BenchCase* bc = &cases[i];
        int ref_fallback = 0;

        inputs[i].magsq_row = bc->magsq;
        inputs[i].phase_row = bc->phase;
        inputs[i].n_freqs = 3u;
        inputs[i].bin = 1u;
        inputs[i].curr_magsq = bc->magsq[1];
        inputs[i].next_max_magsq = bc->next_max_magsq;
        inputs[i].best_next_bin = bc->best_next_bin;
        inputs[i].freq_step_omega = 2.0f;
        inputs[i].freq_step_df = 0.5f;
        inputs[i].inv_hop = 0.25f;
        inputs[i].hop_float = 4.0f;
        inputs[i].candan_correction = spectral_peak_candan_correction_for_n_freqs(inputs[i].n_freqs);
        inputs[i].type = type;

        expected[i].offset = ref_offset(type, bc, &ref_fallback);
        expected[i].amp = sqrtf(bc->magsq[1]);
        expected[i].fallback = ref_fallback;
    }

    /* Correctness accounting is intentionally outside the timed loop. The
     * benchmark's p50/p95 numbers should measure the estimator path, not the
     * reference formulas used to report max error. */
    for (size_t i = 0; i < n_cases; i++) {
        SpectralPeakEstimate out;
        memset(&out, 0, sizeof(out));
        if (spectral_peak_estimate_validated(&inputs[i], &out)) {
            double offset_error = fabs((double)out.bin_offset - (double)expected[i].offset);
            double amp_rel = fabs((double)out.amp - (double)expected[i].amp) / (double)expected[i].amp;
            if (offset_error > max_offset_error) max_offset_error = offset_error;
            if (amp_rel > max_amp_rel_error) max_amp_rel_error = amp_rel;
            if (out.flags & SPECTRAL_PEAK_ESTIMATE_USED_FALLBACK) fallbacks++;
            spectral_peak_bench_sink += (float)expected[i].fallback * 0.0f;
            estimates++;
        }
    }

    for (size_t block = 0; block < SPECTRAL_PEAK_BENCH_BLOCKS; block++) {
        uint64_t t0 = bench_now_ns();
        uint64_t c0 = bench_cycle_counter();
        for (size_t i = 0; i < SPECTRAL_PEAK_BENCH_ITERS; i++) {
            const size_t case_idx = (i + block * 17u) % n_cases;
            SpectralPeakEstimate out;
            memset(&out, 0, sizeof(out));

            if (spectral_peak_estimate_validated(&inputs[case_idx], &out)) {
                spectral_peak_bench_sink += out.omega + out.amp;
            }
        }
        {
            uint64_t c1 = bench_cycle_counter();
            uint64_t t1 = bench_now_ns();
            uint64_t dt = t1 - t0;
            uint64_t dc = (c0 != 0u && c1 >= c0) ? (c1 - c0) : 0u;
            ns_per_estimate[block] = (double)dt / (double)SPECTRAL_PEAK_BENCH_ITERS;
            ticks_per_estimate[block] = dc ? (double)dc / (double)SPECTRAL_PEAK_BENCH_ITERS : 0.0;
        }
    }

    qsort(ns_per_estimate, SPECTRAL_PEAK_BENCH_BLOCKS, sizeof(double), cmp_double);
    qsort(ticks_per_estimate, SPECTRAL_PEAK_BENCH_BLOCKS, sizeof(double), cmp_double);
    printf("%-20s p50_ns=%9.3f p95_ns=%9.3f p50_ticks=%9.3f "
           "max_offset_err=%0.7f max_amp_rel_err=%0.7f fallback_rate=%0.5f\n",
           name,
           ns_per_estimate[SPECTRAL_PEAK_BENCH_BLOCKS / 2u],
           ns_per_estimate[(SPECTRAL_PEAK_BENCH_BLOCKS * 95u) / 100u],
           ticks_per_estimate[SPECTRAL_PEAK_BENCH_BLOCKS / 2u],
           max_offset_error,
           max_amp_rel_error,
           estimates ? (double)fallbacks / (double)estimates : 0.0);
}

int main(void) {
    BenchCase cases[SPECTRAL_PEAK_BENCH_CASES];
    fill_cases(cases, SPECTRAL_PEAK_BENCH_CASES);
    printf("flags approx_log=%d approx_trig=%d approx_inv_sqrt=%d blocks=%u iters=%u\n",
           (int)SPECTRAL_ENABLE_APPROX_PEAK_LOG,
           (int)SPECTRAL_ENABLE_APPROX_TRIG,
           (int)SPECTRAL_ENABLE_APPROX_INV_SQRT,
           (unsigned)SPECTRAL_PEAK_BENCH_BLOCKS,
           (unsigned)SPECTRAL_PEAK_BENCH_ITERS);
    run_method(SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC, "log-parabolic", cases, SPECTRAL_PEAK_BENCH_CASES);
    run_method(SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC, "mag-parabolic", cases, SPECTRAL_PEAK_BENCH_CASES);
    run_method(SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX, "jacobsen-complex", cases, SPECTRAL_PEAK_BENCH_CASES);
    run_method(SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX, "candan-complex", cases, SPECTRAL_PEAK_BENCH_CASES);
    run_method(SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND, "quinn-second", cases, SPECTRAL_PEAK_BENCH_CASES);
    return spectral_peak_bench_sink == 12345.0f ? 1 : 0;
}
