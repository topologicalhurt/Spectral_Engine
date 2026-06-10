/* peak_estimator_sweep.c - synthetic ground-truth peak-estimator sweep
 *
 * This harness complements peak_estimator_bench.c.
 *
 * peak_estimator_bench.c measures estimator throughput and internal consistency
 * on synthetic triplets. This file generates actual windowed time-domain
 * sinusoids, computes the three relevant DFT bins, and compares estimator
 * offsets against the known fractional-bin tone location.
 */
#include "spectral_peak_estimator.h"
#include "spectral_windows.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef SPECTRAL_PEAK_SWEEP_N
#define SPECTRAL_PEAK_SWEEP_N 1024u
#endif

#ifndef SPECTRAL_PEAK_SWEEP_OFFSETS
#define SPECTRAL_PEAK_SWEEP_OFFSETS 41u
#endif

#ifndef SPECTRAL_PEAK_SWEEP_ITERS
#define SPECTRAL_PEAK_SWEEP_ITERS 8u
#endif

#ifndef SPECTRAL_PEAK_SWEEP_CENTER_BIN
#define SPECTRAL_PEAK_SWEEP_CENTER_BIN 37u
#endif

typedef struct SweepWindow {
    SpectralWindowType type;
    const char* name;
} SweepWindow;

typedef struct SweepMethod {
    SpectralPeakEstimatorType type;
    const char* name;
} SweepMethod;

typedef struct SweepStats {
    double sum_sq_err;
    double max_abs_err;
    double total_ns;
    unsigned cases;
    unsigned valid;
    unsigned fallbacks;
    unsigned complex_used;
} SweepStats;

static uint64_t sweep_now_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static float sweep_hash_noise(uint32_t index, uint32_t salt) {
    uint32_t x = index * 747796405u + salt * 2891336453u + 0x9e3779b9u;
    x ^= x >> 16u;
    x *= 2246822519u;
    x ^= x >> 13u;
    x *= 3266489917u;
    x ^= x >> 16u;
    return ((float)(x & 0x00ffffffu) / 8388608.0f) - 1.0f;
}

static float sweep_noise_amp_from_snr_db(float snr_db) {
    if (snr_db >= 100.0f) return 0.0f;
    return powf(10.0f, -snr_db / 20.0f);
}

static void sweep_dft_triplet(const float* window,
                              size_t n_fft,
                              float tone_bin,
                              size_t center_bin,
                              float snr_db,
                              float magsq[3],
                              float phase[3])
{
    float noise_amp = sweep_noise_amp_from_snr_db(snr_db);
    uint32_t salt = (uint32_t)(tone_bin * 1000.0f) ^ (uint32_t)(snr_db * 13.0f);

    for (int tap = -1; tap <= 1; tap++) {
        size_t k = center_bin + (size_t)tap;
        double re = 0.0;
        double im = 0.0;

        for (size_t n = 0; n < n_fft; n++) {
            double tone_phase = (double)SPECTRAL_TWO_PI * (double)tone_bin * (double)n / (double)n_fft;
            double bin_phase = (double)SPECTRAL_TWO_PI * (double)k * (double)n / (double)n_fft;
            double x = cos(tone_phase);
            if (noise_amp > 0.0f) {
                x += (double)noise_amp * (double)sweep_hash_noise((uint32_t)n, salt);
            }
            x *= (double)window[n];

            re += x * cos(bin_phase);
            im -= x * sin(bin_phase);
        }

        magsq[tap + 1] = (float)(re * re + im * im);
        phase[tap + 1] = atan2f((float)im, (float)re);
    }
}

static void sweep_run_one(const SweepWindow* win,
                          const SweepMethod* method,
                          const float* window,
                          float snr_db)
{
    SweepStats stats;
    memset(&stats, 0, sizeof(stats));

    for (unsigned iter = 0u; iter < SPECTRAL_PEAK_SWEEP_ITERS; iter++) {
        for (unsigned oi = 0u; oi < SPECTRAL_PEAK_SWEEP_OFFSETS; oi++) {
            float denom = (float)(SPECTRAL_PEAK_SWEEP_OFFSETS - 1u);
            float true_offset = -0.49f + 0.98f * ((float)oi / denom);
            float tone_bin = (float)SPECTRAL_PEAK_SWEEP_CENTER_BIN + true_offset;
            float magsq[3] = {0.0f, 0.0f, 0.0f};
            float phase[3] = {0.0f, 0.0f, 0.0f};
            SpectralPeakEstimateInput input;
            SpectralPeakEstimate out;
            const SpectralWindowDescriptor* desc = spectral_window_descriptor(win->type);
            uint64_t start_ns = 0u;
            uint64_t end_ns = 0u;

            sweep_dft_triplet(window, SPECTRAL_PEAK_SWEEP_N, tone_bin,
                              SPECTRAL_PEAK_SWEEP_CENTER_BIN, snr_db,
                              magsq, phase);

            memset(&input, 0, sizeof(input));
            memset(&out, 0, sizeof(out));
            input.magsq_row = magsq;
            input.phase_row = phase;
            input.n_freqs = 3u;
            input.bin = 1u;
            input.curr_magsq = magsq[1];
            input.next_max_magsq = magsq[1];
            input.best_next_bin = 1;
            input.freq_step_omega = 1.0f;
            input.freq_step_df = 1.0f;
            input.inv_hop = 1.0f;
            input.hop_float = 1.0f;
            input.candan_correction =
                spectral_peak_candan_correction_for_n_freqs((SPECTRAL_PEAK_SWEEP_N / 2u) + 1u);
            input.interp_magsq = (desc && desc->interp_magsq)
                ? desc->interp_magsq
                : spectral_window_interp_magsq_parabolic;
            input.type = method->type;

            start_ns = sweep_now_ns();
            if (spectral_peak_estimate(&input, &out)) {
                double err = (double)out.bin_offset - (double)true_offset;
                double abs_err = fabs(err);
                stats.valid++;
                stats.sum_sq_err += err * err;
                if (abs_err > stats.max_abs_err) stats.max_abs_err = abs_err;
                if (out.flags & SPECTRAL_PEAK_ESTIMATE_USED_FALLBACK) stats.fallbacks++;
                if (out.flags & SPECTRAL_PEAK_ESTIMATE_COMPLEX_USED) stats.complex_used++;
            }
            end_ns = sweep_now_ns();
            stats.total_ns += (double)(end_ns - start_ns);
            stats.cases++;
        }
    }

    {
        double rms = stats.valid ? sqrt(stats.sum_sq_err / (double)stats.valid) : 0.0;
        double fallback_rate = stats.valid ? (double)stats.fallbacks / (double)stats.valid : 0.0;
        double complex_rate = stats.valid ? (double)stats.complex_used / (double)stats.valid : 0.0;
        double valid_rate = stats.cases ? (double)stats.valid / (double)stats.cases : 0.0;
        double avg_ns = stats.cases ? stats.total_ns / (double)stats.cases : 0.0;

        printf("RESULT window=%s estimator=%s snr_db=%.1f cases=%u valid=%u "
               "valid_rate=%.6f rms_err=%.9f max_abs_err=%.9f "
               "fallback_rate=%.6f complex_rate=%.6f avg_ns=%.3f\n",
               win->name,
               method->name,
               snr_db,
               stats.cases,
               stats.valid,
               valid_rate,
               rms,
               stats.max_abs_err,
               fallback_rate,
               complex_rate,
               avg_ns);
    }
}

int main(void) {
    static const SweepWindow windows[] = {
        { SPECTRAL_WINDOW_HANN, "hann" },
        { SPECTRAL_WINDOW_HAMMING, "hamming" },
        { SPECTRAL_WINDOW_BLACKMAN, "blackman" },
        { SPECTRAL_WINDOW_RECTANGULAR, "rectangular" }
    };
    static const SweepMethod methods[] = {
        { SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC, "log-parabolic" },
        { SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC, "mag-parabolic" },
        { SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX, "jacobsen-complex" },
        { SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX, "candan-complex" },
        { SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND, "quinn-second" }
    };
    static const float snr_values[] = { 120.0f, 60.0f, 30.0f };

    float* window = (float*)malloc((size_t)SPECTRAL_PEAK_SWEEP_N * sizeof(float));
    if (!window) return 2;

    printf("SWEEP n_fft=%u center_bin=%u offsets=%u iters=%u\n",
           (unsigned)SPECTRAL_PEAK_SWEEP_N,
           (unsigned)SPECTRAL_PEAK_SWEEP_CENTER_BIN,
           (unsigned)SPECTRAL_PEAK_SWEEP_OFFSETS,
           (unsigned)SPECTRAL_PEAK_SWEEP_ITERS);

    for (size_t wi = 0u; wi < sizeof(windows) / sizeof(windows[0]); wi++) {
        spectral_window_generate(window, (size_t)SPECTRAL_PEAK_SWEEP_N, windows[wi].type);
        for (size_t si = 0u; si < sizeof(snr_values) / sizeof(snr_values[0]); si++) {
            for (size_t mi = 0u; mi < sizeof(methods) / sizeof(methods[0]); mi++) {
                sweep_run_one(&windows[wi], &methods[mi], window, snr_values[si]);
            }
        }
    }

    free(window);
    return 0;
}
