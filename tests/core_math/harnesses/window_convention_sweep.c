/* window_convention_sweep.c - symmetric (N-1) vs periodic (DFT-even, N) window
 * convention: the decision data for REVIEWER_HANDOFF_2 sec 4.4.
 *
 * The engine's SSOT windows are the symmetric (N-1 denominator) forms
 * (spectral_windows.c). The periodic convention (N denominator; numpy/scipy
 * default, exact COLA at dividing hops) is a PROPOSED alternative whose
 * adoption would shift every backend's analysis output and needs golden
 * re-sign-off. This harness quantifies that shift; it decides nothing.
 *
 * Sections (one RESULT-style line each, parsed by the pytest driver):
 *   COLA      overlap-add envelope ripple + gain per window/form/N/hop
 *             (real spectral_overlap_add_envelope_stats contract)
 *   GAINRATIO window coherent gain sum(w)/N per form and their ratio — the
 *             first-order amplitude shift every magsq/amp golden would see
 *   SPECSHIFT max relative magsq delta across the 3-bin peak triplets of a
 *             fractional-bin tone sweep — how far the raw analysis bins move
 *   ESTBIAS   log-parabolic (engine default) frequency-offset error per form
 *             on the same noiseless tone sweep — does estimation accuracy
 *             actually change?
 *
 * The periodic generators below are the proposed forms, local to this harness
 * on purpose: the engine intentionally has no periodic window path today.
 */
#include "spectral_contracts.h"
#include "spectral_peak_estimator.h"
#include "spectral_windows.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SPECTRAL_WINCONV_OFFSETS
#define SPECTRAL_WINCONV_OFFSETS 41u
#endif

#ifndef SPECTRAL_WINCONV_CENTER_BIN
#define SPECTRAL_WINCONV_CENTER_BIN 37u
#endif

typedef void (*WindowFn)(float*, size_t);

/* Periodic (DFT-even) forms: same coefficients as the engine SSOT, denominator
 * N instead of N-1. Doubles for the angle as in the SSOT generators. */
static void periodic_hann(float* w, size_t n) {
    const float scale = (float)(2.0 * SPECTRAL_PI / (double)n);
    for (size_t i = 0; i < n; i++) w[i] = 0.5f * (1.0f - cosf((float)i * scale));
}
static void periodic_hamming(float* w, size_t n) {
    const float scale = (float)(2.0 * SPECTRAL_PI / (double)n);
    for (size_t i = 0; i < n; i++)
        w[i] = SPECTRAL_HAMMING_A0 - SPECTRAL_HAMMING_A1 * cosf((float)i * scale);
}
static void periodic_blackman(float* w, size_t n) {
    const float scale = (float)(2.0 * SPECTRAL_PI / (double)n);
    for (size_t i = 0; i < n; i++) {
        float angle = (float)i * scale;
        w[i] = SPECTRAL_BLACKMAN_B0 - SPECTRAL_BLACKMAN_B1 * cosf(angle)
             + SPECTRAL_BLACKMAN_B2 * cosf(2.0f * angle);
    }
}

static void symmetric_hann(float* w, size_t n)     { spectral_window_hann(w, n); }
static void symmetric_hamming(float* w, size_t n)  { spectral_window_hamming(w, n); }
static void symmetric_blackman(float* w, size_t n) { spectral_window_blackman(w, n); }

typedef struct ConvWindow {
    const char* name;
    SpectralWindowType type;     /* for the engine interp descriptor */
    WindowFn symmetric;          /* engine SSOT generator */
    WindowFn periodic;           /* proposed form (local) */
} ConvWindow;

static const ConvWindow WINDOWS[] = {
    { "hann",     SPECTRAL_WINDOW_HANN,     symmetric_hann,     periodic_hann },
    { "hamming",  SPECTRAL_WINDOW_HAMMING,  symmetric_hamming,  periodic_hamming },
    { "blackman", SPECTRAL_WINDOW_BLACKMAN, symmetric_blackman, periodic_blackman },
};

static void report_cola(const char* window, const char* form,
                        const float* w, size_t n, size_t hop)
{
    double mn = 0.0, mx = 0.0, mean = 0.0;
    if (!spectral_overlap_add_envelope_stats(w, NULL, n, hop, &mn, &mx, &mean) ||
        mean <= 0.0) {
        printf("COLA window=%s form=%s n=%zu hop=%zu gain=nan ripple=nan\n",
               window, form, n, hop);
        return;
    }
    printf("COLA window=%s form=%s n=%zu hop=%zu gain=%.9f ripple=%.9e\n",
           window, form, n, hop, mean, (mx - mn) / mean);
}

static double window_sum(const float* w, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)w[i];
    return s;
}

/* Noiseless 3-bin DFT triplet of a windowed tone at a fractional bin
 * (the peak_estimator_sweep.c signal model, noise path removed). */
static void dft_triplet(const float* window, size_t n_fft, float tone_bin,
                        size_t center_bin, float magsq[3], float phase[3])
{
    for (int tap = -1; tap <= 1; tap++) {
        size_t k = center_bin + (size_t)tap;
        double re = 0.0, im = 0.0;
        for (size_t n = 0; n < n_fft; n++) {
            double tone_phase = (double)SPECTRAL_TWO_PI * (double)tone_bin * (double)n / (double)n_fft;
            double bin_phase = (double)SPECTRAL_TWO_PI * (double)k * (double)n / (double)n_fft;
            double x = cos(tone_phase) * (double)window[n];
            re += x * cos(bin_phase);
            im -= x * sin(bin_phase);
        }
        magsq[tap + 1] = (float)(re * re + im * im);
        phase[tap + 1] = atan2f((float)im, (float)re);
    }
}

static int estimate_offset(const ConvWindow* cw, size_t n_fft,
                           const float magsq[3], const float phase[3],
                           double* out_offset)
{
    SpectralPeakEstimateInput input;
    SpectralPeakEstimate out;
    const SpectralWindowDescriptor* desc = spectral_window_descriptor(cw->type);

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
        spectral_peak_candan_correction_for_n_freqs((n_fft / 2u) + 1u);
    input.interp_magsq = (desc && desc->interp_magsq)
        ? desc->interp_magsq
        : spectral_window_interp_magsq_parabolic;
    input.type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;

    if (!spectral_peak_estimate(&input, &out)) return 0;
    *out_offset = (double)out.bin_offset;
    return 1;
}

static void sweep_forms(const ConvWindow* cw, size_t n_fft,
                        const float* w_sym, const float* w_per)
{
    double max_rel_magsq_delta = 0.0;
    double sum_sq_err_sym = 0.0, max_err_sym = 0.0;
    double sum_sq_err_per = 0.0, max_err_per = 0.0;
    unsigned valid_sym = 0, valid_per = 0, cases = 0;

    for (unsigned oi = 0; oi < SPECTRAL_WINCONV_OFFSETS; oi++) {
        float denom = (float)(SPECTRAL_WINCONV_OFFSETS - 1u);
        float true_offset = -0.49f + 0.98f * ((float)oi / denom);
        float tone_bin = (float)SPECTRAL_WINCONV_CENTER_BIN + true_offset;
        float magsq_s[3], phase_s[3], magsq_p[3], phase_p[3];
        double off = 0.0;

        dft_triplet(w_sym, n_fft, tone_bin, SPECTRAL_WINCONV_CENTER_BIN, magsq_s, phase_s);
        dft_triplet(w_per, n_fft, tone_bin, SPECTRAL_WINCONV_CENTER_BIN, magsq_p, phase_p);
        cases++;

        for (int t = 0; t < 3; t++) {
            double ref = (double)magsq_s[t];
            if (ref > 0.0) {
                double d = fabs((double)magsq_p[t] - ref) / ref;
                if (d > max_rel_magsq_delta) max_rel_magsq_delta = d;
            }
        }

        if (estimate_offset(cw, n_fft, magsq_s, phase_s, &off)) {
            double err = off - (double)true_offset;
            valid_sym++;
            sum_sq_err_sym += err * err;
            if (fabs(err) > max_err_sym) max_err_sym = fabs(err);
        }
        if (estimate_offset(cw, n_fft, magsq_p, phase_p, &off)) {
            double err = off - (double)true_offset;
            valid_per++;
            sum_sq_err_per += err * err;
            if (fabs(err) > max_err_per) max_err_per = fabs(err);
        }
    }

    printf("SPECSHIFT window=%s n=%zu cases=%u max_rel_magsq_delta=%.9e\n",
           cw->name, n_fft, cases, max_rel_magsq_delta);
    printf("ESTBIAS window=%s form=symmetric n=%zu estimator=log-parabolic "
           "cases=%u valid=%u rms_err=%.9f max_abs_err=%.9f\n",
           cw->name, n_fft, cases, valid_sym,
           valid_sym ? sqrt(sum_sq_err_sym / (double)valid_sym) : 0.0, max_err_sym);
    printf("ESTBIAS window=%s form=periodic n=%zu estimator=log-parabolic "
           "cases=%u valid=%u rms_err=%.9f max_abs_err=%.9f\n",
           cw->name, n_fft, cases, valid_per,
           valid_per ? sqrt(sum_sq_err_per / (double)valid_per) : 0.0, max_err_per);
}

int main(void) {
    static const size_t sizes[] = { 1024u, 4096u };

    printf("WINCONV offsets=%u center_bin=%u\n",
           (unsigned)SPECTRAL_WINCONV_OFFSETS, (unsigned)SPECTRAL_WINCONV_CENTER_BIN);

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        size_t n = sizes[si];
        float* w_sym = (float*)malloc(n * sizeof(float));
        float* w_per = (float*)malloc(n * sizeof(float));
        if (!w_sym || !w_per) return 2;

        for (size_t wi = 0; wi < sizeof(WINDOWS) / sizeof(WINDOWS[0]); wi++) {
            const ConvWindow* cw = &WINDOWS[wi];
            cw->symmetric(w_sym, n);
            cw->periodic(w_per, n);

            for (size_t hop = n / 2; hop >= n / 8; hop /= 2) {
                report_cola(cw->name, "symmetric", w_sym, n, hop);
                report_cola(cw->name, "periodic", w_per, n, hop);
            }

            printf("GAINRATIO window=%s n=%zu sum_sym=%.9f sum_per=%.9f ratio=%.9f\n",
                   cw->name, n, window_sum(w_sym, n), window_sum(w_per, n),
                   window_sum(w_per, n) / window_sum(w_sym, n));

            /* The tone sweep is O(N^2) per case; run it at the smaller N only —
             * the convention delta scales as 1/N, so N=1024 is the worst case
             * of the two and bounds the larger sizes. */
            if (n == 1024u) sweep_forms(cw, n, w_sym, w_per);
        }
        free(w_sym);
        free(w_per);
    }
    return 0;
}
