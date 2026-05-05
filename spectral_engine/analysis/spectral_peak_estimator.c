/* spectral_peak_estimator.c - Peak frequency/amplitude estimation policies
 *
 * Estimator source basis:
 * - F. J. Harris, "On the Use of Windows for Harmonic Analysis with the
 *   Discrete Fourier Transform," Proc. IEEE, 1978.
 *   https://doi.org/10.1109/PROC.1978.10837
 * - J. O. Smith, "Quadratic Interpolation of Spectral Peaks,"
 *   Spectral Audio Signal Processing.
 *   https://www.dsprelated.com/freebooks/sasp/quadratic_interpolation_spectral_peaks.html
 * - B. G. Quinn, "Estimating Frequency by Interpolation Using Fourier
 *   Coefficients," IEEE Trans. Signal Processing, 1994.
 *   https://doi.org/10.1109/78.295186
 * - E. Jacobsen and P. Kootsookos, "Fast, Accurate Frequency Estimators,"
 *   IEEE Signal Processing Magazine, 2007.
 *   https://doi.org/10.1109/MSP.2007.361611
 * - C. Candan, "A Method for Fine Resolution Frequency Estimation from
 *   Three DFT Samples," IEEE Signal Processing Letters, 2011.
 *   https://doi.org/10.1109/LSP.2011.2136378
 * - D. C. Rife and R. R. Boorstyn, "Single Tone Parameter Estimation
 *   from Discrete-Time Observations," IEEE Trans. Information Theory, 1974.
 *   https://doi.org/10.1109/TIT.1974.1055282
 */
#include "spectral_peak_estimator.h"
#include "spectral_fast_math.h"
#include <limits.h>
#include <math.h>
#include <string.h>

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

typedef struct SpectralComplexF32 {
    float re;
    float im;
    float mag;
} SpectralComplexF32;

static int spectral_peak_finite_positive(float v) {
    return isfinite(v) && v > 0.0f;
}

static int spectral_peak_finite_nonnegative(float v) {
    return isfinite(v) && v >= 0.0f;
}

static float spectral_peak_clamp_offset(float p) {
    /* Callers reject non-finite estimator output before clamping. */
    if (p > 0.5f) return 0.5f;
    if (p < -0.5f) return -0.5f;
    return p;
}

static int spectral_peak_best_next_valid(const SpectralPeakEstimateInput* input) {
    size_t best_next = 0u;

    if (!input || input->bin > (size_t)INT_MAX || input->best_next_bin < 0) {
        return 0;
    }

    best_next = (size_t)input->best_next_bin;
    if (best_next >= input->n_freqs) {
        return 0;
    }

    /* best_next is the winner from next_frame[bin-1..bin+1]. Keep this check
     * here as well as in tracker validation because spectral_peak_estimate()
     * is a public-safe wrapper and the int subtraction for df would otherwise
     * have undefined behavior on hostile inputs such as INT_MIN. */
    if (best_next + 1u < input->bin || best_next > input->bin + 1u) {
        return 0;
    }
    return 1;
}

static void spectral_peak_sincosf(float phase, float* out_sin, float* out_cos) {
    /* Reconstructing complex DFT bins needs both sin and cos of the stored
     * analysis phase. Exact builds use libm/builtin sincos. Approx builds
     * intentionally reuse the central fast_sin() path so oscillator and peak
     * code do not carry divergent polynomial trig copies. */
#if defined(SPECTRAL_ENABLE_APPROX_TRIG) && SPECTRAL_ENABLE_APPROX_TRIG
    *out_sin = fast_sin(phase);
    *out_cos = fast_sin(phase + SPECTRAL_HALF_PI);
#elif __has_builtin(__builtin_sincosf)
    __builtin_sincosf(phase, out_sin, out_cos);
#else
    *out_sin = sinf(phase);
    *out_cos = cosf(phase);
#endif
}

const char* spectral_peak_estimator_name(SpectralPeakEstimatorType type) {
    switch (type) {
        case SPECTRAL_PEAK_ESTIMATOR_AUTO:             return "auto";
        case SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC:    return "log-parabolic";
        case SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC:    return "magnitude-parabolic";
        case SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX: return "jacobsen-complex";
        case SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX:   return "candan-complex";
        case SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND:     return "quinn-second";
        default:                                       return "unknown";
    }
}

int spectral_peak_estimator_type_supported(SpectralPeakEstimatorType type) {
    return type >= SPECTRAL_PEAK_ESTIMATOR_AUTO &&
           type < SPECTRAL_PEAK_ESTIMATOR_COUNT;
}

SpectralPeakEstimatorType spectral_peak_estimator_resolve_default(SpectralPeakEstimatorType type) {
    if (!spectral_peak_estimator_type_supported(type)) {
        return SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    }
    if (type == SPECTRAL_PEAK_ESTIMATOR_AUTO) {
        /* Conservative default: current analysis emits Hann-windowed
         * magnitude-squared spectra. A log-power quadratic fit is a local
         * three-bin peak approximation in the Smith/Harris window-analysis
         * model and has bounded contract tests for the engine's Hann path.
         * Jacobsen, Candan and Quinn are exposed explicitly because their
         * complex-DFT assumptions need separate caller intent. */
        return SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    }
    return type;
}

float spectral_peak_candan_correction_for_n_freqs(size_t n_freqs) {
    size_t n_fft = 0u;
    float alpha = 0.0f;
    float correction = 1.0f;

    if (n_freqs < 2u) return 1.0f;
    if (n_freqs - 1u > ((size_t)-1) / 2u) return 1.0f;
    n_fft = (n_freqs - 1u) * 2u;
    if (n_fft == 0u) return 1.0f;

    /* Candan's correction depends only on FFT length. In the tracker hot path
     * it is cached once in SpectralTracker; this fallback keeps standalone
     * public estimator calls correct when the field is left at zero. */
    alpha = SPECTRAL_PI / (float)n_fft;
    if (isfinite(alpha) && alpha > 0.0f) {
        correction = tanf(alpha) / alpha;
        if (!isfinite(correction) || correction <= 0.0f) correction = 1.0f;
    }
    return correction;
}

static int spectral_peak_reconstruct_complex(const SpectralPeakEstimateInput* input,
                                             size_t idx,
                                             int neighborhood_validated,
                                             SpectralComplexF32* out) {
    float magsq = 0.0f;
    float mag = 0.0f;
    float phase = 0.0f;
    float sin_phase = 0.0f;
    float cos_phase = 0.0f;

    (void)neighborhood_validated;

    if (!input || !input->magsq_row || !input->phase_row || !out) return 0;
    if (idx >= input->n_freqs) return 0;

    magsq = input->magsq_row[idx];
    phase = input->phase_row[idx];

    /* Even the tracker-validated hot path must keep this check.  The tracker
     * proves the local candidate neighborhood when called through
     * spectral_tracker_emit_segment(), but spectral_peak_estimate_validated()
     * is still a separately callable internal API.  Keeping finite/nonnegative
     * guards here prevents NaN/negative magnitudes from being converted into
     * complex coefficients when a future caller misuses the fast path. */
    if (!spectral_peak_finite_nonnegative(magsq) || !isfinite(phase)) {
        return 0;
    }

    /* Complex estimators are defined on three neighboring complex DFT
     * coefficients. The tracker stores magnitude-squared and phase separately,
     * so this reconstructs X[k] = sqrt(|X[k]|^2) * exp(j*phase[k]).
     * fast_sqrt() is exact unless SPECTRAL_ENABLE_APPROX_INV_SQRT is enabled. */
    mag = fast_sqrt(magsq);
    spectral_peak_sincosf(phase, &sin_phase, &cos_phase);
    out->mag = mag;
    out->re = mag * cos_phase;
    out->im = mag * sin_phase;
    return isfinite(out->re) && isfinite(out->im);
}


static int spectral_peak_reconstruct_triplet(const SpectralPeakEstimateInput* input,
                                             int neighborhood_validated,
                                             SpectralComplexF32* xm,
                                             SpectralComplexF32* x0,
                                             SpectralComplexF32* xp,
                                             float* out_center_mag) {
    if (!input || !xm || !x0 || !xp ||
        input->bin == 0u || input->bin + 1u >= input->n_freqs) {
        return 0;
    }
    if (!spectral_peak_reconstruct_complex(input, input->bin - 1u, neighborhood_validated, xm) ||
        !spectral_peak_reconstruct_complex(input, input->bin, neighborhood_validated, x0) ||
        !spectral_peak_reconstruct_complex(input, input->bin + 1u, neighborhood_validated, xp)) {
        return 0;
    }
    if (out_center_mag) *out_center_mag = x0->mag;
    return 1;
}

static int spectral_peak_complex_offset_jacobsen(const SpectralPeakEstimateInput* input,
                                                 int neighborhood_validated,
                                                 float* out_offset,
                                                 float* out_center_mag) {
    SpectralComplexF32 xm, x0, xp;
    float num_re = 0.0f, num_im = 0.0f;
    float den_re = 0.0f, den_im = 0.0f;
    float den_mag2 = 0.0f;
    float offset = 0.0f;

    if (!input || !out_offset || input->bin == 0u || input->bin + 1u >= input->n_freqs) return 0;
    if (!spectral_peak_reconstruct_triplet(input, neighborhood_validated,
                                           &xm, &x0, &xp, out_center_mag)) {
        return 0;
    }

    num_re = xm.re - xp.re;
    num_im = xm.im - xp.im;
    den_re = 2.0f * x0.re - xm.re - xp.re;
    den_im = 2.0f * x0.im - xm.im - xp.im;
    den_mag2 = den_re * den_re + den_im * den_im;
    if (!spectral_peak_finite_positive(den_mag2) || den_mag2 < 1.0e-30f) return 0;

    /* Jacobsen/Kootsookos:
     *   delta = Re{(X[k-1] - X[k+1]) / (2X[k] - X[k-1] - X[k+1])}
     * The denominator can vanish for flat/noisy triplets, so those cases fail
     * and the caller records fallback to log-power parabolic. */
    offset = (num_re * den_re + num_im * den_im) / den_mag2;
    if (!isfinite(offset)) return 0;

    *out_offset = spectral_peak_clamp_offset(offset);
    return 1;
}

static int spectral_peak_offset_log_parabolic(const SpectralPeakEstimateInput* input,
                                              int neighborhood_validated,
                                              float* out_offset) {
    float left = 0.0f;
    float center = 0.0f;
    float right = 0.0f;
    float raw_offset = 0.0f;
    SpectralWindowInterpMagsqFn interp = NULL;

    (void)neighborhood_validated;

    if (!input || !out_offset || input->bin == 0u || input->bin + 1u >= input->n_freqs ||
        !input->magsq_row) {
        return 0;
    }

    left = input->magsq_row[input->bin - 1u];
    center = input->curr_magsq;
    right = input->magsq_row[input->bin + 1u];

    /* Keep the finite/nonnegative guard even for the validated hot path.  A
     * custom descriptor callback is allowed here; do not pass NaN/negative
     * magnitudes into arbitrary window-specific interpolation code. */
    if (!spectral_peak_finite_nonnegative(left) ||
        !spectral_peak_finite_nonnegative(center) ||
        !spectral_peak_finite_nonnegative(right)) {
        return 0;
    }

    /* Log-power parabolic estimator (Smith/Harris basis):
     *   p = 0.5 * (log(left) - log(right)) /
     *       (log(left) - 2*log(center) + log(right))
     * where left/center/right are adjacent magnitude-squared STFT bins and
     * p is a bin offset clamped to [-0.5, 0.5]. This is the AUTO baseline for
     * Hann-windowed rows; custom descriptor callbacks must return finite p. */
    interp = input->interp_magsq ? input->interp_magsq
                                 : spectral_window_interp_magsq_parabolic;
    raw_offset = interp(left, center, right);
    if (!isfinite(raw_offset)) return 0;

    *out_offset = spectral_peak_clamp_offset(raw_offset);
    return 1;
}


static int spectral_peak_offset_mag_parabolic(const SpectralPeakEstimateInput* input,
                                              int neighborhood_validated,
                                              float* out_offset,
                                              float* out_center_mag) {
    float left = 0.0f, center = 0.0f, right = 0.0f;
    float denom = 0.0f;
    float offset = 0.0f;

    (void)neighborhood_validated;

    if (!input || !out_offset || input->bin == 0u || input->bin + 1u >= input->n_freqs ||
        !input->magsq_row) {
        return 0;
    }
    if (!spectral_peak_finite_nonnegative(input->magsq_row[input->bin - 1u]) ||
        !spectral_peak_finite_nonnegative(input->curr_magsq) ||
        !spectral_peak_finite_nonnegative(input->magsq_row[input->bin + 1u])) {
        return 0;
    }

    /* Magnitude parabolic is the same quadratic peak fit applied after sqrt.
     * It is exposed as an explicit diagnostic policy; it is not AUTO because
     * the engine's documented default is the Hann log-power contract. */
    left = fast_sqrt(input->magsq_row[input->bin - 1u]);
    center = fast_sqrt(input->curr_magsq);
    right = fast_sqrt(input->magsq_row[input->bin + 1u]);
    if (out_center_mag) *out_center_mag = center;

    denom = left - 2.0f * center + right;
    if (!isfinite(denom) || fabsf(denom) < SPECTRAL_TRACK_PARABOLIC_DENOM_EPS) {
        *out_offset = 0.0f;
        return 1;
    }

    offset = 0.5f * (left - right) / denom;
    if (!isfinite(offset)) return 0;

    *out_offset = spectral_peak_clamp_offset(offset);
    return 1;
}


static int spectral_peak_offset_candan(const SpectralPeakEstimateInput* input,
                                       int neighborhood_validated,
                                       float* out_offset,
                                       float* out_center_mag) {
    float offset = 0.0f;
    float correction = 1.0f;

    if (!spectral_peak_complex_offset_jacobsen(input, neighborhood_validated,
                                               &offset, out_center_mag)) {
        return 0;
    }
    if (!input || input->n_freqs < 2u) return 0;

    correction = input->candan_correction;
    if (!isfinite(correction) || correction <= 0.0f) {
        correction = spectral_peak_candan_correction_for_n_freqs(input->n_freqs);
    }

    /* Candan applies a finite-N correction to the Jacobsen offset. The factor
     * is precomputed per tracker because it depends only on n_fft/n_freqs. */
    offset *= correction;
    if (!isfinite(offset)) return 0;

    *out_offset = spectral_peak_clamp_offset(offset);
    return 1;
}

static float spectral_peak_quinn_tau(float x) {
    const float sqrt_6 = 2.449489742783178f;
    const float sqrt_2_over_3 = 0.816496580927726f;
    float a = 0.0f;
    float b_num = 0.0f;
    float b_den = 0.0f;
    float b_ratio = 0.0f;

    if (!isfinite(x) || x < 0.0f) return 0.0f;

    a = 3.0f * x * x + 6.0f * x + 1.0f;
    b_num = x + 1.0f - sqrt_2_over_3;
    b_den = x + 1.0f + sqrt_2_over_3;
    if (a <= 0.0f || b_num <= 0.0f || b_den <= 0.0f) return 0.0f;

    b_ratio = b_num / b_den;
    if (!spectral_peak_finite_positive(b_ratio)) return 0.0f;

    /* Quinn second estimator's tau term. Logs route through fast_peak_log()
     * so any approximation is controlled by one project-level peak-log gate. */
    return 0.25f * fast_peak_log(a) - (sqrt_6 / 24.0f) * fast_peak_log(b_ratio);
}

static int spectral_peak_offset_quinn_second(const SpectralPeakEstimateInput* input,
                                             int neighborhood_validated,
                                             float* out_offset,
                                             float* out_center_mag) {
    SpectralComplexF32 xm, x0, xp;
    float mag0 = 0.0f;
    float ap = 0.0f, am = 0.0f;
    float dp = 0.0f, dm = 0.0f;
    float den_p = 0.0f, den_m = 0.0f;
    float offset = 0.0f;

    if (!input || !out_offset || input->bin == 0u || input->bin + 1u >= input->n_freqs) return 0;
    if (!spectral_peak_reconstruct_triplet(input, neighborhood_validated,
                                           &xm, &x0, &xp, out_center_mag)) {
        return 0;
    }

    mag0 = input->curr_magsq;
    if (!spectral_peak_finite_positive(mag0)) return 0;

    ap = (xp.re * x0.re + xp.im * x0.im) / mag0;
    am = (xm.re * x0.re + xm.im * x0.im) / mag0;
    den_p = 1.0f - ap;
    den_m = 1.0f - am;
    if (!isfinite(ap) || !isfinite(am) ||
        fabsf(den_p) < 1.0e-12f || fabsf(den_m) < 1.0e-12f) {
        return 0;
    }

    dp = -ap / den_p;
    dm =  am / den_m;
    if (!isfinite(dp) || !isfinite(dm)) return 0;

    offset = 0.5f * (dp + dm) +
             spectral_peak_quinn_tau(dp * dp) -
             spectral_peak_quinn_tau(dm * dm);
    if (!isfinite(offset)) return 0;

    *out_offset = spectral_peak_clamp_offset(offset);
    return 1;
}

static int spectral_peak_estimate_offset(const SpectralPeakEstimateInput* input,
                                         SpectralPeakEstimatorType type,
                                         int neighborhood_validated,
                                         float* out_offset,
                                         float* out_center_mag,
                                         unsigned* out_flags,
                                         SpectralPeakEstimatorType* out_used_type) {
    int ok = 0;
    float offset = 0.0f;

    if (!input || !out_offset || !out_flags || !out_used_type) return 0;

    switch (type) {
        case SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC:
            ok = spectral_peak_offset_log_parabolic(input, neighborhood_validated, &offset);
            break;
        case SPECTRAL_PEAK_ESTIMATOR_MAG_PARABOLIC:
            ok = spectral_peak_offset_mag_parabolic(input, neighborhood_validated,
                                                    &offset, out_center_mag);
            break;
        case SPECTRAL_PEAK_ESTIMATOR_JACOBSEN_COMPLEX:
            ok = spectral_peak_complex_offset_jacobsen(input, neighborhood_validated,
                                                       &offset, out_center_mag);
            if (ok) *out_flags |= SPECTRAL_PEAK_ESTIMATE_COMPLEX_USED;
            break;
        case SPECTRAL_PEAK_ESTIMATOR_CANDAN_COMPLEX:
            ok = spectral_peak_offset_candan(input, neighborhood_validated,
                                             &offset, out_center_mag);
            if (ok) *out_flags |= SPECTRAL_PEAK_ESTIMATE_COMPLEX_USED;
            break;
        case SPECTRAL_PEAK_ESTIMATOR_QUINN_SECOND:
            ok = spectral_peak_offset_quinn_second(input, neighborhood_validated,
                                                   &offset, out_center_mag);
            if (ok) *out_flags |= SPECTRAL_PEAK_ESTIMATE_COMPLEX_USED;
            break;
        case SPECTRAL_PEAK_ESTIMATOR_AUTO:
        case SPECTRAL_PEAK_ESTIMATOR_COUNT:
        default:
            ok = spectral_peak_offset_log_parabolic(input, neighborhood_validated, &offset);
            type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
            break;
    }

    if (!ok && type != SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC) {
        ok = spectral_peak_offset_log_parabolic(input, neighborhood_validated, &offset);
        if (ok) {
            *out_flags |= SPECTRAL_PEAK_ESTIMATE_USED_FALLBACK;
            type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
        }
    }

    if (!ok || !isfinite(offset)) return 0;

    *out_offset = spectral_peak_clamp_offset(offset);
    *out_used_type = type;
    *out_flags |= SPECTRAL_PEAK_ESTIMATE_BIN_OFFSET_VALID;
    return 1;
}

static int spectral_peak_estimate_impl(const SpectralPeakEstimateInput* input,
                                       SpectralPeakEstimate* out,
                                       int neighborhood_validated) {
    SpectralPeakEstimatorType resolved_type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    SpectralPeakEstimatorType used_type = SPECTRAL_PEAK_ESTIMATOR_LOG_PARABOLIC;
    float offset = 0.0f;
    float center_mag = -1.0f;
    float amp = 0.0f;
    float next_amp = 0.0f;
    unsigned flags = 0u;

    if (!input || !out || !input->magsq_row || input->n_freqs < 3u ||
        input->bin == 0u || input->bin >= input->n_freqs - 1u) {
        return 0;
    }

    memset(out, 0, sizeof(*out));
    out->requested_type = input->type;

    if (!spectral_peak_best_next_valid(input)) return 0;

    /* The validated entry point means the tracker has already accepted the
     * local magnitude neighborhood and next-frame triplet.  It does NOT prove
     * that caller-provided frequency-step fields, hop reciprocal, or stored
     * magnitudes are finite.  Keep these scalar-contract checks unconditional
     * so the hot path remains safe if a future caller bypasses the tracker. */
    if (!spectral_peak_finite_nonnegative(input->curr_magsq) ||
        !spectral_peak_finite_nonnegative(input->next_max_magsq) ||
        !isfinite(input->freq_step_omega) ||
        !isfinite(input->freq_step_df) ||
        !isfinite(input->inv_hop)) {
        return 0;
    }

    resolved_type = spectral_peak_estimator_resolve_default(input->type);
    if (!spectral_peak_estimate_offset(input, resolved_type, neighborhood_validated,
                                       &offset, &center_mag, &flags, &used_type)) {
        return 0;
    }

    amp = (center_mag >= 0.0f && isfinite(center_mag))
        ? center_mag
        : fast_sqrt(input->curr_magsq);
    next_amp = fast_sqrt(input->next_max_magsq);
    if (!isfinite(amp) || !isfinite(next_amp)) return 0;

    out->bin_offset = offset;
    out->amp = amp;
    out->next_amp = next_amp;
    out->da = (next_amp - amp) * input->inv_hop;
    out->omega = ((float)input->bin + offset) * input->freq_step_omega;
    out->df = (float)(input->best_next_bin - (int)input->bin) * input->freq_step_df;
    out->flags = flags | SPECTRAL_PEAK_ESTIMATE_AMP_VALID |
                 SPECTRAL_PEAK_ESTIMATE_OMEGA_VALID |
                 SPECTRAL_PEAK_ESTIMATE_DF_VALID;
    out->used_type = used_type;

    if (!isfinite(out->da) || !isfinite(out->omega) || !isfinite(out->df)) {
        return 0;
    }

    return 1;
}


int spectral_peak_estimate(const SpectralPeakEstimateInput* input,
                           SpectralPeakEstimate* out) {
    return spectral_peak_estimate_impl(input, out, 0);
}

int spectral_peak_estimate_validated(const SpectralPeakEstimateInput* input,
                                     SpectralPeakEstimate* out) {
    return spectral_peak_estimate_impl(input, out, 1);
}
