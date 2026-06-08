/* spectral_osc_bandlimited.c - Anti-aliased ("musical") oscillator quality modes.
 *
 * Implements the three opt-in renderers declared in spectral_osc_bandlimited.h.
 * Default quality is NAIVE, which never reaches this file (timbre_synth_segment
 * short-circuits), so the default build output stays bit-identical to the
 * point-sampled path.  CPU float path only.
 *
 * DSP background.  Resynthesis renders each detected partial as the chosen timbre
 * AT that partial's frequency.  The naive waveforms are point-sampled, so any
 * harmonic a hard edge or sharp cusp generates above Nyquist folds back as
 * inharmonic aliasing.  The three modes attack that differently:
 *
 *   POLYBLEP   - add a PolyBLEP step residual at each amplitude discontinuity and
 *                a polyBLAMP slope residual at each slope discontinuity, sized by
 *                the instantaneous frequency.  Cheap, per-sample, no allocation.
 *   ADDITIVE   - rebuild the waveform from its Fourier series, capped at the
 *                highest harmonic that still sits below Nyquist (Chebyshev
 *                recurrence so each sample costs N multiply-adds, no per-harmonic
 *                trig).  Exactly band-limited by construction.
 *   OVERSAMPLE - render the naive waveform OS x, convolve with a windowed-sinc
 *                low-pass, decimate.  Universal: the only mode that tames the
 *                cusp/staircase timbres (asin, quantized) that have no closed-form
 *                BLEP or compact Fourier series.
 *
 * Phase/amplitude bookkeeping mirrors synth_segment_scalar() in oscillator.c so
 * these are drop-in quality swaps: same cubic phase helper, same per-sample
 * amplitude ramp, same Hann fade envelope. */
#include "spectral_osc_bandlimited.h"
#include "spectral_synth_internal.h" /* SegmentLoopParams, spectral_segment_*_f32 */
#include "spectral_envelope.h"       /* FadeParams, fade_params_init, fade_envelope */
#include "spectral_osc_formulas.h"   /* naive waveforms, spectral_normalize_phase, consts */
#include "spectral_config.h"         /* SpectralTimbre, SPECTRAL_FADE_SAMPLES_ACTIVE */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- Process-global quality state (mirrors osc_set_dispatch in oscillator.c) ---- */

static SpectralOscQuality g_osc_quality = SPECTRAL_OSC_QUALITY_NAIVE;

void osc_set_quality(SpectralOscQuality quality) { g_osc_quality = quality; }
SpectralOscQuality osc_get_quality(void) { return g_osc_quality; }

const char* osc_quality_name(SpectralOscQuality quality) {
    switch (quality) {
    case SPECTRAL_OSC_QUALITY_NAIVE:      return "naive";
    case SPECTRAL_OSC_QUALITY_POLYBLEP:   return "polyblep";
    case SPECTRAL_OSC_QUALITY_ADDITIVE:   return "additive";
    case SPECTRAL_OSC_QUALITY_OVERSAMPLE: return "oversample";
    default:                              return "naive";
    }
}

int osc_quality_parse(const char* text, SpectralOscQuality* out) {
    if (!text || !out) return 0;

    /* Single-digit integer 0..3. */
    if (text[0] >= '0' && text[0] <= '3' && text[1] == '\0') {
        *out = (SpectralOscQuality)(text[0] - '0');
        return 1;
    }
    if (strcmp(text, "naive") == 0)      { *out = SPECTRAL_OSC_QUALITY_NAIVE;      return 1; }
    if (strcmp(text, "polyblep") == 0 ||
        strcmp(text, "blep") == 0)       { *out = SPECTRAL_OSC_QUALITY_POLYBLEP;   return 1; }
    if (strcmp(text, "additive") == 0 ||
        strcmp(text, "add") == 0)        { *out = SPECTRAL_OSC_QUALITY_ADDITIVE;   return 1; }
    if (strcmp(text, "oversample") == 0 ||
        strcmp(text, "os") == 0)         { *out = SPECTRAL_OSC_QUALITY_OVERSAMPLE; return 1; }
    return 0;
}

/* ---- Shared helpers ---- */

/* Fractional part in [0,1); used to rotate the cyclic phase so a given edge
 * lands at the 0/1 wrap that poly_blep/poly_blamp expect. */
static inline float frac01(float x) { return x - floorf(x); }

/* Instantaneous normalized frequency |dphase/dj| / 2pi, in cycles/sample.
 * Phase is theta(j) = phase0 + alpha*j + c2*j^2 + c3*j^3, so the angular
 * frequency is alpha + 2*c2*j + 3*c3*j^2 rad/sample.  Clamped to [1e-6, 0.5]:
 * 0.5 is Nyquist (one half-cycle/sample), and the small floor avoids a zero-width
 * BLEP transition / a divide-by-zero in the harmonic cap. */
static inline float osc_bl_norm_freq(const SegmentLoopParams* lp, float jf) {
    float w = lp->alpha + jf * (2.0f * lp->c2 + 3.0f * lp->c3 * jf);
    float dt = fabsf(w) * SPECTRAL_INV_TWO_PI;
    /* `!(dt > 1e-6f)` (not `dt < 1e-6f`) also catches NaN — a NaN chirp slope would
     * otherwise slip past both comparisons and poison floorf(0.5/dt) in additive.
     * Identical to the old clamp for every finite dt. */
    if (!(dt > 1e-6f)) dt = 1e-6f;
    else if (dt > 0.5f) dt = 0.5f;
    return dt;
}

/* PolyBLEP step residual (2nd-order).  Corrects a unit-height trivial step whose
 * discontinuity sits at t==0 (==1).  Add (jump/2)*poly_blep(...) to the naive
 * sample, where jump = value_after - value_before.  Zero outside +/- dt of the
 * edge. */
static inline float poly_blep(float t, float dt) {
    if (t < dt) {
        float x = t / dt;
        return 2.0f * x - x * x - 1.0f;
    }
    if (t > 1.0f - dt) {
        float x = (t - 1.0f) / dt;
        return x * x + 2.0f * x + 1.0f;
    }
    return 0.0f;
}

/* PolyBLAMP slope residual (integral of poly_blep).  Corrects a slope
 * discontinuity (a corner) at t==0 (==1); scale by (slope_change * dt) at the
 * call site.  Returns a small positive bump (peak 1/3) at the corner. */
static inline float poly_blamp(float t, float dt) {
    if (t < dt) {
        float x = t / dt - 1.0f;
        return -(1.0f / 3.0f) * x * x * x;
    }
    if (t > 1.0f - dt) {
        float x = (t - 1.0f) / dt + 1.0f;
        return (1.0f / 3.0f) * x * x * x;
    }
    return 0.0f;
}

/* ---- Mode 1: PolyBLEP / polyBLAMP edge correction ---- */

/* saw/square/triangle/pwm get edge or slope correction; sine/parabola are
 * continuous enough to render naive here; asin/quantized have no BLEP form so we
 * return 0 and let the caller fall back to the naive scalar path. */
static int osc_bl_polyblep(float* dst, const SegmentLoopParams* lp, SpectralTimbre timbre) {
    if (timbre == TIMBRE_ASIN || timbre == TIMBRE_QUANTIZED) return 0;

    const size_t len = lp->length;
    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase, alpha = lp->alpha, c2 = lp->c2, c3 = lp->c3;
    const float amp0 = lp->amp, d_amp = lp->d_amp, width = lp->width;

    for (size_t j = 0; j < len; j++) {
        const float jf = (float)j;
        const float p = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, jf);
        const float rads = spectral_normalize_phase(p);     /* [-pi, pi) */
        const float t = rads * SPECTRAL_INV_TWO_PI + 0.5f;   /* [0, 1)     */
        const float dt = osc_bl_norm_freq(lp, jf);
        float wave;

        switch (timbre) {
        case TIMBRE_SAW:
            /* naive 1-2t: +2 jump at the t=0 wrap -> +poly_blep(t). */
            wave = rads * -SPECTRAL_INV_PI + poly_blep(t, dt);
            break;
        case TIMBRE_SQUARE:
            /* naive (t>0.5)?1:-1: -2 jump at t=0, +2 jump at t=0.5.  Use >= 0 so
             * the exact edge-center sample (rads==0) takes the post-jump value the
             * BLEP residual is built around (residual at tau==0 is -1); with the
             * golden '> 0' it would emit a spurious -2 at that one sample. */
            wave = ((rads >= 0.0f) ? 1.0f : -1.0f)
                 - poly_blep(t, dt)
                 + poly_blep(frac01(t + 0.5f), dt);
            break;
        case TIMBRE_TRIANGLE:
            /* naive 1-2|rads|/pi: corners at t=0 (trough) and t=0.5 (peak).
             * Slope is +/-4 per cycle, so the polyBLAMP scale is 4*dt. */
            wave = ((1.0f - fabsf(rads) * SPECTRAL_INV_PI) * 2.0f - 1.0f)
                 + 4.0f * dt * (poly_blamp(t, dt) - poly_blamp(frac01(t + 0.5f), dt));
            break;
        case TIMBRE_PWM:
            /* naive (t<width)?1:-1 for a real duty: +2 jump at t=0, -2 at t=width.
             * Degenerate duty (<=0 or >=1) is constant -> no edges, render naive. */
            if (width > 0.0f && width < 1.0f) {
                wave = (t < width ? 1.0f : -1.0f)
                     + poly_blep(t, dt)
                     - poly_blep(frac01(t - width), dt);
            } else {
                wave = spectral_osc_pwm(rads, width);
            }
            break;
        case TIMBRE_SINE:
            wave = spectral_fast_sin_inline(rads);
            break;
        case TIMBRE_PARABOLA:
            wave = spectral_osc_parabola(rads, width);
            break;
        default:
            return 0;
        }

        const float amp = spectral_segment_amp_at_f32(amp0, d_amp, jf) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
    return 1;
}

/* ---- Mode 2: Nyquist-capped additive (Fourier) synthesis ---- */

/* Hard ceiling on the harmonic count, so a very low partial does not spin the
 * inner loop hundreds of thousands of times.  512 harmonics is already a clean
 * band-limited shape; the Nyquist cap is usually far below this. */
#define OSC_BL_MAX_HARMONICS 512

/* saw/square/triangle/parabola/sine have compact Fourier series; pwm (variable
 * duty), asin (cusp), quantized (staircase) do not -> return 0 for fallback. */
static int osc_bl_additive(float* dst, const SegmentLoopParams* lp, SpectralTimbre timbre) {
    if (timbre == TIMBRE_PWM || timbre == TIMBRE_ASIN || timbre == TIMBRE_QUANTIZED) return 0;

    const size_t len = lp->length;
    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase, alpha = lp->alpha, c2 = lp->c2, c3 = lp->c3;
    const float amp0 = lp->amp, d_amp = lp->d_amp;

    for (size_t j = 0; j < len; j++) {
        const float jf = (float)j;
        const float p = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, jf);
        const float theta = spectral_normalize_phase(p);
        const float dt = osc_bl_norm_freq(lp, jf);

        int n_harm = (int)floorf(0.5f / dt);   /* highest harmonic below Nyquist */
        if (n_harm < 1) n_harm = 1;
        else if (n_harm > OSC_BL_MAX_HARMONICS) n_harm = OSC_BL_MAX_HARMONICS;

        float wave;
        if (timbre == TIMBRE_SINE) {
            wave = sinf(theta);
        } else {
            /* Chebyshev recurrence for sin(k*theta) and cos(k*theta):
             *   s_{k+1} = 2*cos(theta)*s_k - s_{k-1},  s_0=0, s_1=sin
             *   c_{k+1} = 2*cos(theta)*c_k - c_{k-1},  c_0=1, c_1=cos
             * One cos/sin for the whole partial; each harmonic is then add-only. */
            const float ct = cosf(theta), st = sinf(theta);
            float s_prev = 0.0f, s_k = st;
            float c_prev = 1.0f, c_k = ct;
            float sign = -1.0f;            /* (-1)^k starting at k=1 */
            float series = 0.0f;

            for (int k = 1; k <= n_harm; k++) {
                const float fk = (float)k;
                switch (timbre) {
                case TIMBRE_SAW:      series += (sign / fk) * s_k; break;         /* (2/pi) sum ((-1)^k/k) sin(k.) */
                case TIMBRE_SQUARE:   if (k & 1) series += (1.0f / fk) * s_k; break;       /* (4/pi) sum_odd (1/k) sin(k.) */
                case TIMBRE_TRIANGLE: if (k & 1) series += (1.0f / (fk * fk)) * c_k; break;/* (8/pi^2) sum_odd (1/k^2) cos(k.) */
                case TIMBRE_PARABOLA: series += (sign / (fk * fk)) * c_k; break;  /* 2/3 - (4/pi^2) sum ((-1)^k/k^2) cos(k.) */
                default: break;
                }
                const float s_next = 2.0f * ct * s_k - s_prev;
                const float c_next = 2.0f * ct * c_k - c_prev;
                s_prev = s_k; s_k = s_next;
                c_prev = c_k; c_k = c_next;
                sign = -sign;
            }

            switch (timbre) {
            case TIMBRE_SAW:      wave = (2.0f * SPECTRAL_INV_PI) * series; break;
            case TIMBRE_SQUARE:   wave = (4.0f * SPECTRAL_INV_PI) * series; break;
            case TIMBRE_TRIANGLE: wave = (8.0f * SPECTRAL_INV_PI_SQ) * series; break;
            case TIMBRE_PARABOLA: wave = (2.0f / 3.0f) - (4.0f * SPECTRAL_INV_PI_SQ) * series; break;
            default:              wave = 0.0f; break;
            }
        }

        const float amp = spectral_segment_amp_at_f32(amp0, d_amp, jf) * fade_envelope(j, &fp, len);
        dst[j] += amp * wave;
    }
    return 1;
}

/* ---- Mode 3: oversample + windowed-sinc decimation (universal) ---- */

#define OSC_BL_OVERSAMPLE   4    /* OS factor */
#define OSC_BL_FIR_TAPS     8    /* sinc lobes each side, in original samples */
#define OSC_BL_FIR_HALF     (OSC_BL_OVERSAMPLE * OSC_BL_FIR_TAPS)        /* H */
#define OSC_BL_FIR_LEN      (2 * OSC_BL_FIR_HALF + 1)                    /* L = 65 */

/* Above this segment length the OS buffer (OS*len floats) gets impractical; bail
 * to the naive path instead of a multi-hundred-MB allocation. */
#define OSC_BL_OS_MAX_LEN   ((size_t)4 * 1024 * 1024)

/* Naive point-sampled waveform for every timbre (the thing we oversample). */
static inline float osc_bl_naive(SpectralTimbre timbre, float rads, float width) {
    switch (timbre) {
    case TIMBRE_SINE:      return spectral_osc_sine(rads, width);
    case TIMBRE_SAW:       return spectral_osc_saw(rads, width);
    case TIMBRE_SQUARE:    return spectral_osc_square(rads, width);
    case TIMBRE_TRIANGLE:  return spectral_osc_triangle(rads, width);
    case TIMBRE_ASIN:      return spectral_osc_asin(rads, width);
    case TIMBRE_PARABOLA:  return spectral_osc_parabola(rads, width);
    case TIMBRE_QUANTIZED: return spectral_osc_quantized(rads, width);
    case TIMBRE_PWM:       return spectral_osc_pwm(rads, width);
    default:               return spectral_osc_sine(rads, width);
    }
}

/* Build a zero-phase windowed-sinc low-pass with cutoff at the original Nyquist
 * (= OS-rate Nyquist / OS).  h[n] = sinc((n-H)/OS) * Hann(n), DC-normalized so the
 * filter has unity passband gain.  The filter depends only on compile-time
 * constants, so it is identical for every call -> built once (see osc_bl_fir). */
static void osc_bl_build_fir(float* h) {
    float sum = 0.0f;
    for (int n = 0; n < OSC_BL_FIR_LEN; n++) {
        const float x = (float)(n - OSC_BL_FIR_HALF) / (float)OSC_BL_OVERSAMPLE;
        float sinc;
        if (x == 0.0f) {
            sinc = 1.0f;
        } else {
            const float px = SPECTRAL_PI * x;
            sinc = sinf(px) / px;
        }
        const float hann = 0.5f - 0.5f * cosf(SPECTRAL_TWO_PI * (float)n / (float)(OSC_BL_FIR_LEN - 1));
        h[n] = sinc * hann;
        sum += h[n];
    }
    const float inv = 1.0f / sum;
    for (int n = 0; n < OSC_BL_FIR_LEN; n++) h[n] *= inv;
}

/* The decimation FIR is constant; build it once per thread instead of recomputing
 * 65 sinf/cosf pairs on every segment.  Thread-local so the OMP segment-parallel
 * synth loop stays lock-free (each worker builds its own copy on first use). */
static SPECTRAL_THREAD_LOCAL float g_osc_bl_fir[OSC_BL_FIR_LEN];
static SPECTRAL_THREAD_LOCAL int   g_osc_bl_fir_ready = 0;

static const float* osc_bl_fir(void) {
    if (!g_osc_bl_fir_ready) {
        osc_bl_build_fir(g_osc_bl_fir);
        g_osc_bl_fir_ready = 1;
    }
    return g_osc_bl_fir;
}

/* Reusable per-thread oversample scratch, grown monotonically to the largest
 * segment this thread renders.  Replaces a malloc/free on every segment; the
 * buffer lives for the thread's lifetime (released by the OS at process exit,
 * matching the gpu_tile_cache pattern in spectral_synth_internal.c).  Thread-local
 * for the same lock-free reason as the FIR above. */
static SPECTRAL_THREAD_LOCAL float*  g_osc_bl_os_buf = NULL;
static SPECTRAL_THREAD_LOCAL size_t  g_osc_bl_os_cap = 0;

static float* osc_bl_os_scratch(size_t need) {
    if (g_osc_bl_os_cap < need) {
        float* p = (float*)realloc(g_osc_bl_os_buf, need * sizeof(float));
        if (!p) return NULL;             /* old buffer still valid; caller bails to naive */
        g_osc_bl_os_buf = p;
        g_osc_bl_os_cap = need;
    }
    return g_osc_bl_os_buf;
}

static int osc_bl_oversample(float* dst, const SegmentLoopParams* lp, SpectralTimbre timbre) {
    const size_t len = lp->length;
    if (len > OSC_BL_OS_MAX_LEN) return 0;   /* too big -> fall back to naive */

    const size_t os_len = len * (size_t)OSC_BL_OVERSAMPLE;
    float* os_buf = osc_bl_os_scratch(os_len);
    if (!os_buf) return 0;                   /* alloc failed -> fall back to naive */

    const float phase0 = lp->phase, alpha = lp->alpha, c2 = lp->c2, c3 = lp->c3;
    const float amp0 = lp->amp, d_amp = lp->d_amp, width = lp->width;

    /* Render the naive waveform OS x.  Oversampled index i maps to fractional
     * original-sample position i/OS; phase uses the same cubic helper. */
    for (size_t i = 0; i < os_len; i++) {
        const float pos = (float)i / (float)OSC_BL_OVERSAMPLE;
        const float p = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, pos);
        const float rads = spectral_normalize_phase(p);
        os_buf[i] = osc_bl_naive(timbre, rads, width);
    }

    const float* h = osc_bl_fir();
    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);

    /* Decimate: each output sample j is the FIR centered on os index OS*j (an
     * exact integer alignment, so zero fractional delay).  The window is zero-phase
     * symmetric (h[n] == h[L-1-n]), so each tap pair ±m shares weight h[H-m]; that
     * halves the multiplies.  Samples whose whole [center-H, center+H] window sits
     * inside the buffer take a branch-free interior loop; only the OS*FIR_TAPS edge
     * samples need the clamped form (Hann fade masks any residual edge bias).
     * Folding reorders the tap sum and uses the symmetric coefficient for both
     * lobes; vs the per-tap ascending reference this perturbs the output by max
     * 2.98e-7 / RMS -157 dBFS (~1 ULP, -135 dB below signal) — inaudible, and this
     * is an opt-in audition mode, not a golden contract. */
    const long   H      = OSC_BL_FIR_HALF;
    const long   oslenl = (long)os_len;
    const float  h_ctr  = h[OSC_BL_FIR_HALF];
    for (size_t j = 0; j < len; j++) {
        const long center = (long)(j * (size_t)OSC_BL_OVERSAMPLE);
        float acc;
        if (center - H >= 0 && center + H < oslenl) {
            const float* c = os_buf + center;   /* whole window in-bounds */
            acc = h_ctr * c[0];
            for (long m = 1; m <= H; m++) {
                acc += h[OSC_BL_FIR_HALF - m] * (c[-m] + c[m]);
            }
        } else {
            acc = h_ctr * os_buf[center < 0 ? 0 : (center >= oslenl ? oslenl - 1 : center)];
            for (long m = 1; m <= H; m++) {
                long lo = center - m, hi = center + m;
                if (lo < 0) lo = 0; else if (lo >= oslenl) lo = oslenl - 1;
                if (hi < 0) hi = 0; else if (hi >= oslenl) hi = oslenl - 1;
                acc += h[OSC_BL_FIR_HALF - m] * (os_buf[lo] + os_buf[hi]);
            }
        }
        const float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope(j, &fp, len);
        dst[j] += amp * acc;
    }

    return 1;
}

/* ---- Public dispatch ---- */

int osc_bandlimited_synth_segment(float* dst, const SegmentLoopParams* lp,
                                  SpectralTimbre timbre, SpectralOscQuality quality) {
    if (lp->length == 0) return 1;   /* nothing to render; treat as handled */

    switch (quality) {
    case SPECTRAL_OSC_QUALITY_POLYBLEP:   return osc_bl_polyblep(dst, lp, timbre);
    case SPECTRAL_OSC_QUALITY_ADDITIVE:   return osc_bl_additive(dst, lp, timbre);
    case SPECTRAL_OSC_QUALITY_OVERSAMPLE: return osc_bl_oversample(dst, lp, timbre);
    case SPECTRAL_OSC_QUALITY_NAIVE:
    default:
        return 0;   /* caller renders the naive scalar path */
    }
}
