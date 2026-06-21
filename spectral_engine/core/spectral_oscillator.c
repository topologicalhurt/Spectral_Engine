/* spectral_oscillator.c - host per-segment oscillator: timbre dispatch + scalar/Q15 sustain loops */
#include "spectral_oscillator.h"
#include "spectral_oscillator_dispatch.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"
#include "spectral_osc_bandlimited.h"
#include "spectral_osc_q15.h"
#include "spectral_phase_nco.h"
#include "spectral_utils.h"
#include <math.h>

/* Pin the Q15 waveform contract: this TU's scalar-Q15 dispatch
 * (synth_segment_q15 -> spectral_osc_q15_<timbre>) was validated against v1.
 * A contract bump (SPECTRAL_OSC_Q15_VERSION) fails this build until the Q15
 * path here is re-validated and the pin updated. */
_Static_assert(SPECTRAL_OSC_Q15_VERSION == 1,
               "spectral_osc_q15.h contract changed; re-validate spectral_oscillator.c Q15 path");

/* SIMD is the default CPU execution strategy: the host/embedded osc_simd_segment_*
 * paths are written op-for-op against the scalar oscillator and measure ~1.8x
 * faster on the vectorizable timbres (saw/square/triangle/parabola), ~1.1x on
 * sine (sinf-bound).  Output is <=1 ULP from scalar (FMA-contraction only under
 * -ffast-math), not byte-identical; the scalar reference stays reachable via the
 * CLI --scalar flag (spectral_osc_set_dispatch(OSC_DISPATCH_ALL_SCALAR)).  Timbres without
 * a SIMD path (e.g. asin) fall through to scalar via spectral_osc_simd_available(). */
static OscDispatchWord g_osc_dispatch = OSC_DISPATCH_ALL_SIMD;

void spectral_osc_set_dispatch(OscDispatchWord dispatch) { g_osc_dispatch = dispatch; }

/* Opt-in Q15 compute domain. Float is the default — this mask is 0 unless
 * a caller explicitly enables a path, so the default build moves no bytes. */
static uint16_t g_osc_q15_enable = 0;
/* The desktop Q15 compute path (and its ~8 KB sine LUT) is a host-only
 * optimisation: real embedded firmware synthesises Q15 in spectral_synth_arm32.c
 * (integer NCO) and never reaches this dispatch, so the LUT and the scalar/SIMD
 * Q15 segment helpers are pure .bss/.text waste there.  Compile them out of
 * embedded targets.  The mask, its setter, and spectral_osc_q15_available() stay in
 * every build because the CLI pipeline (run_synthesis) references them
 * unconditionally — only the LUT-bearing body is conditional. */
#if !SPECTRAL_EMBEDDED
static q15_t g_osc_q15_sine_lut[SPECTRAL_OSC_LUT_SIZE + 1];
static int g_osc_q15_sine_lut_ready = 0;
#endif

int spectral_osc_q15_available(SpectralTimbre timbre) {
    switch (timbre) {
    case TIMBRE_SINE: case TIMBRE_SAW: case TIMBRE_SQUARE:
    case TIMBRE_TRIANGLE: case TIMBRE_PARABOLA:
        return 1;
    default:
        return 0;   /* asin/quantized/pwm: no Q15 evaluator (spectral_osc_q15.h) */
    }
}

void spectral_osc_set_q15_enable(uint16_t mask) {
    g_osc_q15_enable = mask;
#if !SPECTRAL_EMBEDDED
    /* Build the sine table once here (single-threaded setup call, like
     * spectral_osc_set_dispatch) so the per-segment OMP render loop only reads it —
     * no lazy-init race on the hot path. */
    if ((mask & OSC_Q15_BIT(TIMBRE_SINE)) && !g_osc_q15_sine_lut_ready) {
        spectral_osc_q15_init_sine_lut(g_osc_q15_sine_lut);
        g_osc_q15_sine_lut_ready = 1;
    }
#endif
}

float timbre_oscillator(float p, float a, SpectralTimbre timbre, float width) {
    /* Single-sample API — not hot — uses the L1 kernel directly. Out-of-range
     * timbre folds to sine in its default case, matching the old clamp. */
    return a * spectral_osc_eval(p, (int)timbre, width);
}

typedef float (*TimbreFunc)(float rads, float width);

/* Host-hoisted specialization of the spectral_osc_eval() map: resolve the timbre
 * to its fixed L0 waveform ONCE per segment so the inner sample loop calls a
 * single waveform (and the compiler can inline/vectorize it) rather than running
 * the loop-invariant 8-way dispatch per sample. Same map as the L1 switch via the
 * shared SPECTRAL_OSC_TIMBRE_LIST X-macro, so the two can never drift. */
static inline TimbreFunc osc_waveform_fn(SpectralTimbre timbre) {
    switch (timbre) {
#define SPECTRAL_OSC_FN_CASE(T, FN) case T: return FN;
        SPECTRAL_OSC_TIMBRE_LIST(SPECTRAL_OSC_FN_CASE)
#undef SPECTRAL_OSC_FN_CASE
        default: return spectral_osc_sine;
    }
}

static void synth_segment_scalar(
    float* dst, size_t len, float phase0, float alpha, float c2, float c3,
    float amp0, float d_amp, float width, const FadeParams* fp, TimbreFunc osc_fn
) {
    size_t fade_in_end = fp->fade_len;
    size_t fade_out_start = fp->fade_out_start;

    /* Fade-in region */
    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float p = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_in(j, fp->inv_fade);
        dst[j] += amp * wave;
    }

    /* Sustain region (envelope = 1.0, no branching) */
    for (size_t j = fade_in_end; j < fade_out_start && j < len; j++) {
        float p = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j);
        dst[j] += amp * wave;
    }

    /* Fade-out region */
    for (size_t j = (fade_out_start > fade_in_end ? fade_out_start : fade_in_end); j < len; j++) {
        float p = spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j);
        float rads = phase_to_rads(p);
        float wave = osc_fn(rads, width);
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_out(j, len, fp->inv_fade);
        dst[j] += amp * wave;
    }
}

#if !SPECTRAL_EMBEDDED
/* Scalar Q15-compute sustain path (QTYPE_DOMAIN_PLAN.md). Op-for-op
 * the float synth_segment_scalar above, except (a) the Q15 phase index is produced
 * by the integer-NCO cubic forward-difference accumulator (spectral_phase_nco.h) —
 * 3 integer adds per sample, no float and no per-sample float->Q15 conversion on the
 * phase path — and (b) the WAVEFORM is evaluated in Q15, then converted back to float
 * for the (unchanged, float) amplitude + fade envelope and accumulation. The integer
 * phase sits at/below the Q15-eval floor (pinned by test_phase_nco_precision: tighter
 * than float-cubic, square bit-identical), so it drops in for the old float-cubic phase +
 * spectral_osc_q15_phase_from_rads pipeline with no audible change.
 *
 * The NCO is stepped exactly ONCE per sample. The three fade regions below together
 * cover [0,len) contiguously and in sample order in every case (normal, overlapping
 * fades, or len shorter than a fade), so stepping the (stateful) forward-difference
 * chain once per loop body reproduces the exact per-sample phase the float path
 * recomputed from j — the step order IS the sample order. */
static void synth_segment_q15(
    float* dst, size_t len, float phase0, float alpha, float c2, float c3,
    float amp0, float d_amp, const FadeParams* fp, SpectralTimbre timbre,
    const q15_t* sine_lut
) {
    size_t fade_in_end = fp->fade_len;
    size_t fade_out_start = fp->fade_out_start;

    SpectralPhaseNco nco;
    spectral_phase_nco_init(&nco, phase0, alpha, c2, c3);

    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float wave = Q15_TO_FLOAT(spectral_osc_q15_eval(spectral_phase_nco_step(&nco), timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_in(j, fp->inv_fade);
        dst[j] += amp * wave;
    }

    for (size_t j = fade_in_end; j < fade_out_start && j < len; j++) {
        float wave = Q15_TO_FLOAT(spectral_osc_q15_eval(spectral_phase_nco_step(&nco), timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j);
        dst[j] += amp * wave;
    }

    for (size_t j = (fade_out_start > fade_in_end ? fade_out_start : fade_in_end); j < len; j++) {
        float wave = Q15_TO_FLOAT(spectral_osc_q15_eval(spectral_phase_nco_step(&nco), timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_out(j, len, fp->inv_fade);
        dst[j] += amp * wave;
    }
}
#endif /* !SPECTRAL_EMBEDDED */

void timbre_synth_segment(float* __restrict__ dst, const struct SegmentLoopParams* lp, SpectralTimbre timbre) {
    if ((unsigned int)timbre >= TIMBRE_COUNT) {
        char resolution[384] = {0};
        SpectralResolutionContext context = {
            .event = SPECTRAL_RESOLUTION_EVENT_FALLBACK,
            .backend = SPECTRAL_RESOLUTION_BACKEND_CPU,
            .scope = SPECTRAL_RESOLUTION_SCOPE_TIMBRE,
            .requested_label = timbre_name(timbre),
            .requested_id = (int)timbre,
            .effective_label = timbre_name(TIMBRE_SINE),
            .effective_id = (int)TIMBRE_SINE,
            .max_supported_id = (int)(TIMBRE_COUNT - 1),
            .mode = spectral_exec_mode_name(),
            .reason = SPECTRAL_RESOLUTION_REASON_INVALID_TIMBRE_ID
        };
        spectral_format_resolution_context(resolution, sizeof(resolution), &context);
        SPECTRAL_WARN_ONCE(TIMBRE_COUNT, "%s", resolution);
        timbre = TIMBRE_SINE;
    }
    
    const size_t len = lp->length;
    if (len == 0) return;

    /* Band-limited ("musical") quality modes (PolyBLEP / additive / oversample).
     * Default is NAIVE, which returns 0 here so the point-sampled path below runs
     * unchanged (bit-identical default build).  A non-naive mode renders the
     * segment itself and returns 1, except for timbre/mode pairs it does not
     * support (e.g. asin under PolyBLEP), which return 0 to fall through to naive. */
    SpectralOscQuality osc_quality = spectral_osc_get_quality();
    if (osc_quality != SPECTRAL_OSC_QUALITY_NAIVE &&
        spectral_osc_bandlimited_synth_segment(dst, lp, timbre, osc_quality)) {
        return;
    }

    /* Opt-in Q15 compute domain: a domain choice orthogonal to the
     * float scalar/SIMD dispatch below. Engaged only when this timbre's bit is
     * set AND it has a Q15 path; otherwise fall through to the float dispatch.
     * The point-sampled (NAIVE) path only — band-limited quality already
     * returned above. Host-only (the LUT and Q15 segment helpers above are
     * compiled out of embedded, which synthesises Q15 via spectral_synth_arm32.c). */
#if !SPECTRAL_EMBEDDED
    if ((g_osc_q15_enable & OSC_Q15_BIT(timbre)) && spectral_osc_q15_available(timbre)) {
#if defined(OSC_SIMD_GENERIC)
        /* Packed 8xQ15 SIMD twin: the Q15 compute domain composes with the
         * float scalar/SIMD axis. When this timbre's dispatch resolves to SIMD and
         * it is one of the algebraic Q15 timbres, render it 8-wide (sine's serial
         * LUT loses, so it stays on the scalar Q15 path below). --scalar honoured. */
        OscDispatchMode q15_mode = OSC_GET_MODE(g_osc_dispatch, timbre);
        if (q15_mode == OSC_MODE_FALLBACK) {
            q15_mode = spectral_osc_simd_q15_available(timbre) ? OSC_MODE_CPU_SIMD : OSC_MODE_CPU_SCALAR;
        }
        if (q15_mode == OSC_MODE_CPU_SIMD && spectral_osc_simd_q15_available(timbre)) {
            spectral_osc_simd_q15_segment(dst, lp, timbre, g_osc_q15_sine_lut);
            return;
        }
#endif
        FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
        synth_segment_q15(
            dst, len, lp->phase, lp->alpha, lp->c2, lp->c3,
            lp->amp, lp->d_amp, &fp, timbre, g_osc_q15_sine_lut
        );
        return;
    }
#endif /* !SPECTRAL_EMBEDDED */

    OscDispatchMode dispatch_mode = OSC_GET_MODE(g_osc_dispatch, timbre);

    if (dispatch_mode == OSC_MODE_FALLBACK) {
        dispatch_mode = spectral_osc_simd_available(timbre) ? OSC_MODE_CPU_SIMD : OSC_MODE_CPU_SCALAR;
    }

    /* Cubic phase (SPECTRAL_PRECISE_PHASE) needs no special redirect: the host
     * SIMD oscillator evaluates the cubic Horner under the same flag, and every
     * scalar path uses spectral_segment_phase_at_cubic_f32() directly, so c2/c3
     * are honoured whichever path is dispatched. */
    if (dispatch_mode == OSC_MODE_CPU_SIMD && spectral_osc_simd_available(timbre)) {
        switch (timbre) {
        case TIMBRE_SINE:      spectral_osc_simd_segment_sine(dst, lp);      return;
        case TIMBRE_SAW:       spectral_osc_simd_segment_saw(dst, lp);       return;
        case TIMBRE_SQUARE:    spectral_osc_simd_segment_square(dst, lp);    return;
        case TIMBRE_TRIANGLE:  spectral_osc_simd_segment_triangle(dst, lp);  return;
        case TIMBRE_PARABOLA:  spectral_osc_simd_segment_parabola(dst, lp);  return;
        case TIMBRE_QUANTIZED: spectral_osc_simd_segment_quantized(dst, lp); return;
        case TIMBRE_PWM:       spectral_osc_simd_segment_pwm(dst, lp);       return;
        default: break;
        }
    }
    
    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    synth_segment_scalar(
        dst, len, lp->phase, lp->alpha, lp->c2, lp->c3,
        lp->amp, lp->d_amp, lp->width, &fp, osc_waveform_fn(timbre)
    );
}
