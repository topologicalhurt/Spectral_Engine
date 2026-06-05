/* oscillator_simd.c (host profile) - SIMDe SSE/AVX oscillator synthesis.
 *
 * Build-selected for host builds: SIMDe maps the intrinsics to native NEON
 * (macOS ARM), native SSE/AVX (Linux x86), or scalar fallback. The CMSIS
 * (ARM Cortex-M) counterpart lives in core/port/embedded/oscillator_simd.c.
 * Both implement the oscillator_dispatch.h SIMD segment interface.
 *
 * The vector sustain kernel is width-templated in oscillator_simd_kernel.inc and
 * instantiated here at the machine's natural float vector width (Q2 of
 * QTYPE_DOMAIN_PLAN.md): 8-wide __m256 on a 256-bit-float AVX2 target, 4-wide
 * __m128 everywhere else (SSE2, NEON). The scalar single-sample lanes used in
 * the fade regions are width-independent and live in
 * oscillator_simd_scalar_waves.h. */
#include "oscillator_dispatch.h"
#include "oscillator.h"
#include "spectral_config.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"
#include "oscillator_simd_scalar_waves.h"
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#if defined(OSC_SIMD_GENERIC)
#include "spectral_osc_q15.h"           /* Q15 waveform evaluators + sine LUT */
#include "spectral_phase_nco.h"         /* scalar integer-NCO cubic phase */
#include "spectral_phase_nco8.h"        /* vectorized uint32 8-wide NCO (c3==0 fast phase) */
#include "simde/x86/ssse3.h"            /* abs_epi16 / mulhrs_epi16 */
#include "simde/x86/sse4.1.h"           /* cvtepi16_epi32 (1-op sign-extend) */
#endif

/* Width tier. SIMDe auto-detects the machine's max native width via
 * SIMDE_NATURAL_FLOAT_VECTOR_SIZE; pair it with __AVX2__ (which guarantees the
 * 256-bit intrinsics map to native AVX/AVX2 codegen and that avx2.h is in
 * scope). On Apple Silicon and any SSE2-only x86, neither holds, so the kernel
 * stays 4-wide - byte-for-byte the pre-parameterization __m128 code. Only an
 * AVX2 x86 build (which today wastes the upper 128 bits of every YMM register)
 * gets the 8-wide instantiation. */
#if defined(__AVX2__) && SIMDE_NATURAL_FLOAT_VECTOR_SIZE_GE(256)
  #define OSC_KERNEL_W 8
  #include "simde/x86/avx2.h"
#else
  #define OSC_KERNEL_W 4
#endif

#define OSC_VW   OSC_KERNEL_W
#define OSC_VSUF _impl
#include "oscillator_simd_kernel.inc"

void osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_sine_v), wave_sine_1, NULL);
}

void osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_saw_v), wave_saw_1, NULL);
}

void osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_square_v), wave_square_1, NULL);
}

void osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_triangle_v), wave_triangle_1, NULL);
}

void osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_parabola_v), wave_parabola_1, NULL);
}

void osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) {
    float width = lp->width;
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_quantized_v), wave_quantized_1, &width);
}

void osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) {
    float width = lp->width;
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_pwm_v), wave_pwm_1, &width);
}

int osc_simd_available(SpectralTimbre timbre) {
    return timbre == TIMBRE_SINE || timbre == TIMBRE_SAW ||
           timbre == TIMBRE_SQUARE || timbre == TIMBRE_TRIANGLE ||
           timbre == TIMBRE_PARABOLA || timbre == TIMBRE_QUANTIZED ||
           timbre == TIMBRE_PWM;
}

#if defined(OSC_SIMD_GENERIC)
/* ===== Packed 8xQ15 SIMD oscillator (Q5c, QTYPE_DOMAIN_PLAN.md) ============
 *
 * The throughput twin of the scalar synth_segment_q15: an opt-in desktop kernel
 * that renders the 8x16-bit Q15 waveform of one 128-bit register per iteration
 * (vs 4 floats in a 128-bit / 8 in a 256-bit register), then widens to float for
 * the UNCHANGED float amp ramp + accumulate. bench_q15_pack8 measured ~1.35-1.5x
 * over the production float-SIMD path on the four algebraic timbres; sine LOSES
 * (its 8x serial LUT gather has no SIMD form) so it is NOT routed here -- it stays
 * on the scalar Q15 path. Phase is the integer NCO (scalar x8 per block); the
 * vectorized uint32 NCO that bench_q15_pack8 probed is a deferred follow-up
 * (extra ~10%, needs its own precision re-validation).
 *
 * The 8-wide Q15 eval matches the scalar spectral_osc_q15_* evaluators to <=1 LSB.
 * Two corners the bench probe glossed over are fixed here so the kernel is correct
 * (not just fast): (a) the probe's triangle subtracted a SATURATED 2|pq|, clamping
 * the whole |pq|>0.5 half to 0 -- the double-subtract MAX-|pq|-|pq| keeps the full
 * q31 range; (b) at pq == -32768 (phase exactly -pi) abs/mulhrs overflow int16 and
 * flip triangle/parabola to +full-scale, so pq is pre-clamped to [-32767, 32767]
 * (a single max, touching only that one phase value -> <=1 LSB). */

/* Scalar Q15 eval for the per-sample fade regions (bit-identical to the scalar
 * synth_segment_q15). Mirrors oscillator.c's osc_q15_eval; sine kept for safety
 * even though sine never reaches this file. */
static inline q15_t osc_q15_wave_scalar(q15_t pq, SpectralTimbre timbre, const q15_t* lut) {
    switch (timbre) {
    case TIMBRE_SAW:      return spectral_osc_q15_saw(pq);
    case TIMBRE_SQUARE:   return spectral_osc_q15_square(pq);
    case TIMBRE_TRIANGLE: return spectral_osc_q15_triangle(pq);
    case TIMBRE_PARABOLA: return spectral_osc_q15_parabola(pq);
    case TIMBRE_SINE:     return spectral_osc_q15_sine(pq, lut);
    default:              return Q15_ZERO;
    }
}

/* 8 scalar integer-NCO steps -> one packed 8xQ15 phase-index register. */
static inline simde__m128i osc_q15_nco_pack8(SpectralPhaseNco* nco) {
    q15_t idx[8];
    for (int k = 0; k < 8; k++) idx[k] = spectral_phase_nco_step(nco);
    return simde_mm_loadu_si128((const simde__m128i*)idx);
}

/* Pull ONE Q15 index from the vectorized NCO, refilling its 8-lane buffer when drained.
 * Lets the per-sample fade/tail regions draw from the SAME vec phase chain the SIMD blocks
 * use, so the whole [fade_in_end, len) span stays vec-sourced and phase-continuous. */
static inline q15_t osc_q15_vnco_next(SpectralPhaseNco8* v, int16_t* buf, int* pos) {
    if (*pos >= 8) {
        simde_mm_storeu_si128((simde__m128i*)buf, spectral_phase_nco8_step(v));
        *pos = 0;
    }
    return (q15_t)buf[(*pos)++];
}

/* 8-wide Q15 waveform eval -- SIMD twin of the scalar spectral_osc_q15_* set. Pure
 * fixed point (no float between the markers); pq is pre-clamped so neither abs nor
 * the squaring multiply can overflow int16 at the -1.0 phase corner. */
// SPECTRAL_Q_DOMAIN BEGIN
static inline simde__m128i osc_q15_pack8_eval(simde__m128i pq, SpectralTimbre timbre) {
    const simde__m128i zero = simde_mm_setzero_si128();
    const simde__m128i qmax = simde_mm_set1_epi16(Q15_MAX);
    pq = simde_mm_max_epi16(pq, simde_mm_set1_epi16((int16_t)(Q15_MIN + 1)));
    switch (timbre) {
    case TIMBRE_SAW:                            /* -pq, saturating negate */
        return simde_mm_subs_epi16(zero, pq);
    case TIMBRE_SQUARE: {                       /* sign(pq): +MAX / -... */
        simde__m128i gt = simde_mm_cmpgt_epi16(pq, zero);
        return simde_mm_or_si128(simde_mm_and_si128(gt, qmax),
                                 simde_mm_andnot_si128(gt, simde_mm_set1_epi16(Q15_MIN)));
    }
    case TIMBRE_TRIANGLE: {                      /* MAX - 2|pq|, no mid clamp */
        simde__m128i a = simde_mm_abs_epi16(pq);
        return simde_mm_subs_epi16(simde_mm_subs_epi16(qmax, a), a);
    }
    case TIMBRE_PARABOLA:                        /* MAX - (pq/MAX)^2 */
        return simde_mm_subs_epi16(qmax, simde_mm_mulhrs_epi16(pq, pq));
    default:
        return zero;
    }
}
// SPECTRAL_Q_DOMAIN END

int osc_simd_q15_available(SpectralTimbre timbre) {
    /* The algebraic Q15 timbres only. Sine has a Q15 path but its serial LUT loses
     * to float here, so it is excluded and dispatch keeps it on scalar Q15. */
    return timbre == TIMBRE_SAW || timbre == TIMBRE_SQUARE ||
           timbre == TIMBRE_TRIANGLE || timbre == TIMBRE_PARABOLA;
}

void osc_simd_q15_segment(float* dst, const SegmentLoopParams* lp,
                          SpectralTimbre timbre, const q15_t* sine_lut) {
    const size_t len = lp->length;
    if (len == 0) return;

    const FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const size_t fade_in_end = fp.fade_len;
    const size_t fade_out_start = fp.fade_out_start;
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
    const float inv_fade = fp.inv_fade;
    const float inv_q15 = Q15_TO_FLOAT((q15_t)1);

    SpectralPhaseNco nco;
    spectral_phase_nco_init(&nco, lp->phase, lp->alpha, lp->c2, lp->c3);

    /* Fade-in: scalar Q15 (per-sample envelope), bit-identical to synth_segment_q15. */
    size_t j = 0;
    for (; j < fade_in_end && j < len; j++) {
        float wave = Q15_TO_FLOAT(osc_q15_wave_scalar(spectral_phase_nco_step(&nco), timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_in(j, inv_fade);
        dst[j] += amp * wave;
    }

    /* Vectorized uint32 8-wide phase replaces the serial scalar phase for the whole
     * [fade_in_end, len) span -- but ONLY when the cubic term is absent (c3==0). There its
     * 16 fractional bits hold the Q15 index to <=1 LSB of the scalar uint64 NCO
     * (phase_nco_precision Part 3: <=-120 dBFS); with c3!=0 the narrowed third difference
     * drifts (tens of LSB), so we keep the serial scalar pack8 there. Seeded ONCE from the
     * scalar NCO at the sustain start, it is then the SOLE phase source (SIMD blocks pull
     * whole registers; the per-sample tail/fade pull one index at a time via vbuf), so phase
     * stays continuous without re-stepping the scalar NCO 8x per block. */
    const int use_vec = (lp->c3 == 0.0f) && (j < len);
    SpectralPhaseNco8 vphase;
    int16_t vbuf[8];
    int vpos = 8;
    if (use_vec) vphase = spectral_phase_nco8_seed(&nco);

    /* Sustain: packed 8xQ15 blocks. amp = amp0 + d_amp*j per lane (matches scalar). */
    const size_t sustain_end = (fade_out_start < len) ? fade_out_start : len;
    const simde__m128 invv = simde_mm_set1_ps(inv_q15);
    const simde__m128 amp0v = simde_mm_set1_ps(amp0);
    const simde__m128 dampv = simde_mm_set1_ps(d_amp);
    for (; j + 8 <= sustain_end; j += 8) {
        simde__m128i idx = use_vec ? spectral_phase_nco8_step(&vphase)
                                   : osc_q15_nco_pack8(&nco);
        simde__m128i wq = osc_q15_pack8_eval(idx, timbre);
        simde__m128 wf_lo = simde_mm_mul_ps(
            simde_mm_cvtepi32_ps(simde_mm_cvtepi16_epi32(wq)), invv);
        simde__m128 wf_hi = simde_mm_mul_ps(
            simde_mm_cvtepi32_ps(simde_mm_cvtepi16_epi32(simde_mm_srli_si128(wq, 8))), invv);

        simde__m128 jlo = simde_mm_set_ps((float)(j + 3), (float)(j + 2), (float)(j + 1), (float)j);
        simde__m128 jhi = simde_mm_set_ps((float)(j + 7), (float)(j + 6), (float)(j + 5), (float)(j + 4));
        simde__m128 amp_lo = simde_mm_add_ps(amp0v, simde_mm_mul_ps(dampv, jlo));
        simde__m128 amp_hi = simde_mm_add_ps(amp0v, simde_mm_mul_ps(dampv, jhi));

        simde__m128 b_lo = simde_mm_loadu_ps(&dst[j]);
        simde__m128 b_hi = simde_mm_loadu_ps(&dst[j + 4]);
        b_lo = simde_mm_add_ps(b_lo, simde_mm_mul_ps(amp_lo, wf_lo));
        b_hi = simde_mm_add_ps(b_hi, simde_mm_mul_ps(amp_hi, wf_hi));
        simde_mm_storeu_ps(&dst[j], b_lo);
        simde_mm_storeu_ps(&dst[j + 4], b_hi);
    }
    /* Sustain scalar tail (< 8 remaining). Phase from the vec buffer when use_vec, so the
     * tail stays on the same chain as the SIMD blocks (no scalar re-step). */
    for (; j < sustain_end; j++) {
        q15_t pidx = use_vec ? osc_q15_vnco_next(&vphase, vbuf, &vpos)
                             : spectral_phase_nco_step(&nco);
        float wave = Q15_TO_FLOAT(osc_q15_wave_scalar(pidx, timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j);
        dst[j] += amp * wave;
    }

    /* Fade-out: scalar Q15. Same vec buffer continues across the tail->fade boundary. */
    for (; j < len; j++) {
        q15_t pidx = use_vec ? osc_q15_vnco_next(&vphase, vbuf, &vpos)
                             : spectral_phase_nco_step(&nco);
        float wave = Q15_TO_FLOAT(osc_q15_wave_scalar(pidx, timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_out(j, len, inv_fade);
        dst[j] += amp * wave;
    }
}
#endif /* OSC_SIMD_GENERIC */
