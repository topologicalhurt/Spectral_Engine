/* oscillator_simd.c (host profile) - SIMDe SSE/AVX oscillator synthesis.
 *
 * Build-selected for host builds: SIMDe maps the intrinsics to native NEON
 * (macOS ARM), native SSE/AVX (Linux x86), or scalar fallback. The CMSIS
 * (ARM Cortex-M) counterpart lives in arch/arm/spectral_oscillator_cmsis.c.
 * Both implement the spectral_oscillator_dispatch.h SIMD segment interface.
 *
 * The vector sustain kernel is width-templated in spectral_oscillator_simd_kernel.inc and
 * instantiated here at the machine's natural float vector width (Q2 of
 * QTYPE_DOMAIN_PLAN.md): 8-wide __m256 on a 256-bit-float AVX2 target, 4-wide
 * __m128 everywhere else (SSE2, NEON). The scalar single-sample lanes used in
 * the fade regions are width-independent and live in
 * spectral_oscillator_simd_scalar_waves.h. */
#include "spectral_oscillator_dispatch.h"
#include "spectral_oscillator.h"
#include "spectral_config.h"
#include "spectral_synth_internal.h"
#include "spectral_envelope.h"
#include "spectral_fast_math.h"
#include "spectral_osc_formulas.h"
#include "spectral_oscillator_simd_scalar_waves.h"
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
/* Pin the Q15 waveform contract: the pack8 8xQ15 kernel below (spectral_osc_simd_q15_segment)
 * is written op-for-op against the v1 spectral_osc_q15_<timbre> evaluators and must
 * stay <=1 LSB of them. A contract bump fails this build until the kernel is
 * re-validated against q15_simd_parity and the pin updated. */
_Static_assert(SPECTRAL_OSC_Q15_VERSION == 1,
               "spectral_osc_q15.h contract changed; re-validate pack8 Q15 kernel");
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
#include "spectral_oscillator_simd_kernel.inc"

void spectral_osc_simd_segment_sine(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_sine_v), wave_sine_1, NULL);
}

void spectral_osc_simd_segment_saw(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_saw_v), wave_saw_1, NULL);
}

void spectral_osc_simd_segment_square(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_square_v), wave_square_1, NULL);
}

void spectral_osc_simd_segment_triangle(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_triangle_v), wave_triangle_1, NULL);
}

void spectral_osc_simd_segment_parabola(float* dst, const SegmentLoopParams* lp) {
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_parabola_v), wave_parabola_1, NULL);
}

void spectral_osc_simd_segment_quantized(float* dst, const SegmentLoopParams* lp) {
    float width = lp->width;
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_quantized_v), wave_quantized_1, &width);
}

void spectral_osc_simd_segment_pwm(float* dst, const SegmentLoopParams* lp) {
    float width = lp->width;
    OSC_FN(osc_simd_fused_sustain)(dst, lp, OSC_FN(wave_pwm_v), wave_pwm_1, &width);
}

int spectral_osc_simd_available(SpectralTimbre timbre) {
    return timbre == TIMBRE_SINE || timbre == TIMBRE_SAW ||
           timbre == TIMBRE_SQUARE || timbre == TIMBRE_TRIANGLE ||
           timbre == TIMBRE_PARABOLA || timbre == TIMBRE_QUANTIZED ||
           timbre == TIMBRE_PWM;
}

#if defined(OSC_SIMD_GENERIC)
/* ===== Packed 8xQ15 SIMD oscillator (QTYPE_DOMAIN_PLAN.md) =================
 *
 * The throughput twin of the scalar synth_segment_q15: an opt-in desktop kernel
 * that renders the 8x16-bit Q15 waveform of one 128-bit register per iteration
 * (vs 4 floats in a 128-bit / 8 in a 256-bit register), then widens to float for
 * the UNCHANGED float amp ramp + accumulate. bench_q15_pack8 measures ~1.4-1.6x
 * over the production float-SIMD path on the four algebraic timbres. Sine is ALSO
 * routed here (B1 re-validation): its eval has no vector form -- an 8x serial LUT
 * gather, bit-identical to the scalar spectral_osc_q15_sine -- but with the
 * vectorized phase + float widen/amp/accumulate around it, pack8 sine (0.74 ns/sample)
 * still beats both the scalar Q15 sine it replaces under --q15 and production
 * float-SIMD (0.93), at <=1 LSB of the scalar oracle. Phase is the integer NCO:
 * the vectorized uint32 NCO (SpectralPhaseNco8) when c3==0, scalar x8 per block when
 * the cubic term is present.
 *
 * The 8-wide Q15 eval matches the scalar spectral_osc_q15_* evaluators to <=1 LSB.
 * Two corners the bench probe glossed over are fixed here so the kernel is correct
 * (not just fast): (a) the probe's triangle subtracted a SATURATED 2|pq|, clamping
 * the whole |pq|>0.5 half to 0 -- the double-subtract MAX-|pq|-|pq| keeps the full
 * q31 range; (b) at pq == -32768 (phase exactly -pi) abs/mulhrs overflow int16 and
 * flip triangle/parabola to +full-scale, so pq is pre-clamped to [-32767, 32767]
 * (a single max, touching only that one phase value -> <=1 LSB). */

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
 * fixed point (no float between the markers). Sine has no vector form, so it is an
 * 8x serial LUT gather, bit-identical to scalar spectral_osc_q15_sine, taken BEFORE
 * the algebraic clamp (the LUT index is exact for every pq incl. -32768). The
 * algebraic cases pre-clamp pq so neither abs nor the squaring multiply overflows
 * int16 at the pq == -32768 (phase -pi) corner. By default (exact mode) triangle and
 * parabola then restore the exact corner value with a masked fix-up, so the SIMD
 * output is bit-identical to the scalar/float reference for EVERY pq. Under
 * SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY the fix-up is dropped (1 LSB at that one corner)
 * to save two ops -- the canonical "exact default, cheap approximation only when gated"
 * example (AI_CANON rule 3). SAW and SQUARE are already exact at the corner. */
// SPECTRAL_Q_DOMAIN BEGIN
static inline simde__m128i osc_q15_pack8_eval(simde__m128i pq, SpectralTimbre timbre,
                                              const q15_t* lut) {
    if (timbre == TIMBRE_SINE) {
        q15_t idx[8], out[8];
        simde_mm_storeu_si128((simde__m128i*)idx, pq);
        for (int k = 0; k < 8; k++)
            out[k] = spectral_lut_sin((uq16_t)(uint16_t)idx[k], lut);
        return simde_mm_loadu_si128((const simde__m128i*)out);
    }
    const simde__m128i zero = simde_mm_setzero_si128();
    const simde__m128i qmax = simde_mm_set1_epi16(Q15_MAX);
#if !SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY
    /* Lanes exactly on the -1.0 phase corner, captured before the clamp rewrites
     * pq == Q15_MIN to Q15_MIN+1. Used to restore the exact triangle/parabola value. */
    const simde__m128i at_corner = simde_mm_cmpeq_epi16(pq, simde_mm_set1_epi16(Q15_MIN));
#endif
    pq = simde_mm_max_epi16(pq, simde_mm_set1_epi16((int16_t)(Q15_MIN + 1)));
    switch (timbre) {
    case TIMBRE_SAW:                            /* -pq, saturating negate (exact at corner) */
        return simde_mm_subs_epi16(zero, pq);
    case TIMBRE_SQUARE: {                       /* sign(pq): +MAX / -MIN (exact at corner) */
        simde__m128i gt = simde_mm_cmpgt_epi16(pq, zero);
        return simde_mm_or_si128(simde_mm_and_si128(gt, qmax),
                                 simde_mm_andnot_si128(gt, simde_mm_set1_epi16(Q15_MIN)));
    }
    case TIMBRE_TRIANGLE: {                      /* MAX - 2|pq| */
        simde__m128i a = simde_mm_abs_epi16(pq);
        simde__m128i r = simde_mm_subs_epi16(simde_mm_subs_epi16(qmax, a), a);
#if !SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY
        /* triangle(-pi) = -1.0 exactly; the clamp yielded Q15_MIN+1, restore Q15_MIN. */
        r = simde_mm_or_si128(simde_mm_andnot_si128(at_corner, r),
                              simde_mm_and_si128(at_corner, simde_mm_set1_epi16(Q15_MIN)));
#endif
        return r;
    }
    case TIMBRE_PARABOLA: {                      /* MAX - (pq/MAX)^2 */
#if SPECTRAL_ENABLE_APPROX_Q15_BOUNDARY
        /* Cheap: rounding mulhrs. Differs from the scalar truncating multiply by 1 LSB
         * across ~half the domain, plus 1 LSB at the -pi corner. */
        return simde_mm_subs_epi16(qmax, simde_mm_mulhrs_epi16(pq, pq));
#else
        /* Exact: truncating (pq*pq) >> 15 -- reconstructed from the lo/hi halves so it
         * is bit-for-bit the scalar spectral_mul_q15 (smulbb >> 15) for every pq -- then
         * restore parabola(-pi) = 0 at the clamped corner. (pq*pq >= 0, so the >>15 is a
         * plain truncation: (hi << 1) | (lo >> 15).) */
        simde__m128i lo = simde_mm_mullo_epi16(pq, pq);
        simde__m128i hi = simde_mm_mulhi_epi16(pq, pq);
        simde__m128i sq = simde_mm_or_si128(simde_mm_slli_epi16(hi, 1),
                                            simde_mm_srli_epi16(lo, 15));
        return simde_mm_andnot_si128(at_corner, simde_mm_subs_epi16(qmax, sq));
#endif
    }
    default:
        return zero;
    }
}
// SPECTRAL_Q_DOMAIN END

#ifdef SPECTRAL_EXPOSE_Q15_PACK8_FOR_TEST
/* Test-only external handle on the static-inline packed eval, so the SIMD-vs-scalar
 * parity test can sweep it directly across all 65536 phase indices. Defined in no
 * production target. */
simde__m128i spectral_q15_pack8_eval_for_test(simde__m128i pq, SpectralTimbre timbre,
                                              const q15_t* lut) {
    return osc_q15_pack8_eval(pq, timbre, lut);
}
#endif

int spectral_osc_simd_q15_available(SpectralTimbre timbre) {
    /* The 4 algebraic timbres plus sine. Sine's eval is a serial LUT gather (no
     * vector form), but B1 re-measured it: with the vectorized phase + float
     * widen/amp/accumulate around the gather, pack8 sine beats both the scalar Q15
     * sine it replaces under --q15 and production float-SIMD, at <=1 LSB of the
     * scalar oracle (bench_q15_pack8 / q15_simd_parity). */
    return timbre == TIMBRE_SINE || timbre == TIMBRE_SAW || timbre == TIMBRE_SQUARE ||
           timbre == TIMBRE_TRIANGLE || timbre == TIMBRE_PARABOLA;
}

void spectral_osc_simd_q15_segment(float* dst, const SegmentLoopParams* lp,
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
        float wave = Q15_TO_FLOAT(spectral_osc_q15_eval(spectral_phase_nco_step(&nco), timbre, sine_lut));
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
        simde__m128i wq = osc_q15_pack8_eval(idx, timbre, sine_lut);
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
        float wave = Q15_TO_FLOAT(spectral_osc_q15_eval(pidx, timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j);
        dst[j] += amp * wave;
    }

    /* Fade-out: scalar Q15. Same vec buffer continues across the tail->fade boundary. */
    for (; j < len; j++) {
        q15_t pidx = use_vec ? osc_q15_vnco_next(&vphase, vbuf, &vpos)
                             : spectral_phase_nco_step(&nco);
        float wave = Q15_TO_FLOAT(spectral_osc_q15_eval(pidx, timbre, sine_lut));
        float amp = spectral_segment_amp_at_f32(amp0, d_amp, (float)j) * fade_envelope_out(j, len, inv_fade);
        dst[j] += amp * wave;
    }
}
#endif /* OSC_SIMD_GENERIC */
