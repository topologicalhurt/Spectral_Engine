/* oscillator_simd_kernel.inc - width-templated SIMDe oscillator sustain kernel.
 *
 * This file is included one or more times per translation unit, once per SIMD
 * vector width tier, to instantiate the float oscillator's vector lanes at that
 * width. The includer defines, before each include:
 *
 *   OSC_VW    - floats per vector (4 for SSE2/NEON __m128, 8 for AVX __m256)
 *   OSC_VSUF  - a token suffix appended to every emitted symbol (e.g. _w4, _w8,
 *               _impl) so multiple widths can coexist in one TU
 *
 * and must already have included the SIMDe headers for that width (sse2.h for
 * width 4; avx.h/avx2.h for width 8) plus the engine headers the body needs
 * (spectral_synth_internal.h, spectral_envelope.h, spectral_osc_formulas.h,
 * spectral_fast_math.h) and oscillator_simd_scalar_waves.h for WaveformFn1.
 *
 * The width-4 macro expansion is intentionally token-equivalent to the original
 * hand-written __m128 kernel: every OSC_V* op below maps 1:1 to the simde_mm_*
 * call it replaced, so the 4-wide instantiation is byte-for-byte identical to
 * the pre-parameterization code on a 128-bit target (NEON / SSE2). The 8-wide
 * instantiation is the same body over __m256; it agrees with the 4-wide path to
 * within the FMA-contraction budget pinned by the osc_width_parity ctest (it is
 * NOT bit-identical only because the SIMD/scalar-tail boundary moves from a
 * multiple of 4 to a multiple of 8, shifting a few boundary samples between the
 * vector and scalar lanes - the same <=1 ULP class already accepted at PASS200).
 *
 * No include guard: this file is deliberately re-includable. */

/* Symbol-suffixing paste helpers (width-independent; defined once). */
#ifndef OSC_PASTE
#define OSC_PASTE_(a, b) a##b
#define OSC_PASTE(a, b)  OSC_PASTE_(a, b)
#define OSC_FN(base)     OSC_PASTE(base, OSC_VSUF)
#endif

/* Per-width vector op vocabulary. Redefined on every include for the current
 * OSC_VW, so the body below is written once and reads the same at any width. */
#undef OSC_VF
#undef OSC_VSET1
#undef OSC_VZERO
#undef OSC_VADD
#undef OSC_VSUB
#undef OSC_VMUL
#undef OSC_VAND
#undef OSC_VANDNOT
#undef OSC_VOR
#undef OSC_VCMPGT
#undef OSC_VCMPGE
#undef OSC_VCMPLE
#undef OSC_VCMPLT
#undef OSC_VLOADU
#undef OSC_VSTOREU
#undef OSC_VTRUNC2F
#undef OSC_VIOTA

#if OSC_VW == 8
  #define OSC_VF            simde__m256
  #define OSC_VSET1(x)      simde_mm256_set1_ps(x)
  #define OSC_VZERO()       simde_mm256_setzero_ps()
  #define OSC_VADD(a, b)    simde_mm256_add_ps((a), (b))
  #define OSC_VSUB(a, b)    simde_mm256_sub_ps((a), (b))
  #define OSC_VMUL(a, b)    simde_mm256_mul_ps((a), (b))
  #define OSC_VAND(a, b)    simde_mm256_and_ps((a), (b))
  #define OSC_VANDNOT(a, b) simde_mm256_andnot_ps((a), (b))
  #define OSC_VOR(a, b)     simde_mm256_or_ps((a), (b))
  /* AVX has one compare op keyed by an ordered, non-signaling predicate; _OQ
   * makes a NaN operand compare false, matching the SSE cmp*_ps lanes below. */
  #define OSC_VCMPGT(a, b)  simde_mm256_cmp_ps((a), (b), SIMDE_CMP_GT_OQ)
  #define OSC_VCMPGE(a, b)  simde_mm256_cmp_ps((a), (b), SIMDE_CMP_GE_OQ)
  #define OSC_VCMPLE(a, b)  simde_mm256_cmp_ps((a), (b), SIMDE_CMP_LE_OQ)
  #define OSC_VCMPLT(a, b)  simde_mm256_cmp_ps((a), (b), SIMDE_CMP_LT_OQ)
  #define OSC_VLOADU(p)     simde_mm256_loadu_ps(p)
  #define OSC_VSTOREU(p, v) simde_mm256_storeu_ps((p), (v))
  #define OSC_VTRUNC2F(v)   simde_mm256_cvtepi32_ps(simde_mm256_cvttps_epi32(v))
  #define OSC_VIOTA(b)      simde_mm256_set_ps( \
      (float)((b) + 7), (float)((b) + 6), (float)((b) + 5), (float)((b) + 4), \
      (float)((b) + 3), (float)((b) + 2), (float)((b) + 1), (float)(b))
#else /* OSC_VW == 4 */
  #define OSC_VF            simde__m128
  #define OSC_VSET1(x)      simde_mm_set1_ps(x)
  #define OSC_VZERO()       simde_mm_setzero_ps()
  #define OSC_VADD(a, b)    simde_mm_add_ps((a), (b))
  #define OSC_VSUB(a, b)    simde_mm_sub_ps((a), (b))
  #define OSC_VMUL(a, b)    simde_mm_mul_ps((a), (b))
  #define OSC_VAND(a, b)    simde_mm_and_ps((a), (b))
  #define OSC_VANDNOT(a, b) simde_mm_andnot_ps((a), (b))
  #define OSC_VOR(a, b)     simde_mm_or_ps((a), (b))
  #define OSC_VCMPGT(a, b)  simde_mm_cmpgt_ps((a), (b))
  #define OSC_VCMPGE(a, b)  simde_mm_cmpge_ps((a), (b))
  #define OSC_VCMPLE(a, b)  simde_mm_cmple_ps((a), (b))
  #define OSC_VCMPLT(a, b)  simde_mm_cmplt_ps((a), (b))
  #define OSC_VLOADU(p)     simde_mm_loadu_ps(p)
  #define OSC_VSTOREU(p, v) simde_mm_storeu_ps((p), (v))
  #define OSC_VTRUNC2F(v)   simde_mm_cvtepi32_ps(simde_mm_cvttps_epi32(v))
  #define OSC_VIOTA(b)      simde_mm_set_ps( \
      (float)((b) + 3), (float)((b) + 2), (float)((b) + 1), (float)(b))
#endif

/* W-wide floorf(). Uses the native rounding op when the build has it; otherwise
 * spills to the stack and floors each lane with libm floorf, so the result is
 * bit-identical to the scalar path on every target. */
static inline OSC_VF OSC_FN(osc_vfloor)(OSC_VF x) {
#if OSC_VW == 8
  #if defined(SIMDE_X86_AVX_NATIVE)
    return simde_mm256_floor_ps(x);
  #else
    simde_float32 lanes[8];
    simde_mm256_storeu_ps(lanes, x);
    for (int i = 0; i < 8; i++) lanes[i] = floorf(lanes[i]);
    return simde_mm256_loadu_ps(lanes);
  #endif
#else
  #if defined(SIMDE_X86_SSE4_1_NATIVE) || defined(SIMDE_X86_SSE4_1_NO_NATIVE)
    return simde_mm_floor_ps(x);
  #else
    simde_float32 lanes[4];
    simde_mm_storeu_ps(lanes, x);
    lanes[0] = floorf(lanes[0]);
    lanes[1] = floorf(lanes[1]);
    lanes[2] = floorf(lanes[2]);
    lanes[3] = floorf(lanes[3]);
    return simde_mm_loadu_ps(lanes);
  #endif
#endif
}

/* W-wide phase wrap into the principal interval [-pi, pi).
 * Mirrors spectral_normalize_phase() (spectral_osc_formulas.h) op-for-op:
 *   p - 2*pi * floor(p/(2*pi) + 0.5)
 * The +0.5 then floor is round-to-nearest of the cycle count, so subtracting
 * that many full turns lands p in the symmetric principal range. Each scalar op
 * has the same operands here, so the vector result equals the scalar bit-for-bit. */
static inline OSC_VF OSC_FN(osc_vnormphase)(OSC_VF phase) {
    const OSC_VF v_inv_2pi = OSC_VSET1(SPECTRAL_INV_TWO_PI);
    const OSC_VF v_2pi = OSC_VSET1(SPECTRAL_TWO_PI);
    const OSC_VF v_half = OSC_VSET1(0.5f);
    OSC_VF cycles = OSC_VADD(OSC_VMUL(phase, v_inv_2pi), v_half);
    OSC_VF n = OSC_FN(osc_vfloor)(cycles);
    return OSC_VSUB(phase, OSC_VMUL(v_2pi, n));
}

/* W-wide sine, the vector twin of spectral_fast_sin_inline().
 *
 * Default (SPECTRAL_ENABLE_APPROX_TRIG == 1): a degree-9 odd minimax polynomial
 * on the principal quadrant, folded by q=round(x/pi); x'=x-q*pi; sin=(-1)^q sin(x').
 * Every op below mirrors spectral_fast_sin_inline() in the same order with the
 * same constants, so the vector result equals the scalar bit-for-bit except for
 * FMA contraction (the scalar Horner contracts to FMA under -ffast-math; these
 * intrinsics emit separate mul+add), a <=1 ULP divergence pinned by the
 * osc_parity ctest.
 *
 * SPECTRAL_ENABLE_APPROX_TRIG == 0: exact per-lane libm sinf (the lane spill is
 * the only portable way to reach sinf W-wide), bit-identical to the scalar
 * reference build. */
static inline OSC_VF OSC_FN(osc_vfastsin)(OSC_VF x) {
#if SPECTRAL_ENABLE_APPROX_TRIG
    const OSC_VF inv_pi = OSC_VSET1(SPECTRAL_INV_PI);
    const OSC_VF pi     = OSC_VSET1(SPECTRAL_PI);
    const OSC_VF half   = OSC_VSET1(0.5f);
    const OSC_VF two    = OSC_VSET1(2.0f);
    const OSC_VF one    = OSC_VSET1(1.0f);

    /* q = round(x/pi); xr = x - q*pi, folded into [-pi/2, pi/2] */
    OSC_VF q  = OSC_FN(osc_vfloor)(OSC_VADD(OSC_VMUL(x, inv_pi), half));
    OSC_VF xr = OSC_VSUB(x, OSC_VMUL(q, pi));
    OSC_VF x2 = OSC_VMUL(xr, xr);

    OSC_VF c9 = OSC_VSET1(SPECTRAL_MINIMAX_SIN_C9);
    OSC_VF c7 = OSC_VSET1(SPECTRAL_MINIMAX_SIN_C7);
    OSC_VF c5 = OSC_VSET1(SPECTRAL_MINIMAX_SIN_C5);
    OSC_VF c3 = OSC_VSET1(SPECTRAL_MINIMAX_SIN_C3);

    OSC_VF p = c9;
    p = OSC_VADD(c7,  OSC_VMUL(x2, p));
    p = OSC_VADD(c5,  OSC_VMUL(x2, p));
    p = OSC_VADD(c3,  OSC_VMUL(x2, p));
    p = OSC_VADD(one, OSC_VMUL(x2, p));
    p = OSC_VMUL(xr, p);

    /* sign = (-1)^q : frac = q - 2*floor(q/2) in {0,1}; sign = 1 - 2*frac */
    OSC_VF frac = OSC_VSUB(q, OSC_VMUL(two, OSC_FN(osc_vfloor)(OSC_VMUL(q, half))));
    OSC_VF sign = OSC_VSUB(one, OSC_VMUL(two, frac));
    return OSC_VMUL(p, sign);
#else
    simde_float32 lanes[OSC_VW];
    OSC_VSTOREU(lanes, x);
    for (int i = 0; i < OSC_VW; i++) lanes[i] = sinf(lanes[i]);
    return OSC_VLOADU(lanes);
#endif
}

/* Each wave_*_v is the SIMD twin of its scalar wave_*_1 (oscillator_simd_scalar_
 * waves.h) and of the canonical spectral_osc_*(), written op-for-op so the
 * unity-envelope sustain matches the fade regions with no seam at the boundary. */

/* Sawtooth: linear ramp +1 -> -1 across phase 0 -> 2*pi. Single multiply by
 * -1/pi maps rads in [-pi, pi) to (-1, 1]. Matches spectral_osc_saw(). */
static inline OSC_VF OSC_FN(wave_saw_v)(OSC_VF rads, const void* ctx) {
    (void)ctx;
    return OSC_VMUL(rads, OSC_VSET1(-SPECTRAL_INV_PI));
}

/* Square: +1 where rads > 0, else -1 (so rads == 0 and the whole negative half
 * give -1, matching the scalar `(rads > 0) ? 1 : -1`). cmpgt yields an all-ones
 * lane mask; select +1/-1 with and / andnot. A NaN lane compares false -> -1,
 * exactly like the scalar branch. */
static inline OSC_VF OSC_FN(wave_square_v)(OSC_VF rads, const void* ctx) {
    (void)ctx;
    OSC_VF zero = OSC_VZERO();
    OSC_VF mask = OSC_VCMPGT(rads, zero);
    return OSC_VOR(
        OSC_VAND(mask, OSC_VSET1(1.0f)),
        OSC_VANDNOT(mask, OSC_VSET1(-1.0f)));
}

/* Triangle: peak +1 at rads 0, troughs -1 at +/-pi. Evaluated in the SAME op
 * order as spectral_osc_triangle(): (1 - |rads|/pi) * 2 - 1. The earlier form
 * (1 - 2*|rads|/pi) is algebraically equal but rounds differently, which made
 * the SIMD sustain disagree with the scalar fade regions by up to 1 ULP. */
static inline OSC_VF OSC_FN(wave_triangle_v)(OSC_VF rads, const void* ctx) {
    (void)ctx;
    OSC_VF abs_rads = OSC_VANDNOT(OSC_VSET1(-0.0f), rads);
    OSC_VF scaled = OSC_VMUL(abs_rads, OSC_VSET1(SPECTRAL_INV_PI));
    OSC_VF inner = OSC_VSUB(OSC_VSET1(1.0f), scaled);
    OSC_VF doubled = OSC_VMUL(inner, OSC_VSET1(2.0f));
    return OSC_VSUB(doubled, OSC_VSET1(1.0f));
}

/* Parabola: inverted parabola, +1 at rads 0 down to ~-1 at +/-pi. Computed as
 * 1 + (rads^2)*(-1/pi^2); negating the coefficient keeps this bit-identical to
 * the scalar `1 - rads*rads/pi^2`. Matches spectral_osc_parabola(). */
static inline OSC_VF OSC_FN(wave_parabola_v)(OSC_VF rads, const void* ctx) {
    (void)ctx;
    OSC_VF sq = OSC_VMUL(rads, rads);
    return OSC_VADD(OSC_VSET1(1.0f),
                    OSC_VMUL(sq, OSC_VSET1(-SPECTRAL_INV_PI_SQ)));
}

/* Sine: the canonical timbre. Delegates to the W-wide sine, matching
 * spectral_osc_sine(). */
static inline OSC_VF OSC_FN(wave_sine_v)(OSC_VF rads, const void* ctx) {
    (void)ctx;
    return OSC_FN(osc_vfastsin)(rads);
}

static inline OSC_VF OSC_FN(wave_quantized_v)(OSC_VF rads, const void* ctx) {
    float width = *(const float*)ctx;
    if (width <= 0.0f) return OSC_VZERO();
    OSC_VF v_width = OSC_VSET1(width);
    OSC_VF v_inv_w = OSC_VSET1(1.0f / width);
    OSC_VF scaled = OSC_VMUL(rads, v_width);
    /* Mirror the canonical spectral_osc_quantized() domain guard: it returns 0
     * when `scaled` is non-finite or falls outside [INT_MIN, INT_MAX]. Without
     * this, the truncate saturates such lanes to INT_MIN and emits an
     * out-of-[-1,1] value, diverging from the scalar contract and this segment's
     * own fade-region scalar lane. The >=/<= comparisons reject NaN/Inf too. */
    OSC_VF in_range = OSC_VAND(
        OSC_VCMPGE(scaled, OSC_VSET1((float)INT_MIN)),
        OSC_VCMPLE(scaled, OSC_VSET1((float)INT_MAX)));
    OSC_VF truncated = OSC_VTRUNC2F(scaled);
    return OSC_VAND(in_range, OSC_VMUL(truncated, v_inv_w));
}

static inline OSC_VF OSC_FN(wave_pwm_v)(OSC_VF rads, const void* ctx) {
    float width = *(const float*)ctx;
    /* Mirror the canonical spectral_osc_pwm() domain guard, exactly as
     * wave_quantized_v mirrors spectral_osc_quantized(). The scalar contract is
     *   !isfinite(rads) || !isfinite(width) -> 0;  width <= 0 -> 1;  else +/-1. */
    OSC_VF wave;
    if (!isfinite(width)) return OSC_VZERO();
    if (width <= 0.0f) {
        wave = OSC_VSET1(1.0f);
    } else {
        OSC_VF v_pi = OSC_VSET1(SPECTRAL_PI);
        OSC_VF v_inv_2pi = OSC_VSET1(SPECTRAL_INV_TWO_PI);
        OSC_VF v_width = OSC_VSET1(width);
        OSC_VF v_one = OSC_VSET1(1.0f);
        OSC_VF v_neg_one = OSC_VSET1(-1.0f);
        OSC_VF norm = OSC_VMUL(OSC_VADD(rads, v_pi), v_inv_2pi);
        OSC_VF cmp = OSC_VCMPLT(norm, v_width);
        wave = OSC_VOR(
            OSC_VAND(cmp, v_one),
            OSC_VANDNOT(cmp, v_neg_one));
    }
    /* Per-lane finite-rads mask: |rads| <= FLT_MAX rejects NaN (unordered
     * compare is false) and +/-Inf, forcing those lanes to 0 like the scalar. */
    OSC_VF abs_rads = OSC_VANDNOT(OSC_VSET1(-0.0f), rads);
    OSC_VF finite = OSC_VCMPLE(abs_rads, OSC_VSET1(FLT_MAX));
    return OSC_VAND(finite, wave);
}

/* W-wide waveform fn pointer: the vector twin of WaveformFn1. */
typedef OSC_VF (*OSC_FN(WaveformFnV))(OSC_VF phase, const void* ctx);

/* Fused single-pass synthesis: phase, waveform, envelope, and accumulation are
 * computed inline with zero temporary buffers. The segment is split into three
 * regions, exactly matching the canonical scalar synth (synth_segment_scalar):
 *   fade-in  [0, fade_in_end)         raised-cosine ramp 0 -> 1   (scalar)
 *   sustain  [fade_in_end, fo_start)  unity envelope, W-wide SIMD hot loop
 *   fade-out [fo_start, len)          raised-cosine ramp 1 -> 0   (scalar)
 * Only the sustain region is vectorized; the fade regions are short and their
 * per-sample envelope branching makes a scalar lane simpler and no slower.
 *
 * Phase model. The scalar fade/tail regions always evaluate the cubic
 * McAulay-Quatieri phase polynomial via spectral_segment_phase_at_cubic_f32().
 * The vectorized sustain loop is gated on SPECTRAL_PRECISE_PHASE: when precise
 * phase is on it runs the full cubic Horner, otherwise the cheaper quadratic
 * Horner, so the default build pays nothing extra in the inner loop. The two
 * forms are bit-identical when c2 == beta and c3 == 0. */
static void OSC_FN(osc_simd_fused_sustain)(float* dst, const SegmentLoopParams* lp,
                                    OSC_FN(WaveformFnV) wave_fnV, WaveformFn1 wave_fn1,
                                    const void* ctx) {
    const size_t len = lp->length;
    if (len == 0) return;

    FadeParams fp = fade_params_init(len, SPECTRAL_FADE_SAMPLES_ACTIVE);
    const float phase0 = lp->phase;
    const float alpha = lp->alpha;   /* linear per-sample phase rate            */
    const float c2 = lp->c2;         /* offset^2 phase coeff (== beta by default)*/
    const float c3 = lp->c3;         /* offset^3 phase coeff (== 0 by default)   */
    const float amp0 = lp->amp;
    const float d_amp = lp->d_amp;
#if !SPECTRAL_PRECISE_PHASE
    const float beta = lp->beta;     /* quadratic-only sustain-loop coefficient  */
#endif

    const size_t fade_in_end = fp.fade_len;
    const size_t fade_out_start = fp.fade_out_start;

    /* Fade-in region (scalar): cubic phase, amp ramp, raised-cosine envelope. */
    for (size_t j = 0; j < fade_in_end && j < len; j++) {
        float rads = phase_to_rads(
            spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j));
        float amp = (amp0 + d_amp * (float)j) * fade_envelope_in(j, fp.inv_fade);
        dst[j] += amp * wave_fn1(rads, ctx);
    }

    /* Sustain region: fused W-wide SIMD, unity envelope, no temp buffers. */
    {
        const OSC_VF v_alpha = OSC_VSET1(alpha);
        const OSC_VF v_phase0 = OSC_VSET1(phase0);
        const OSC_VF v_amp0 = OSC_VSET1(amp0);
        const OSC_VF v_d_amp = OSC_VSET1(d_amp);
        const OSC_VF v_width_step = OSC_VSET1((float)OSC_VW);
#if SPECTRAL_PRECISE_PHASE
        const OSC_VF v_c2 = OSC_VSET1(c2);
        const OSC_VF v_c3 = OSC_VSET1(c3);
#else
        const OSC_VF v_beta = OSC_VSET1(beta);
#endif

        size_t sustain_end = (fade_out_start < len) ? fade_out_start : len;
        size_t j = fade_in_end;
        /* v_j carries the W sample offsets [j, j+1, ..., j+W-1] of the batch;
         * it is bumped by W each iteration so the offsets stay exact (no drift). */
        OSC_VF v_j = OSC_VIOTA(j);

        for (; j + OSC_VW <= sustain_end; j += OSC_VW) {
#if SPECTRAL_PRECISE_PHASE
            /* Cubic phase (McAulay-Quatieri 1986), Horner form, per lane:
             *   p = phase0 + j*(alpha + j*(c2 + j*c3))
             * Same operands/order as spectral_segment_phase_at_cubic_f32(). */
            OSC_VF inner = OSC_VADD(v_c2, OSC_VMUL(v_j, v_c3));
            OSC_VF mid   = OSC_VADD(v_alpha, OSC_VMUL(v_j, inner));
            OSC_VF raw   = OSC_VADD(v_phase0, OSC_VMUL(v_j, mid));
#else
            /* Quadratic phase, Horner form, per lane:
             *   p = phase0 + j*(alpha + beta*j)
             * Same operands/order as spectral_segment_phase_at_f32(). */
            OSC_VF bj  = OSC_VMUL(v_beta, v_j);
            OSC_VF ab  = OSC_VADD(v_alpha, bj);
            OSC_VF raw = OSC_VADD(v_phase0, OSC_VMUL(v_j, ab));
#endif
            /* Wrap to [-pi, pi), shape the waveform, then multiply-accumulate the
             * per-lane amplitude (amp0 + d_amp*j) onto the existing output. */
            OSC_VF rads = OSC_FN(osc_vnormphase)(raw);
            OSC_VF wave = wave_fnV(rads, ctx);
            OSC_VF amp = OSC_VADD(v_amp0, OSC_VMUL(v_d_amp, v_j));
            OSC_VF existing = OSC_VLOADU(&dst[j]);
            OSC_VSTOREU(&dst[j],
                OSC_VADD(existing, OSC_VMUL(amp, wave)));

            v_j = OSC_VADD(v_j, v_width_step);
        }
        /* Scalar tail: the final 0..W-1 sustain samples below the W-wide width. */
        for (; j < sustain_end; j++) {
            float rads = phase_to_rads(
                spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j));
            dst[j] += (amp0 + d_amp * (float)j) * wave_fn1(rads, ctx);
        }
    }

    /* Fade-out region (scalar): cubic phase, amp ramp, raised-cosine envelope. */
    size_t fo_start = (fade_out_start > fade_in_end) ? fade_out_start : fade_in_end;
    for (size_t j = fo_start; j < len; j++) {
        float rads = phase_to_rads(
            spectral_segment_phase_at_cubic_f32(phase0, alpha, c2, c3, (float)j));
        float amp = (amp0 + d_amp * (float)j) * fade_envelope_out(j, len, fp.inv_fade);
        dst[j] += amp * wave_fn1(rads, ctx);
    }
}
