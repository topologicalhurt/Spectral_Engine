/* spectral_fast_sin_simd.inc - the width-templated SIMD minimax sin (the vector SSOT
 * for spectral_fast_sin_inline). Single source: the oscillator kernel AND the IFFT
 * synth both include this rather than each rolling their own copy (AI.md "extract,
 * don't copy"; the prior ifft_fast_sin4 3rd copy was the failure that motivated it).
 *
 * Re-includable (no guard): include once per SIMD width. Before each include the
 * includer must have defined the vector-op vocabulary and pasted the SIMDe headers:
 *   OSC_VW (4|8), OSC_VF, OSC_FN(base), OSC_VSET1, OSC_VADD, OSC_VSUB, OSC_VMUL,
 *   OSC_VSTOREU, OSC_VLOADU
 * and have in scope: the minimax coefficients SPECTRAL_MINIMAX_SIN_C9..C3 plus
 * SPECTRAL_INV_PI and SPECTRAL_PI; SPECTRAL_OSC_FORMULAS_VERSION (the version pin below -
 * bringing spectral_osc_formulas.h satisfies both that and the coupling to the scalar
 * SSOT this mirrors); and <math.h> (floorf/sinf for the non-native-floor and exact-trig
 * spills).
 *
 * Pin the formula version here, at the single SSOT, so EVERY includer re-validates when
 * spectral_fast_sin_inline changes - a bump fails this assert until the SIMD mirror and
 * its parity ctests (osc_parity, ifft_synth_parity) are re-checked. */
_Static_assert(SPECTRAL_OSC_FORMULAS_VERSION == 6,
               "fast_sin formula changed — re-validate the SIMD minimax sin against the scalar SSOT");

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

/* W-wide sine, the vector twin of spectral_fast_sin_inline().
 *
 * Default (SPECTRAL_ENABLE_APPROX_TRIG == 1): a degree-9 odd minimax polynomial
 * on the principal quadrant, folded by q=round(x/pi); x'=x-q*pi; sin=(-1)^q sin(x').
 * Every op below mirrors spectral_fast_sin_inline() in the same order with the
 * same constants, so the vector result equals the scalar bit-for-bit except for
 * FMA contraction (the scalar Horner contracts to FMA under -ffast-math; these
 * intrinsics emit separate mul+add), a <=1 ULP divergence pinned by the
 * osc_parity ctest (oscillator) and the ifft_synth_parity SSOT rung (IFFT).
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
