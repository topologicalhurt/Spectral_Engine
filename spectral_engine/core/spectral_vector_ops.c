/* spectral_vector_ops.c - SIMDe-based vector operations implementation */
#include "spectral_vector_ops.h"

#if (!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && !defined(ARM_MATH_CM4) && !defined(ARM_MATH_CM7)

#include "simde/x86/sse2.h"
#ifdef __AVX2__
#include "simde/x86/avx2.h"
#endif
#include <math.h>

void spectral_vmul(const float* a, const float* b, float* dst, size_t len) {
    size_t i = 0;
#ifdef __AVX2__
    for (; i + 8 <= len; i += 8) {
        simde__m256 va = simde_mm256_loadu_ps(&a[i]);
        simde__m256 vb = simde_mm256_loadu_ps(&b[i]);
        simde_mm256_storeu_ps(&dst[i], simde_mm256_mul_ps(va, vb));
    }
#endif
    for (; i + 4 <= len; i += 4) {
        simde__m128 va = simde_mm_loadu_ps(&a[i]);
        simde__m128 vb = simde_mm_loadu_ps(&b[i]);
        simde_mm_storeu_ps(&dst[i], simde_mm_mul_ps(va, vb));
    }
    for (; i < len; i++) {
        dst[i] = a[i] * b[i];
    }
}

void spectral_vadd(const float* a, const float* b, float* dst, size_t len) {
    size_t i = 0;
#ifdef __AVX2__
    for (; i + 8 <= len; i += 8) {
        simde__m256 va = simde_mm256_loadu_ps(&a[i]);
        simde__m256 vb = simde_mm256_loadu_ps(&b[i]);
        simde_mm256_storeu_ps(&dst[i], simde_mm256_add_ps(va, vb));
    }
#endif
    for (; i + 4 <= len; i += 4) {
        simde__m128 va = simde_mm_loadu_ps(&a[i]);
        simde__m128 vb = simde_mm_loadu_ps(&b[i]);
        simde_mm_storeu_ps(&dst[i], simde_mm_add_ps(va, vb));
    }
    for (; i < len; i++) {
        dst[i] = a[i] + b[i];
    }
}

void spectral_vsq(const float* src, float* dst, size_t len) {
    size_t i = 0;
#ifdef __AVX2__
    for (; i + 8 <= len; i += 8) {
        simde__m256 v = simde_mm256_loadu_ps(&src[i]);
        simde_mm256_storeu_ps(&dst[i], simde_mm256_mul_ps(v, v));
    }
#endif
    for (; i + 4 <= len; i += 4) {
        simde__m128 v = simde_mm_loadu_ps(&src[i]);
        simde_mm_storeu_ps(&dst[i], simde_mm_mul_ps(v, v));
    }
    for (; i < len; i++) {
        dst[i] = src[i] * src[i];
    }
}

void spectral_vmax(const float* src, float* out, size_t len) {
    if (len == 0) { *out = 0.0f; return; }
    float m = src[0];
    size_t i = 0;
#ifdef __AVX2__
    {
        simde__m256 vmax8 = simde_mm256_set1_ps(src[0]);
        for (; i + 8 <= len; i += 8) {
            simde__m256 v = simde_mm256_loadu_ps(&src[i]);
            vmax8 = simde_mm256_max_ps(vmax8, v);
        }
        /* Reduce 8-wide to scalar */
        float tmp8[8];
        simde_mm256_storeu_ps(tmp8, vmax8);
        for (int k = 0; k < 8; k++) {
            if (tmp8[k] > m) m = tmp8[k];
        }
    }
#endif
    simde__m128 vmax = simde_mm_set1_ps(m);
    for (; i + 4 <= len; i += 4) {
        simde__m128 v = simde_mm_loadu_ps(&src[i]);
        vmax = simde_mm_max_ps(vmax, v);
    }
    float tmp[4];
    simde_mm_storeu_ps(tmp, vmax);
    m = tmp[0];
    if (tmp[1] > m) m = tmp[1];
    if (tmp[2] > m) m = tmp[2];
    if (tmp[3] > m) m = tmp[3];
    for (; i < len; i++) {
        if (src[i] > m) m = src[i];
    }
    *out = m;
}

void spectral_vmaxmgv(const float* src, float* out, size_t len) {
    if (len == 0) { *out = 0.0f; return; }
    float m = 0.0f;
    size_t i = 0;
#ifdef __AVX2__
    {
        simde__m256 abs_mask8 = simde_mm256_castsi256_ps(
            simde_mm256_set1_epi32(0x7FFFFFFF));
        simde__m256 vmax8 = simde_mm256_setzero_ps();
        for (; i + 8 <= len; i += 8) {
            simde__m256 v = simde_mm256_loadu_ps(&src[i]);
            simde__m256 va = simde_mm256_and_ps(v, abs_mask8);
            vmax8 = simde_mm256_max_ps(vmax8, va);
        }
        /* Reduce 8-wide to scalar */
        float tmp8[8];
        simde_mm256_storeu_ps(tmp8, vmax8);
        for (int k = 0; k < 8; k++) {
            if (tmp8[k] > m) m = tmp8[k];
        }
    }
#endif
    simde__m128 abs_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7FFFFFFF));
    simde__m128 vmax = simde_mm_set1_ps(m);
    for (; i + 4 <= len; i += 4) {
        simde__m128 v = simde_mm_loadu_ps(&src[i]);
        simde__m128 va = simde_mm_and_ps(v, abs_mask);
        vmax = simde_mm_max_ps(vmax, va);
    }
    float tmp[4];
    simde_mm_storeu_ps(tmp, vmax);
    if (tmp[0] > m) m = tmp[0];
    if (tmp[1] > m) m = tmp[1];
    if (tmp[2] > m) m = tmp[2];
    if (tmp[3] > m) m = tmp[3];
    for (; i < len; i++) {
        float a = fabsf(src[i]);
        if (a > m) m = a;
    }
    *out = m;
}

void spectral_vsmul(const float* src, float scalar, float* dst, size_t len) {
    size_t i = 0;
#ifdef __AVX2__
    {
        simde__m256 vs8 = simde_mm256_set1_ps(scalar);
        for (; i + 8 <= len; i += 8) {
            simde__m256 v = simde_mm256_loadu_ps(&src[i]);
            simde_mm256_storeu_ps(&dst[i], simde_mm256_mul_ps(v, vs8));
        }
    }
#endif
    simde__m128 vs = simde_mm_set1_ps(scalar);
    for (; i + 4 <= len; i += 4) {
        simde__m128 v = simde_mm_loadu_ps(&src[i]);
        simde_mm_storeu_ps(&dst[i], simde_mm_mul_ps(v, vs));
    }
    for (; i < len; i++) {
        dst[i] = src[i] * scalar;
    }
}

void spectral_vatan2(const float* y, const float* x, float* dst, size_t len) {
    size_t i = 0;

#if !SPECTRAL_ENABLE_APPROX_ATAN2
    for (; i < len; i++) {
        dst[i] = atan2f(y[i], x[i]);
    }
    return;
#endif

#ifdef __AVX2__
    {
        const simde__m256 eps8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_EPS);
        const simde__m256 pi8    = simde_mm256_set1_ps(SPECTRAL_PI_F);
        const simde__m256 hpi8   = simde_mm256_set1_ps(SPECTRAL_HALF_PI);
        const simde__m256 c0_8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_A0);
        const simde__m256 c1_8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_A1);
        const simde__m256 c2_8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_A2);
        const simde__m256 zero8  = simde_mm256_setzero_ps();
        const simde__m256 abs_mask8 = simde_mm256_castsi256_ps(
            simde_mm256_set1_epi32(0x7FFFFFFF));
        const simde__m256 sign_mask8 = simde_mm256_castsi256_ps(
            simde_mm256_set1_epi32((int)0x80000000));

        for (; i + 8 <= len; i += 8) {
            simde__m256 vx = simde_mm256_loadu_ps(&x[i]);
            simde__m256 vy = simde_mm256_loadu_ps(&y[i]);

            simde__m256 ax = simde_mm256_and_ps(vx, abs_mask8);
            simde__m256 ay = simde_mm256_and_ps(vy, abs_mask8);

            /* swap = ay > ax */
            simde__m256 swap = simde_mm256_cmp_ps(ay, ax, SIMDE_CMP_GT_OQ);

            simde__m256 num = simde_mm256_blendv_ps(ay, ax, swap);
            simde__m256 den = simde_mm256_add_ps(
                simde_mm256_blendv_ps(ax, ay, swap), eps8);
            simde__m256 a = simde_mm256_div_ps(num, den);

            simde__m256 s = simde_mm256_mul_ps(a, a);

            simde__m256 r = simde_mm256_add_ps(
                simde_mm256_mul_ps(c0_8, s), c1_8);
            r = simde_mm256_add_ps(simde_mm256_mul_ps(r, s), c2_8);
            r = simde_mm256_add_ps(
                simde_mm256_mul_ps(simde_mm256_mul_ps(r, s), a), a);

            simde__m256 r_swap = simde_mm256_sub_ps(hpi8, r);
            r = simde_mm256_blendv_ps(r, r_swap, swap);

            /* x < 0 */
            simde__m256 x_neg = simde_mm256_cmp_ps(vx, zero8, SIMDE_CMP_LT_OQ);
            simde__m256 r_xneg = simde_mm256_sub_ps(pi8, r);
            r = simde_mm256_blendv_ps(r, r_xneg, x_neg);

            /* Apply sign of y */
            simde__m256 y_sign = simde_mm256_and_ps(vy, sign_mask8);
            r = simde_mm256_xor_ps(r, y_sign);

            simde_mm256_storeu_ps(&dst[i], r);
        }
    }
#endif

    const simde__m128 eps   = simde_mm_set1_ps(SPECTRAL_ATAN2_EPS);
    const simde__m128 pi    = simde_mm_set1_ps(SPECTRAL_PI_F);
    const simde__m128 hpi   = simde_mm_set1_ps(SPECTRAL_HALF_PI);
    const simde__m128 c0    = simde_mm_set1_ps(SPECTRAL_ATAN2_A0);
    const simde__m128 c1    = simde_mm_set1_ps(SPECTRAL_ATAN2_A1);
    const simde__m128 c2    = simde_mm_set1_ps(SPECTRAL_ATAN2_A2);
    const simde__m128 zero  = simde_mm_setzero_ps();
    const simde__m128 abs_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7FFFFFFF));
    const simde__m128 sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32((int)0x80000000));

    for (; i + 4 <= len; i += 4) {
        simde__m128 vx = simde_mm_loadu_ps(&x[i]);
        simde__m128 vy = simde_mm_loadu_ps(&y[i]);

        simde__m128 ax = simde_mm_and_ps(vx, abs_mask);
        simde__m128 ay = simde_mm_and_ps(vy, abs_mask);

        simde__m128 swap = simde_mm_cmpgt_ps(ay, ax);

        simde__m128 num = simde_mm_or_ps(
            simde_mm_and_ps(swap, ax),
            simde_mm_andnot_ps(swap, ay));
        simde__m128 den = simde_mm_add_ps(
            simde_mm_or_ps(
                simde_mm_and_ps(swap, ay),
                simde_mm_andnot_ps(swap, ax)),
            eps);
        simde__m128 a = simde_mm_div_ps(num, den);

        simde__m128 s = simde_mm_mul_ps(a, a);

        simde__m128 r = simde_mm_add_ps(simde_mm_mul_ps(c0, s), c1);
        r = simde_mm_add_ps(simde_mm_mul_ps(r, s), c2);
        r = simde_mm_add_ps(simde_mm_mul_ps(simde_mm_mul_ps(r, s), a), a);

        simde__m128 r_swap = simde_mm_sub_ps(hpi, r);
        r = simde_mm_or_ps(
            simde_mm_and_ps(swap, r_swap),
            simde_mm_andnot_ps(swap, r));

        simde__m128 x_neg = simde_mm_cmplt_ps(vx, zero);
        simde__m128 r_xneg = simde_mm_sub_ps(pi, r);
        r = simde_mm_or_ps(
            simde_mm_and_ps(x_neg, r_xneg),
            simde_mm_andnot_ps(x_neg, r));

        simde__m128 y_sign = simde_mm_and_ps(vy, sign_mask);
        r = simde_mm_xor_ps(r, y_sign);

        simde_mm_storeu_ps(&dst[i], r);
    }

    for (; i < len; i++) {
        float abs_x = fabsf(x[i]), abs_y = fabsf(y[i]);
        float a = (abs_x < abs_y) ? abs_x / (abs_y + SPECTRAL_ATAN2_EPS) : abs_y / (abs_x + SPECTRAL_ATAN2_EPS);
        float s = a * a;
        float r = ((SPECTRAL_ATAN2_A0 * s + SPECTRAL_ATAN2_A1) * s + SPECTRAL_ATAN2_A2) * s * a + a;
        if (abs_y > abs_x) r = SPECTRAL_HALF_PI - r;
        if (x[i] < 0) r = SPECTRAL_PI_F - r;
        if (y[i] < 0) r = -r;
        dst[i] = r;
    }
}

void spectral_deinterleave(const float* interleaved,
                            float* re, float* im, size_t count) {
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float* src = interleaved + i * 2;
        simde__m128 v0 = simde_mm_loadu_ps(src);
        simde__m128 v1 = simde_mm_loadu_ps(src + 4);
        simde__m128 lo = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(2, 0, 2, 0));
        simde__m128 hi = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(3, 1, 3, 1));
        simde_mm_storeu_ps(&re[i], lo);
        simde_mm_storeu_ps(&im[i], hi);
    }
    for (; i < count; i++) {
        re[i] = interleaved[i * 2];
        im[i] = interleaved[i * 2 + 1];
    }
}

void spectral_magsq_phase(const float* interleaved,
                           float* magsq, float* phase,
                           float* max_magsq, size_t count) {
    size_t i = 0;
    float m = 0.0f;

#if !SPECTRAL_ENABLE_APPROX_ATAN2
    for (; i < count; i++) {
        float re = interleaved[i * 2];
        float im = interleaved[i * 2 + 1];
        float msq = re * re + im * im;
        magsq[i] = msq;
        if (msq > m) m = msq;
        phase[i] = atan2f(im, re);
    }
    *max_magsq = m;
    return;
#endif

#ifdef __AVX2__
    {
        const simde__m256 eps8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_EPS);
        const simde__m256 pi8    = simde_mm256_set1_ps(SPECTRAL_PI_F);
        const simde__m256 hpi8   = simde_mm256_set1_ps(SPECTRAL_HALF_PI);
        const simde__m256 c0_8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_A0);
        const simde__m256 c1_8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_A1);
        const simde__m256 c2_8   = simde_mm256_set1_ps(SPECTRAL_ATAN2_A2);
        const simde__m256 zero8  = simde_mm256_setzero_ps();
        const simde__m256 abs_mask8 = simde_mm256_castsi256_ps(
            simde_mm256_set1_epi32(0x7FFFFFFF));
        const simde__m256 sign_mask8 = simde_mm256_castsi256_ps(
            simde_mm256_set1_epi32((int)0x80000000));

        simde__m256 vmax8 = simde_mm256_setzero_ps();

        for (; i + 8 <= count; i += 8) {
            const float* src = interleaved + i * 2;

            /* Deinterleave 16 floats (8 complex pairs) into 8 reals + 8 imaginaries.
             * Load as 4x 128-bit, shuffle within each pair of loads, then combine. */
            simde__m128 v0 = simde_mm_loadu_ps(src);       /* r0 i0 r1 i1 */
            simde__m128 v1 = simde_mm_loadu_ps(src + 4);   /* r2 i2 r3 i3 */
            simde__m128 v2 = simde_mm_loadu_ps(src + 8);   /* r4 i4 r5 i5 */
            simde__m128 v3 = simde_mm_loadu_ps(src + 12);  /* r6 i6 r7 i7 */

            simde__m128 re_lo = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(2, 0, 2, 0)); /* r0 r1 r2 r3 */
            simde__m128 im_lo = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(3, 1, 3, 1)); /* i0 i1 i2 i3 */
            simde__m128 re_hi = simde_mm_shuffle_ps(v2, v3, SIMDE_MM_SHUFFLE(2, 0, 2, 0)); /* r4 r5 r6 r7 */
            simde__m128 im_hi = simde_mm_shuffle_ps(v2, v3, SIMDE_MM_SHUFFLE(3, 1, 3, 1)); /* i4 i5 i6 i7 */

            simde__m256 re = simde_mm256_insertf128_ps(
                simde_mm256_castps128_ps256(re_lo), re_hi, 1);
            simde__m256 im = simde_mm256_insertf128_ps(
                simde_mm256_castps128_ps256(im_lo), im_hi, 1);

            /* Compute magsq = re*re + im*im */
            simde__m256 msq = simde_mm256_add_ps(
                simde_mm256_mul_ps(re, re),
                simde_mm256_mul_ps(im, im));
            simde_mm256_storeu_ps(&magsq[i], msq);
            vmax8 = simde_mm256_max_ps(vmax8, msq);

            /* Polynomial atan2 approximation (8-wide) */
            simde__m256 ax = simde_mm256_and_ps(re, abs_mask8);
            simde__m256 ay = simde_mm256_and_ps(im, abs_mask8);

            simde__m256 swap = simde_mm256_cmp_ps(ay, ax, SIMDE_CMP_GT_OQ);

            simde__m256 num = simde_mm256_blendv_ps(ay, ax, swap);
            simde__m256 den = simde_mm256_add_ps(
                simde_mm256_blendv_ps(ax, ay, swap), eps8);
            simde__m256 a = simde_mm256_div_ps(num, den);

            simde__m256 s = simde_mm256_mul_ps(a, a);

            simde__m256 r = simde_mm256_add_ps(
                simde_mm256_mul_ps(c0_8, s), c1_8);
            r = simde_mm256_add_ps(simde_mm256_mul_ps(r, s), c2_8);
            r = simde_mm256_add_ps(
                simde_mm256_mul_ps(simde_mm256_mul_ps(r, s), a), a);

            simde__m256 r_swap = simde_mm256_sub_ps(hpi8, r);
            r = simde_mm256_blendv_ps(r, r_swap, swap);

            simde__m256 x_neg = simde_mm256_cmp_ps(re, zero8, SIMDE_CMP_LT_OQ);
            simde__m256 r_xneg = simde_mm256_sub_ps(pi8, r);
            r = simde_mm256_blendv_ps(r, r_xneg, x_neg);

            simde__m256 y_sign = simde_mm256_and_ps(im, sign_mask8);
            r = simde_mm256_xor_ps(r, y_sign);

            simde_mm256_storeu_ps(&phase[i], r);
        }

        /* Reduce 8-wide max to scalar */
        float tmp8[8];
        simde_mm256_storeu_ps(tmp8, vmax8);
        for (int k = 0; k < 8; k++) {
            if (tmp8[k] > m) m = tmp8[k];
        }
    }
#endif

    const simde__m128 eps   = simde_mm_set1_ps(SPECTRAL_ATAN2_EPS);
    const simde__m128 pi    = simde_mm_set1_ps(SPECTRAL_PI_F);
    const simde__m128 hpi   = simde_mm_set1_ps(SPECTRAL_HALF_PI);
    const simde__m128 c0    = simde_mm_set1_ps(SPECTRAL_ATAN2_A0);
    const simde__m128 c1    = simde_mm_set1_ps(SPECTRAL_ATAN2_A1);
    const simde__m128 c2    = simde_mm_set1_ps(SPECTRAL_ATAN2_A2);
    const simde__m128 zero  = simde_mm_setzero_ps();
    const simde__m128 abs_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32(0x7FFFFFFF));
    const simde__m128 sign_mask = simde_mm_castsi128_ps(simde_mm_set1_epi32((int)0x80000000));

    simde__m128 vmax = simde_mm_set1_ps(m);

    for (; i + 4 <= count; i += 4) {
        const float* src = interleaved + i * 2;
        simde__m128 v0 = simde_mm_loadu_ps(src);
        simde__m128 v1 = simde_mm_loadu_ps(src + 4);
        simde__m128 re = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(2, 0, 2, 0));
        simde__m128 im = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(3, 1, 3, 1));

        simde__m128 msq = simde_mm_add_ps(
            simde_mm_mul_ps(re, re),
            simde_mm_mul_ps(im, im));
        simde_mm_storeu_ps(&magsq[i], msq);
        vmax = simde_mm_max_ps(vmax, msq);

        simde__m128 ax = simde_mm_and_ps(re, abs_mask);
        simde__m128 ay = simde_mm_and_ps(im, abs_mask);
        simde__m128 swap = simde_mm_cmpgt_ps(ay, ax);
        simde__m128 num = simde_mm_or_ps(
            simde_mm_and_ps(swap, ax),
            simde_mm_andnot_ps(swap, ay));
        simde__m128 den = simde_mm_add_ps(
            simde_mm_or_ps(
                simde_mm_and_ps(swap, ay),
                simde_mm_andnot_ps(swap, ax)),
            eps);
        simde__m128 a = simde_mm_div_ps(num, den);
        simde__m128 s = simde_mm_mul_ps(a, a);
        simde__m128 r = simde_mm_add_ps(simde_mm_mul_ps(c0, s), c1);
        r = simde_mm_add_ps(simde_mm_mul_ps(r, s), c2);
        r = simde_mm_add_ps(simde_mm_mul_ps(simde_mm_mul_ps(r, s), a), a);
        simde__m128 r_swap = simde_mm_sub_ps(hpi, r);
        r = simde_mm_or_ps(
            simde_mm_and_ps(swap, r_swap),
            simde_mm_andnot_ps(swap, r));
        simde__m128 x_neg = simde_mm_cmplt_ps(re, zero);
        simde__m128 r_xneg = simde_mm_sub_ps(pi, r);
        r = simde_mm_or_ps(
            simde_mm_and_ps(x_neg, r_xneg),
            simde_mm_andnot_ps(x_neg, r));
        simde__m128 y_sign = simde_mm_and_ps(im, sign_mask);
        r = simde_mm_xor_ps(r, y_sign);
        simde_mm_storeu_ps(&phase[i], r);
    }

    float tmp[4];
    simde_mm_storeu_ps(tmp, vmax);
    if (tmp[0] > m) m = tmp[0];
    if (tmp[1] > m) m = tmp[1];
    if (tmp[2] > m) m = tmp[2];
    if (tmp[3] > m) m = tmp[3];

    for (; i < count; i++) {
        float re = interleaved[i * 2];
        float im = interleaved[i * 2 + 1];
        float msq = re * re + im * im;
        magsq[i] = msq;
        if (msq > m) m = msq;

        float abs_x = fabsf(re), abs_y = fabsf(im);
        float a = (abs_x < abs_y) ? abs_x / (abs_y + SPECTRAL_ATAN2_EPS) : abs_y / (abs_x + SPECTRAL_ATAN2_EPS);
        float s = a * a;
        float r = ((SPECTRAL_ATAN2_A0 * s + SPECTRAL_ATAN2_A1) * s + SPECTRAL_ATAN2_A2) * s * a + a;
        if (abs_y > abs_x) r = SPECTRAL_HALF_PI - r;
        if (re < 0) r = SPECTRAL_PI_F - r;
        if (im < 0) r = -r;
        phase[i] = r;
    }

    *max_magsq = m;
}

/* Like spectral_magsq_phase but skips the expensive atan2 phase computation.
 * Used for the max-scan pass where only magsq + global max are needed. */
void spectral_magsq_only(const float* interleaved,
                          float* magsq, float* max_magsq, size_t count) {
    size_t i = 0;
    float m = 0.0f;

#ifdef __AVX2__
    {
        simde__m256 vmax8 = simde_mm256_setzero_ps();

        for (; i + 8 <= count; i += 8) {
            const float* src = interleaved + i * 2;

            simde__m128 v0 = simde_mm_loadu_ps(src);
            simde__m128 v1 = simde_mm_loadu_ps(src + 4);
            simde__m128 v2 = simde_mm_loadu_ps(src + 8);
            simde__m128 v3 = simde_mm_loadu_ps(src + 12);

            simde__m128 re_lo = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(2, 0, 2, 0));
            simde__m128 im_lo = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(3, 1, 3, 1));
            simde__m128 re_hi = simde_mm_shuffle_ps(v2, v3, SIMDE_MM_SHUFFLE(2, 0, 2, 0));
            simde__m128 im_hi = simde_mm_shuffle_ps(v2, v3, SIMDE_MM_SHUFFLE(3, 1, 3, 1));

            simde__m256 re = simde_mm256_insertf128_ps(
                simde_mm256_castps128_ps256(re_lo), re_hi, 1);
            simde__m256 im = simde_mm256_insertf128_ps(
                simde_mm256_castps128_ps256(im_lo), im_hi, 1);

            simde__m256 msq = simde_mm256_add_ps(
                simde_mm256_mul_ps(re, re),
                simde_mm256_mul_ps(im, im));
            simde_mm256_storeu_ps(&magsq[i], msq);
            vmax8 = simde_mm256_max_ps(vmax8, msq);
        }

        float tmp8[8];
        simde_mm256_storeu_ps(tmp8, vmax8);
        for (int k = 0; k < 8; k++) {
            if (tmp8[k] > m) m = tmp8[k];
        }
    }
#endif

    simde__m128 vmax = simde_mm_set1_ps(m);

    for (; i + 4 <= count; i += 4) {
        const float* src = interleaved + i * 2;
        simde__m128 v0 = simde_mm_loadu_ps(src);
        simde__m128 v1 = simde_mm_loadu_ps(src + 4);
        simde__m128 re = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(2, 0, 2, 0));
        simde__m128 im = simde_mm_shuffle_ps(v0, v1, SIMDE_MM_SHUFFLE(3, 1, 3, 1));

        simde__m128 msq = simde_mm_add_ps(
            simde_mm_mul_ps(re, re),
            simde_mm_mul_ps(im, im));
        simde_mm_storeu_ps(&magsq[i], msq);
        vmax = simde_mm_max_ps(vmax, msq);
    }

    float tmp[4];
    simde_mm_storeu_ps(tmp, vmax);
    if (tmp[0] > m) m = tmp[0];
    if (tmp[1] > m) m = tmp[1];
    if (tmp[2] > m) m = tmp[2];
    if (tmp[3] > m) m = tmp[3];

    for (; i < count; i++) {
        float re = interleaved[i * 2];
        float im = interleaved[i * 2 + 1];
        float msq = re * re + im * im;
        magsq[i] = msq;
        if (msq > m) m = msq;
    }

    *max_magsq = m;
}

void spectral_magsq_split(const float* re, const float* im, float* dst, size_t len) {
    size_t i = 0;
    for (; i + 4 <= len; i += 4) {
        simde__m128 vr = simde_mm_loadu_ps(&re[i]);
        simde__m128 vi = simde_mm_loadu_ps(&im[i]);
        simde__m128 msq = simde_mm_add_ps(
            simde_mm_mul_ps(vr, vr),
            simde_mm_mul_ps(vi, vi));
        simde_mm_storeu_ps(&dst[i], msq);
    }
    for (; i < len; i++) {
        dst[i] = re[i] * re[i] + im[i] * im[i];
    }
}

#endif /* (!SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM) && !CMSIS */
