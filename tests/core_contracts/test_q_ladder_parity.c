/* test_q_ladder_parity.c - signed Q-ladder conversion parity (q15<->q31<->q63).
 *
 * Gates the SIX directed conversions of the signed fixed-point ladder added to
 * spectral_q.h, each against an INTEGER-EXACT double-checked oracle. The ladder
 * treats every code as a same-valued fraction in [-1, 1):
 *
 *     q15 = Q1.15  (value = code / 2^15)
 *     q31 = Q1.31  (value = code / 2^31)
 *     q63 = Q1.63  (value = code / 2^63)
 *
 * so the three widenings are exact arithmetic left shifts (gain fractional bits,
 * lossless) and the three narrowings are round-half-up + saturate (drop the low
 * fractional bits, round to nearest, clamp to the asymmetric two's-complement
 * range). This is the format ladder, NOT the Q2.30 MAC-accumulator role the same
 * int64 carrier plays in spectral_q63_to_q15_scaled (that one is a >>15 scaled
 * mix-down, deliberately a different function).
 *
 *   conversion                       direction   rule
 *   -------------------------------  ----------  --------------------------------
 *   spectral_q15_to_q31              widen +16   exact  (code << 16)
 *   spectral_q15_to_q63              widen +48   exact  (code << 48)
 *   spectral_q31_to_q63              widen +32   exact  (code << 32)
 *   spectral_q31_to_q15_round        narrow -16  round-half-up, saturate to q15
 *   spectral_q63_to_q31_round        narrow -32  round-half-up, saturate to q31
 *   spectral_q63_to_q15_round        narrow -48  round-half-up, saturate to q15
 *
 * THE ORACLE (non-tautological). The reference is integer-exact in __int128, with
 * a DIFFERENT structure than the impl (wider carrier, no overflow-band guard):
 *   widen  : oracle = (__int128)code << k,  compared bit-for-bit.
 *   narrow : oracle = sat( ((__int128)code + (1<<(s-1))) >> s ), s = frac drop.
 * A double oracle is REJECTED on purpose: a q63 code has 64 significant bits and
 * cannot be held exactly in a 53-bit double, so q63->q31 needs integer arithmetic
 * to be a true reference. Two structurally-distinct integer computations agreeing
 * is the cross-check; a copy of the impl would be a tautology and is avoided.
 *
 * TOLERANCE: EXACT. Widening and round-trip identities are bit-exact (==). Each
 * narrowing is bit-exact against the integer oracle for every probe (no epsilon).
 *
 * COVERAGE per conversion:
 *   (1) reference parity  - exhaustive over ALL 65536 q15 (widen side); dense
 *       random + a directed boundary table over q31 and q63 (narrow side).
 *   (2) round-trip identity - q15->q31->q15 == q15 and q15->q63->q15 == q15 for
 *       ALL 65536 q15 (widen-then-narrow is identity); and widened codes have
 *       zero low bits so narrow-then-widen restores them exactly.
 *   (3) explicit boundaries - Q15/Q31/Q63 min & max, +-0.5 LSB ties (the value
 *       that rounds up), and the directed saturation bands (deep positive AND
 *       deep negative, so a dropped min-clamp cannot hide behind random luck).
 *
 * Header-only: pulls spectral_q.h and the shared deterministic RNG; nothing from
 * the engine is linked. Record-and-continue CHECK prints every violation.
 *
 * Run: cmake --build build --target q_ladder_parity_test \
 *      && ctest --test-dir build -R q_ladder_parity
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "spectral_q.h"

#include "../support/check.h"
#include "../support/xorshift_rng.h"

/* Q63_MAX_L / Q63_MIN_L (the q63 extremes for the boundary table) come from
 * spectral_q.h alongside the ladder conversions under test. */

#define N_RANDOM 4000000   /* per random sweep (q31, q63); ~12M samples total */

/* ---------------------------------------------------------------------------
 * Integer-exact oracle (independent of the impl). __int128 holds any q63 code
 * plus the half-LSB and the shift with zero loss; saturation is explicit.
 * ------------------------------------------------------------------------- */
static int64_t oracle_narrow(__int128 code, int frac_drop, int64_t lo, int64_t hi) {
    __int128 half = (__int128)1 << (frac_drop - 1);
    __int128 w = (code + half) >> frac_drop;   /* round-half-up, arithmetic shift */
    if (w > (__int128)hi) return hi;
    if (w < (__int128)lo) return lo;
    return (int64_t)w;
}

/* =========================================================================
 * (A) WIDENING - exact, and the round-trip identity, over ALL 65536 q15.
 * ========================================================================= */
static void test_widen_and_roundtrip(void) {
    int rt31 = 0, rt63 = 0, w31 = 0, w63 = 0, w3163 = 0;
    for (int32_t i = Q15_MIN; i <= Q15_MAX; i++) {
        q15_t x = (q15_t)i;

        q31_t a = spectral_q15_to_q31(x);
        q63_t b = spectral_q15_to_q63(x);
        q63_t c = spectral_q31_to_q63(a);   /* widen the widened value */

        /* exactness vs the __int128 oracle. Multiply (not signed <<) so the
         * oracle itself is UBSan-clean for the negative min code. */
        if ((__int128)a != (__int128)x * ((__int128)1 << 16)) w31++;
        if ((__int128)b != (__int128)x * ((__int128)1 << 48)) w63++;
        if ((__int128)c != (__int128)x * ((__int128)1 << 48)) w3163++; /* q15->q31->q63 == q15*2^48 */

        /* widen-then-narrow is identity (round-trip) */
        if (spectral_q31_to_q15_round(a) != x) rt31++;
        if (spectral_q63_to_q15_round(b) != x) rt63++;
    }
    CHECK(w31 == 0,   "q15->q31 widen not exact (%d of 65536)", w31);
    CHECK(w63 == 0,   "q15->q63 widen not exact (%d of 65536)", w63);
    CHECK(w3163 == 0, "q15->q31->q63 widen not exact (%d of 65536)", w3163);
    CHECK(rt31 == 0,  "q15->q31->q15 not identity (%d of 65536)", rt31);
    CHECK(rt63 == 0,  "q15->q63->q15 not identity (%d of 65536)", rt63);
    printf("  [A] widen exact + round-trip over all 65536 q15:  "
           "w31=%d w63=%d w3163=%d rt31=%d rt63=%d\n", w31, w63, w3163, rt31, rt63);
}

/* =========================================================================
 * (B) NARROWING parity vs the integer oracle - directed boundary table.
 *     This table is the load-bearing part: it probes BOTH saturation bands and
 *     the +-0.5 LSB ties, which random sampling cannot reliably hit.
 * ========================================================================= */
static void test_narrow_boundaries(void) {
    /* q31 -> q15 : drop 16 bits. Ties live at +-(1<<15). */
    const q31_t p31[] = {
        Q31_MIN, Q31_MIN + 1, Q31_MAX, Q31_MAX - 1,
        0, 1, -1, 2, -2,
        (q31_t)(1 << 15), (q31_t)((1 << 15) - 1), (q31_t)(1 << 16),
        -(q31_t)(1 << 15), -(q31_t)((1 << 15) - 1), -(q31_t)((1 << 15) + 1),
        (q31_t)0x7fff8000,           /* rounds up into +sat (Q15_MAX) */
        (q31_t)0x7fff7fff,           /* just below: rounds to 32767 too, no overflow */
        Q31_MIN + (1 << 15),         /* -sat band: rounds to Q15_MIN */
    };
    int f = 0;
    for (size_t k = 0; k < sizeof(p31) / sizeof(p31[0]); k++) {
        q15_t got = spectral_q31_to_q15_round(p31[k]);
        int64_t exp = oracle_narrow((__int128)p31[k], 16, Q15_MIN, Q15_MAX);
        if (got != exp) {
            CHECK(0, "q31->q15 boundary x=%d got=%d exp=%lld", p31[k], got, (long long)exp);
            f++;
        }
    }

    /* q63 -> q31 : drop 32 bits. Ties at +-(1<<31). Deep -band probe forces the
     * negative saturation clamp (a dropped min-clamp is otherwise invisible). */
    const q63_t p63a[] = {
        Q63_MIN_L, Q63_MIN_L + 1, Q63_MAX_L, Q63_MAX_L - 1,
        0, 1, -1,
        (q63_t)(1LL << 31), (q63_t)((1LL << 31) - 1), (q63_t)(1LL << 32),
        -(q63_t)(1LL << 31), -(q63_t)((1LL << 31) + 1),
        -((q63_t)1 << 62),           /* deep negative -> saturate to Q31_MIN */
        ((q63_t)1 << 62),            /* deep positive -> saturate to Q31_MAX */
        Q63_MAX_L - (q63_t)(1LL << 31), /* rounds up into +sat */
    };
    for (size_t k = 0; k < sizeof(p63a) / sizeof(p63a[0]); k++) {
        q31_t got = spectral_q63_to_q31_round(p63a[k]);
        int64_t exp = oracle_narrow((__int128)p63a[k], 32, Q31_MIN, Q31_MAX);
        if (got != exp) {
            CHECK(0, "q63->q31 boundary x=%lld got=%d exp=%lld",
                  (long long)p63a[k], got, (long long)exp);
            f++;
        }
    }

    /* q63 -> q15 : drop 48 bits. Ties at +-(1<<47). */
    const q63_t p63b[] = {
        Q63_MIN_L, Q63_MIN_L + 1, Q63_MAX_L, Q63_MAX_L - 1,
        0, 1, -1,
        (q63_t)(1LL << 47), (q63_t)((1LL << 47) - 1), (q63_t)(1LL << 48),
        -(q63_t)(1LL << 47), -(q63_t)((1LL << 47) + 1),
        -((q63_t)1 << 62),           /* deep negative -> Q15_MIN */
        ((q63_t)1 << 62),            /* deep positive -> Q15_MAX */
        (q63_t)32767 << 48,          /* exactly Q15_MAX widened -> back to 32767 */
        ((q63_t)32767 << 48) + (q63_t)(1LL << 47), /* rounds up: 32768 -> sat 32767 */
    };
    for (size_t k = 0; k < sizeof(p63b) / sizeof(p63b[0]); k++) {
        q15_t got = spectral_q63_to_q15_round(p63b[k]);
        int64_t exp = oracle_narrow((__int128)p63b[k], 48, Q15_MIN, Q15_MAX);
        if (got != exp) {
            CHECK(0, "q63->q15 boundary x=%lld got=%d exp=%lld",
                  (long long)p63b[k], got, (long long)exp);
            f++;
        }
    }
    CHECK(f == 0, "narrowing boundary table: %d mismatch(es)", f);
    printf("  [B] narrow boundary table (ties + both sat bands):  %s\n",
           f ? "FAIL" : "all exact");
}

/* =========================================================================
 * (C) NARROWING parity vs the oracle over a dense random sweep.
 * ========================================================================= */
static void test_narrow_random(void) {
    unsigned s = 0x1234567u;     /* fixed seed: sequence is load-bearing */
    int f31 = 0;
    for (long n = 0; n < N_RANDOM; n++) {
        q31_t x = (q31_t)xorshift32_step(&s);
        q15_t got = spectral_q31_to_q15_round(x);
        int64_t exp = oracle_narrow((__int128)x, 16, Q15_MIN, Q15_MAX);
        if (got != exp) {
            if (f31 < 8)
                CHECK(0, "q31->q15 random x=%d got=%d exp=%lld", x, got, (long long)exp);
            f31++;
        }
    }
    CHECK(f31 == 0, "q31->q15 random sweep: %d mismatch(es) of %d", f31, N_RANDOM);

    /* 64-bit inputs from two 32-bit xorshift draws (top<<32 | bottom). */
    unsigned sh = 0x9e3779b9u, sl = 0x85ebca6bu;
    int f63a = 0, f63b = 0;
    for (long n = 0; n < N_RANDOM; n++) {
        uint64_t hi = xorshift32_step(&sh);
        uint64_t lo = xorshift32_step(&sl);
        q63_t x = (q63_t)((hi << 32) | lo);

        q31_t g31 = spectral_q63_to_q31_round(x);
        int64_t e31 = oracle_narrow((__int128)x, 32, Q31_MIN, Q31_MAX);
        if (g31 != e31) {
            if (f63a < 8)
                CHECK(0, "q63->q31 random x=%lld got=%d exp=%lld",
                      (long long)x, g31, (long long)e31);
            f63a++;
        }

        q15_t g15 = spectral_q63_to_q15_round(x);
        int64_t e15 = oracle_narrow((__int128)x, 48, Q15_MIN, Q15_MAX);
        if (g15 != e15) {
            if (f63b < 8)
                CHECK(0, "q63->q15 random x=%lld got=%d exp=%lld",
                      (long long)x, g15, (long long)e15);
            f63b++;
        }
    }
    CHECK(f63a == 0, "q63->q31 random sweep: %d mismatch(es) of %d", f63a, N_RANDOM);
    CHECK(f63b == 0, "q63->q15 random sweep: %d mismatch(es) of %d", f63b, N_RANDOM);
    printf("  [C] random parity (%d each):  q31->q15 f=%d  q63->q31 f=%d  q63->q15 f=%d\n",
           N_RANDOM, f31, f63a, f63b);
}

/* =========================================================================
 * (D) narrow-then-widen loses exactly the low bits.
 *     Narrowing q31->q15 then widening back gives the rounded value with the low
 *     16 bits zeroed; that must equal the oracle's rounded code << 16. Likewise
 *     q63->q15->q63 zeroes the low 48 bits. (No saturation in the probed range.)
 * ========================================================================= */
static void test_narrow_then_widen(void) {
    unsigned s = 0x0badf00du;
    int f = 0;
    for (long n = 0; n < N_RANDOM; n++) {
        /* keep |x| well inside range so no saturation: top bit class restricted */
        q31_t x = (q31_t)(xorshift32_step(&s) >> 1);   /* [0, 2^31), non-negative */
        x -= (q31_t)(1 << 30);                          /* recenter ~[-2^30, 2^30) */

        q15_t n15 = spectral_q31_to_q15_round(x);
        q31_t back = spectral_q15_to_q31(n15);
        /* oracle: rounded code, then * 2^16 (multiply keeps it UBSan-clean) */
        int64_t rounded = oracle_narrow((__int128)x, 16, Q15_MIN, Q15_MAX);
        if ((__int128)back != (__int128)rounded * ((__int128)1 << 16)) {
            if (f < 8)
                CHECK(0, "narrow-then-widen x=%d n15=%d back=%d rounded*2^16=%lld",
                      x, n15, back, (long long)(rounded * (1LL << 16)));
            f++;
        }
    }
    CHECK(f == 0, "narrow-then-widen low-bit-loss: %d mismatch(es)", f);
    printf("  [D] narrow-then-widen zeroes exactly the low bits:  %s\n",
           f ? "FAIL" : "ok");
}

int main(void) {
    printf("== signed Q-ladder conversion parity "
           "(q15<->q31<->q63, exact vs __int128 oracle) ==\n");
    test_widen_and_roundtrip();
    test_narrow_boundaries();
    test_narrow_random();
    test_narrow_then_widen();
    printf("RESULT: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
