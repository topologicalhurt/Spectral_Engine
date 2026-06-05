/* spectral_guarantees.h - Active-guarantee manifest + self-report (ULTRAPLAN Phase B1/B2)
 *
 * The single machine-readable place that records which kernel numeric/behavioral
 * invariants are in force in *this* build. Each bit corresponds to one invariant
 * that is guaranteed to hold when SET and has been relaxed by a build flag when
 * CLEARED. The active set is derived purely from compile-time gates so a host or
 * test can read SPECTRAL_ACTIVE_GUARANTEES (compile time, usable in #if and
 * _Static_assert) or spectral_active_guarantees() (run time), and fail closed
 * when handed a descriptor that assumes a guarantee this build does not provide.
 *
 * The gates are the ones the source code actually branches on (verified by grep,
 * not by the plan's aspirational list): see docs/core_audit/CORE_CONTRACTS.md for
 * the per-gate manifest (invariant, default, cost, error budget, drift test).
 * The companion drift test (tests/core_contracts/test_guarantees.c) pins the
 * measured error budget for each approximation so a flip of the documented
 * effect breaks CI.
 *
 * NOTE: SPECTRAL_OPT_LEVEL is deliberately absent — it is unused in the C sources
 * today, so it gates no guarantee. Do not add a bit for it until code reads it.
 */
#ifndef SPECTRAL_GUARANTEES_H
#define SPECTRAL_GUARANTEES_H

#include <stdint.h>
#include <stddef.h>

#include "spectral_config.h"

/* Guarantee bit positions. Stable ABI: append only, never renumber. */
#define SPECTRAL_GUARANTEE_IEEE_STRICT_FP          (1u << 0) /* no project-wide -ffast-math (SPECTRAL_CUSTOM_FAST_MATH_MODE==0) */
#define SPECTRAL_GUARANTEE_EXACT_TRIG              (1u << 1) /* libm sin/cos (SPECTRAL_ENABLE_APPROX_TRIG==0) */
#define SPECTRAL_GUARANTEE_EXACT_ATAN2             (1u << 2) /* libm atan2  (SPECTRAL_ENABLE_APPROX_ATAN2==0) */
#define SPECTRAL_GUARANTEE_EXACT_INV_SQRT          (1u << 3) /* exact 1/sqrt (SPECTRAL_ENABLE_APPROX_INV_SQRT==0) */
#define SPECTRAL_GUARANTEE_EXACT_PEAK_LOG          (1u << 4) /* libm log in peak dB (SPECTRAL_ENABLE_APPROX_PEAK_LOG==0) */
#define SPECTRAL_GUARANTEE_EXACT_GPU_FP            (1u << 5) /* Metal strict shader FP (SPECTRAL_METAL_FAST_MATH==0) */
#define SPECTRAL_GUARANTEE_DETERMINISTIC_REDUCTION (1u << 6) /* bounded CPU reduction order (SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS>0) */

#define SPECTRAL_GUARANTEE_COUNT    7u
#define SPECTRAL_GUARANTEE_ALL_MASK ((1u << SPECTRAL_GUARANTEE_COUNT) - 1u)

/* Contribute a bit iff its invariant holds. Valid in #if and integer constant
 * expressions (the ?: operator is permitted in both). */
#define SPECTRAL_GUARANTEE_IF(cond, bit) ((cond) ? (bit) : 0u)

#define SPECTRAL_ACTIVE_GUARANTEES ( \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_CUSTOM_FAST_MATH_MODE == 0,        SPECTRAL_GUARANTEE_IEEE_STRICT_FP) | \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_ENABLE_APPROX_TRIG == 0,           SPECTRAL_GUARANTEE_EXACT_TRIG) | \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_ENABLE_APPROX_ATAN2 == 0,          SPECTRAL_GUARANTEE_EXACT_ATAN2) | \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_ENABLE_APPROX_INV_SQRT == 0,       SPECTRAL_GUARANTEE_EXACT_INV_SQRT) | \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_ENABLE_APPROX_PEAK_LOG == 0,       SPECTRAL_GUARANTEE_EXACT_PEAK_LOG) | \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_METAL_FAST_MATH == 0,              SPECTRAL_GUARANTEE_EXACT_GPU_FP) | \
    SPECTRAL_GUARANTEE_IF(SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS > 0, SPECTRAL_GUARANTEE_DETERMINISTIC_REDUCTION))

/* Runtime self-report (always available, allocation-free, embedded-safe). */
static inline uint32_t spectral_active_guarantees(void)
{
    return (uint32_t)(SPECTRAL_ACTIVE_GUARANTEES);
}

/* 1 iff every bit in guarantee_bit is active. A zero mask never "holds". */
static inline int spectral_guarantee_holds(uint32_t guarantee_bit)
{
    return guarantee_bit != 0u &&
           (spectral_active_guarantees() & guarantee_bit) == guarantee_bit;
}

/* Fail-closed gate: 1 iff every required guarantee is active in this build.
 * A caller passes the guarantees its descriptor/codepath assumes; a build that
 * relaxed any of them returns 0 so the caller can refuse rather than silently
 * produce out-of-contract output. The empty requirement is vacuously satisfied. */
static inline int spectral_guarantees_satisfy(uint32_t required_mask)
{
    return (spectral_active_guarantees() & required_mask) == required_mask;
}

/* Human-readable manifest row. Numeric query above is always present; the string
 * table is host/sim only so embedded firmware need not carry the descriptions. */
typedef struct {
    uint32_t    bit;     /* SPECTRAL_GUARANTEE_* */
    const char* name;    /* stable identifier */
    const char* gate;    /* build flag that controls it */
    const char* relaxes; /* what is relaxed when the bit is cleared */
} spectral_guarantee_info_t;

#if !SPECTRAL_EMBEDDED || SPECTRAL_IS_EMBEDDED_SIM
static inline const spectral_guarantee_info_t* spectral_guarantee_table(size_t* out_count)
{
    static const spectral_guarantee_info_t table[] = {
        { SPECTRAL_GUARANTEE_IEEE_STRICT_FP,          "ieee_strict_fp",          "SPECTRAL_CUSTOM_FAST_MATH_MODE",          "project-wide -ffast-math: reassociation, no-signed-zero, reciprocal, fp-contract=fast" },
        { SPECTRAL_GUARANTEE_EXACT_TRIG,              "exact_trig",              "SPECTRAL_ENABLE_APPROX_TRIG",             "libm sinf -> degree-9 odd minimax fold in oscillator + peak estimator (relaxed by default)" },
        { SPECTRAL_GUARANTEE_EXACT_ATAN2,             "exact_atan2",             "SPECTRAL_ENABLE_APPROX_ATAN2",            "atan2f -> rational poly in phase extraction" },
        { SPECTRAL_GUARANTEE_EXACT_INV_SQRT,          "exact_inv_sqrt",          "SPECTRAL_ENABLE_APPROX_INV_SQRT",         "1/sqrtf -> Quake rsqrt (two Newton steps) in magnitude/normalization" },
        { SPECTRAL_GUARANTEE_EXACT_PEAK_LOG,          "exact_peak_log",          "SPECTRAL_ENABLE_APPROX_PEAK_LOG",         "logf -> atanh series (z^11) in peak-dB interpolation" },
        { SPECTRAL_GUARANTEE_EXACT_GPU_FP,            "exact_gpu_fp",            "SPECTRAL_METAL_FAST_MATH",                "Metal shader fastMathEnabled (GPU-side relaxed FP)" },
        { SPECTRAL_GUARANTEE_DETERMINISTIC_REDUCTION, "deterministic_reduction", "SPECTRAL_SYNTH_DETERMINISTIC_PARTITIONS", "(inverse) caps CPU reduction partitions for bit-reproducible summation order" },
    };
    if (out_count) *out_count = sizeof(table) / sizeof(table[0]);
    return table;
}
#endif

#endif /* SPECTRAL_GUARANTEES_H */
