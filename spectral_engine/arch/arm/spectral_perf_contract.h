/* spectral_perf_contract.h — the ONE ARM-M performance/determinism contract.
 *
 * A single versioned schema shared by TWO producers so a real-hardware run is
 * line-comparable to the QEMU/model prediction (DETERMINISM_SURFACE_PLAN P3):
 *   - on-device: spectral_debug_embedded_arm.c reads the DWT + CPUID and emits
 *     these "PERF key=value" lines over semihosting/ITM.
 *   - CI: the Python surface (performance/embedded/contract.py) parses the same
 *     lines back into the unified report shape.
 *
 * The Cortex-M7 has NO PMU: the only HW counters are the DWT (CYCCNT is 32-bit
 * and exact; CPICNT/EXCCNT/SLEEPCNT/LSUCNT/FOLDCNT are 8-bit, saturating at 255,
 * so they must be read+cleared per block). Erratum 850724 (r0p1/r0p2/r1p0)
 * MIS-ATTRIBUTES the profiling counters (LSU stalls -> CPICNT; FPU lazy-stacking
 * -> LSUCNT), so the sample records the CPUID revision and a flag; a consumer
 * must treat the affected counters as a combined/suspect bucket, not read cleanly.
 */
#ifndef SPECTRAL_PERF_CONTRACT_H
#define SPECTRAL_PERF_CONTRACT_H

#include <stdint.h>

#define SPECTRAL_PERF_CONTRACT_VERSION 1u

typedef struct {
    uint32_t contract_version;   /* == SPECTRAL_PERF_CONTRACT_VERSION */
    uint32_t cpuid;              /* SCB CPUID (0xE000ED00) — carries the revision */
    uint32_t cpu_hz;             /* rated clock (determinism is tethered to it) */
    uint32_t block_samples;      /* audio block this sample covers */
    uint32_t active_voices;      /* polyphony during the sample */
    uint32_t cyccnt;             /* DWT cycles for the block (32-bit, EXACT) */
    uint32_t budget_cyc;         /* cpu_hz/sample_rate*block — the deadline */
    /* DWT 8-bit profiling counters (read+cleared per block; 0..255). */
    uint8_t  cpicnt;             /* extra cycles per instruction (stalls) */
    uint8_t  exccnt;             /* exception-entry/exit overhead cycles */
    uint8_t  sleepcnt;           /* cycles asleep (idle) */
    uint8_t  lsucnt;             /* extra load/store cycles */
    uint8_t  foldcnt;            /* folded (zero-cycle) instructions */
    uint8_t  dwt_erratum_850724; /* 1 => CPICNT/LSUCNT/EXCCNT are mis-attributed */
} SpectralPerfSample;

/* ARMv7-M CPUID: Variant[23:20] = r, Revision[3:0] = p (so rXpY). */
#define SPECTRAL_CPUID_VARIANT(id)  (((id) >> 20) & 0xFu)
#define SPECTRAL_CPUID_REVISION(id) ((id) & 0xFu)

/* Cortex-M7 erratum 850724 is present on r0p1, r0p2, r1p0; fixed in r1p1. */
static inline int spectral_perf_dwt_erratum_850724(uint32_t cpuid) {
    uint32_t r = SPECTRAL_CPUID_VARIANT(cpuid);
    uint32_t p = SPECTRAL_CPUID_REVISION(cpuid);
    return (r == 0u && (p == 1u || p == 2u)) || (r == 1u && p == 0u);
}

#endif /* SPECTRAL_PERF_CONTRACT_H */
