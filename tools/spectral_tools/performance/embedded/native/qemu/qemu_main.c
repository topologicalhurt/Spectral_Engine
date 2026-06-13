/* Deterministic workload for the QEMU counts rig (M7_PERF_MODEL_PLAN P2).
 *
 * Drives the REAL spectral_arm32_init/load/process — the same code the
 * arm32_process_correctness oracle verifies — with the fixture emitted by
 * spectral_tools.performance.embedded.fixture (the workload SSOT), so the
 * plugin's dynamic instruction/memory counts are exactly reproducible
 * run-to-run and every counts report is traceable to its workload digest.
 *
 * Output is a checksum printed over semihosting; the value is asserted
 * non-zero only (correctness lives in the oracle, not here — this binary
 * exists to be counted, and the checksum forces the synthesis to be live). */
#include <stdint.h>

#include "spectral_fixture_generated.h"
#include "spectral_synth_arm32.h"
#include "spectral_lut.h"
#include "spectral_osc_recursive.h"
#include "spectral_q15.h"

/* No newlib: route the LUT-init sinf through the kernel's own self-contained
 * f64 sine (<1e-11 over the reduced range — beyond q15 LUT resolution, so the
 * table matches a libm-built one at q15 precision). Cold path, not counted. */
float sinf(float x) { return (float)spectral_osc_sin_init_f64((double)x); }

#define N_SEG   FIXTURE_N_SEG
#define BLOCK   FIXTURE_BLOCK
#define TOTAL   FIXTURE_TOTAL
#define SR      FIXTURE_SR
#define TWO_PI_F 6.28318530717958647692f

typedef struct {
    uint32_t start;
    uint16_t length;
    float    freq_hz;
    float    amp;
    int16_t  phase_q15;
} FixtureVoice;

#define X(s, l, f, a, p) { (s), (l), (f), (a), (p) },
static const FixtureVoice fixture_spec[N_SEG] = { FIXTURE_VOICES(X) };
#undef X

#include "rig_support.h"

static q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1];
/* The segment store mirrors the Daisy placement contract (SPECTRAL_MEM_BULK ->
 * .sdram_bss): it lands in the 0x60000000 BULK region so the plugin separates
 * the SDRAM-class scan stream from DTCM-class hot state (P4 memory layer).
 * .bulk_bss is NOLOAD and NOT zeroed by startup — main() clears it. */
static SpectralSegmentQ15 segbuf[N_SEG] __attribute__((section(".bulk_bss")));
static SpectralSegmentQ15 fixture[N_SEG];
static SpectralArm32Ctx ctx;
static q15_t blk[BLOCK];

/* Block-boundary marker for the P4 trace: one store to this fixed-address
 * word (ldscript .bulk_marker @ 0x60F00000) splits the address trace into
 * per-block epochs. The first marker closes the init/load epoch. */
static volatile uint32_t block_marker __attribute__((section(".bulk_marker")));

int main(void) {
    {   /* zero the NOLOAD bulk store (libc-free; startup only clears .bss) */
        volatile unsigned char* p = (volatile unsigned char*)segbuf;
        for (uint32_t i = 0; i < sizeof segbuf; i++) p[i] = 0u;
    }
    spectral_lut_init_sine(lut);

    for (uint32_t i = 0; i < N_SEG; i++) {
        const FixtureVoice* v = &fixture_spec[i];
        fixture[i].start     = v->start;             /* monotonic, validator-required */
        fixture[i].length    = v->length;
        fixture[i].freq_q88  = OMEGA_TO_Q88(TWO_PI_F * v->freq_hz / (float)SR);
        fixture[i].phase_q15 = v->phase_q15;
        fixture[i].amp_q15   = FLOAT_TO_Q15(v->amp);
        fixture[i].da_q15    = 0;
        fixture[i].df_q15    = 0;
    }

    spectral_arm32_init(&ctx, segbuf, N_SEG, lut, SR);
    if (spectral_arm32_load(&ctx, fixture, N_SEG, TOTAL) != SPECTRAL_OK) {
        semihost_write0("RESULT: FAIL (load)\n");
        return 1;
    }

    uint32_t checksum = 0u;
    uint32_t rendered = 0u;
    block_marker = 0u;   /* close the init/load epoch (trace block 0) */
    while (rendered < TOTAL) {
        uint32_t want = TOTAL - rendered;
        if (want > BLOCK) want = BLOCK;
        uint32_t got = spectral_arm32_process(&ctx, blk, 0, want);
        if (got == 0u) break;
        for (uint32_t i = 0; i < got; i++) {
            checksum = (checksum << 5) | (checksum >> 27);   /* rotl5 */
            checksum ^= (uint16_t)blk[i];
        }
        rendered += got;
        block_marker = rendered;   /* one epoch per processed block */
    }

    write_hex_u32("RENDERED=", rendered);
    write_hex_u32("CHECKSUM=", checksum);
    if (rendered != TOTAL || checksum == 0u) {
        semihost_write0("RESULT: FAIL\n");
        return 1;
    }
    semihost_write0("RESULT: PASS\n");
    return 0;
}
