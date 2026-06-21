/* QEMU audio-dump harness — a SIBLING to the counts rig (qemu_main.c), which stays counts-only.
 *
 * Renders the SAME fixture through the SAME real spectral_arm32_init/load/process the counts rig
 * and the arm32_process_correctness oracle drive, and writes the rendered Q15 audio to a host
 * .wav over semihosting so the QEMU result can actually be listened to. Q15 is 16-bit PCM, so
 * the dump is the canonical WAV header (spectral_wav.h) + the raw samples — exactly the bytes the
 * M7 binary produced.
 *
 * It also prints CHECKSUM= using the counts rig's rotl5/xor so a host test can confirm the dumped
 * audio is bit-identical to the measured render (same fixture digest -> same CHECKSUM).
 */
#include <stdint.h>

#include "spectral_fixture_generated.h"
#include "spectral_synth_arm32.h"
#include "spectral_lut.h"
#include "spectral_osc_q31.h"
#include "spectral_q.h"
#include "spectral_wav.h"

/* No newlib: route the LUT-init sinf through the kernel's own f64 sine (see qemu_main.c). */
float sinf(float x) { return (float)spectral_osc_sin_init_f64((double)x); }

#define N_SEG    FIXTURE_N_SEG
#define BLOCK    FIXTURE_BLOCK
#define TOTAL    FIXTURE_TOTAL
#define SR       FIXTURE_SR
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

/* Output WAV name; written relative to qemu's working directory (the runner sets cwd). */
#ifndef SPECTRAL_QEMU_RENDER_WAV
#define SPECTRAL_QEMU_RENDER_WAV "qemu_render.wav"
#endif

#if !SPECTRAL_ARM_M7
static q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1];
#endif
static SpectralSegmentQ15 segbuf[N_SEG] __attribute__((section(".bulk_bss")));
static SpectralSegmentQ15 fixture[N_SEG];
static SpectralArm32Ctx ctx;
static q15_t blk[BLOCK];

int main(void) {
    {   /* zero the NOLOAD bulk store (libc-free; startup only clears .bss) */
        volatile unsigned char* p = (volatile unsigned char*)segbuf;
        for (uint32_t i = 0; i < sizeof segbuf; i++) p[i] = 0u;
    }
#if !SPECTRAL_ARM_M7
    spectral_lut_init_sine(lut);   /* LUT-gather path only; coupled osc reads no LUT */
#endif

    for (uint32_t i = 0; i < N_SEG; i++) {
        const FixtureVoice* v = &fixture_spec[i];
        fixture[i].start     = v->start;
        fixture[i].length    = v->length;
        fixture[i].freq_q88  = OMEGA_TO_Q88(TWO_PI_F * v->freq_hz / (float)SR);
        fixture[i].phase_q15 = v->phase_q15;
        fixture[i].amp_q15   = FLOAT_TO_Q15(v->amp);
        fixture[i].da_q15    = 0;
        fixture[i].df_q15    = 0;
    }

#if SPECTRAL_ARM_M7
    spectral_arm32_init(&ctx, segbuf, N_SEG, (const q15_t*)0, SR);   /* coupled osc: no LUT */
#else
    spectral_arm32_init(&ctx, segbuf, N_SEG, lut, SR);
#endif
    if (spectral_arm32_load(&ctx, fixture, N_SEG, TOTAL) != SPECTRAL_OK) {
        semihost_write0("RESULT: FAIL (load)\n");
        return 1;
    }

    long fh = semihost_open_wb(SPECTRAL_QEMU_RENDER_WAV);
    if (fh < 0) {
        semihost_write0("RESULT: FAIL (open)\n");
        return 1;
    }
    /* Mono, 16-bit PCM; the total frame count is known from the fixture. */
    uint8_t hdr[SPECTRAL_WAV_HEADER_BYTES];
    spectral_wav_pcm16_header(hdr, TOTAL, SR, 1u);
    (void)semihost_write(fh, hdr, SPECTRAL_WAV_HEADER_BYTES);

    uint32_t checksum = 0u;
    uint32_t rendered = 0u;
    while (rendered < TOTAL) {
        uint32_t want = TOTAL - rendered;
        if (want > BLOCK) want = BLOCK;
        uint32_t got = spectral_arm32_process(&ctx, blk, 0, want);
        if (got == 0u) break;
        for (uint32_t i = 0; i < got; i++) {
            checksum = (checksum << 5) | (checksum >> 27);   /* rotl5 (matches the counts rig) */
            checksum ^= (uint16_t)blk[i];
        }
        (void)semihost_write(fh, blk, got * (uint32_t)sizeof(q15_t));   /* q15 == int16 LE PCM */
        rendered += got;
    }
    semihost_close(fh);

    write_hex_u32("RENDERED=", rendered);
    write_hex_u32("CHECKSUM=", checksum);
    if (rendered != TOTAL || checksum == 0u) {
        semihost_write0("RESULT: FAIL\n");
        return 1;
    }
    semihost_write0("RESULT: PASS\n");
    return 0;
}
