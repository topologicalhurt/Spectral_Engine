/* test_arm32_process.c - correctness harness for the REAL embedded ARM synth.
 *
 * Drives spectral_arm32_init/load/process (the actual Cortex-M codepath, built on
 * the host with SPECTRAL_ARM_M7 forced and portable intrinsic fallbacks) and
 * asserts behavioral correctness properties of the rendered audio. This replaces
 * the vacuous sim-reimplementation oracle for correctness (ULTRAPLAN A1b).
 *
 * Checks are assumption-minimal: the single-tone frequency is verified against the
 * NOMINAL input frequency (not derived from the synth's own scaling), so a broken
 * frequency mapping would fail rather than agree with itself.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spectral_synth_arm32.h"
#include "spectral_lut.h"
#include "spectral_q.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_fail = 1; } } while (0)

/* Goertzel power of a real signal at frequency f (Hz). */
static double goertzel_power(const float* x, uint32_t n, double f, double sr) {
    double w = 2.0 * M_PI * f / sr;
    double c = 2.0 * cos(w);
    double s1 = 0.0, s2 = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        double s0 = (double)x[i] + c * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    return s1 * s1 + s2 * s2 - c * s1 * s2;
}

/* Render the whole context block-by-block (<=256) through the real process(). */
static uint32_t render(SpectralArm32Ctx* ctx, float* out, uint32_t n) {
    q15_t blk[256];
    uint32_t pos = 0;
    while (pos < n) {
        uint32_t want = n - pos;
        if (want > 256u) want = 256u;
        uint32_t got = spectral_arm32_process(ctx, blk, NULL, want);
        if (got == 0u) break;
        for (uint32_t i = 0; i < got; i++) out[pos + i] = Q15_TO_FLOAT(blk[i]);
        pos += got;
    }
    return pos;
}

static void test_no_segments(const q15_t* lut, uint32_t sr) {
    printf("test_no_segments:\n");
    static SpectralSegmentQ15 segbuf[8];
    SpectralArm32Ctx ctx;
    spectral_arm32_init(&ctx, segbuf, 8, lut, sr);
    CHECK(spectral_arm32_load(&ctx, NULL, 0, 1024) == SPECTRAL_OK, "load(0 segs) should be OK");

    q15_t blk[128];
    for (uint32_t i = 0; i < 128; i++) blk[i] = (q15_t)0x7777; /* poison */
    uint32_t got = spectral_arm32_process(&ctx, blk, NULL, 128);
    CHECK(got == 0u, "process with 0 segments should render 0 samples (got %u)", got);
    int nonzero = 0;
    for (uint32_t i = 0; i < 128; i++) if (blk[i] != 0) nonzero = 1;
    CHECK(!nonzero, "process with 0 segments should zero the output block");
}

static void test_single_tone(const q15_t* lut, uint32_t sr) {
    const double nominal_f = 1000.0;        /* independent of the synth's encoding */
    const uint32_t n = sr / 2u;             /* 0.5 s (fits seg.length uint16) */
    printf("test_single_tone (nominal %.0f Hz):\n", nominal_f);

    SpectralSegmentQ15 seg;
    memset(&seg, 0, sizeof seg);
    seg.start = 0;
    seg.length = (uint16_t)n;
    seg.freq_q88 = OMEGA_TO_Q88(2.0 * M_PI * nominal_f / (double)sr);
    seg.phase_q15 = 0;
    seg.amp_q15 = FLOAT_TO_Q15(0.5f);
    seg.da_q15 = 0;
    /* df_q15 stays 0 (chirp not consumed by the ARM hot path). */

    static SpectralSegmentQ15 segbuf[8];
    SpectralArm32Ctx ctx;
    spectral_arm32_init(&ctx, segbuf, 8, lut, sr);
    CHECK(spectral_arm32_load(&ctx, &seg, 1, n) == SPECTRAL_OK, "load(1 seg) should be OK");

    float* out = (float*)malloc((size_t)n * sizeof(float));
    if (!out) { printf("  FAIL: oom\n"); g_fail = 1; return; }
    uint32_t got = render(&ctx, out, n);
    CHECK(got == n, "should render all %u samples (got %u)", n, got);

    float peak = 0.0f;
    for (uint32_t i = 0; i < got; i++) { float a = fabsf(out[i]); if (a > peak) peak = a; }

    /* Find the dominant frequency by Goertzel scan. */
    double best_p = -1.0, best_f = 0.0;
    for (double f = 50.0; f <= 5000.0; f += 5.0) {
        double p = goertzel_power(out, got, f, (double)sr);
        if (p > best_p) { best_p = p; best_f = f; }
    }
    printf("  peak=%.4f dominant=%.0f Hz (nominal %.0f, quantized by Q8.8)\n",
           peak, best_f, nominal_f);

    /* A single amp=0.5 segment must render at peak ~0.5, matching the float CPU
     * backend (out += amp*osc). Pass 145 fixed the Q30->Q15 conversion (>>15). */
    CHECK(peak > 0.45f && peak < 0.52f, "peak should be ~0.5 (amp), got %.4f", peak);
    /* Q8.8 omega quantization at 1 kHz is ~14 Hz/LSB; allow generous slack. */
    CHECK(fabs(best_f - nominal_f) < 25.0, "dominant freq should track nominal (got %.0f)", best_f);
    free(out);
}

/* Short-segment fade continuity (pass 168).
 *
 * The fade ramp must span [0, full-scale] over the ACTUAL fade_len, which the
 * activator clamps to seg_length/2 (down to 1) for segments shorter than
 * 2*SPECTRAL_FADE_SAMPLES_EMBEDDED. The pre-fix code used a fixed step
 * (Q15_MAX/SPECTRAL_FADE_SAMPLES_EMBEDDED) decoupled from fade_len, so a short
 * segment's fade-in stopped well short of full scale and then jumped to the
 * sustain/fade-out level -> an audible click.
 *
 * Probe: freeze the oscillator at its positive peak (freq=0, phase at the
 * quarter turn) with amp=1.0, so each rendered sample reads back the fade
 * envelope value directly. fade_len = len/2 for a short segment, so the
 * envelope rises over [0, len/2), peaks at len/2 (fade-out starts at full
 * scale), then falls over [len/2, len). */
static void test_short_segment_fade(const q15_t* lut, uint32_t sr) {
    const uint32_t len = 32u;                 /* < 2*FADE_SAMPLES -> fade_len = len/2 */
    const uint32_t fade_len = (SPECTRAL_FADE_SAMPLES_EMBEDDED > len / 2u)
                                  ? len / 2u : (uint32_t)SPECTRAL_FADE_SAMPLES_EMBEDDED;
    printf("test_short_segment_fade (len=%u fade_len=%u):\n", len, fade_len);

    SpectralSegmentQ15 seg;
    memset(&seg, 0, sizeof seg);
    seg.start = 0;
    seg.length = (uint16_t)len;
    seg.freq_q88 = 0;                         /* frozen oscillator */
    seg.phase_q15 = (q15_t)-16384;            /* quarter turn -> sin peak (+full scale) */
    seg.amp_q15 = Q15_MAX;                    /* unity, so output == envelope */
    seg.da_q15 = 0;

    static SpectralSegmentQ15 segbuf[8];
    SpectralArm32Ctx ctx;
    spectral_arm32_init(&ctx, segbuf, 8, lut, sr);
    CHECK(spectral_arm32_load(&ctx, &seg, 1, len) == SPECTRAL_OK, "load(short seg) should be OK");

    float out[32];
    uint32_t got = render(&ctx, out, len);
    CHECK(got == len, "should render all %u samples (got %u)", len, got);

    /* Envelope readback: osc pinned near +full scale, so out[i] ~ fade_val[i]. */
    float fade_in_end = out[fade_len - 1u];   /* last rising sample */
    float peak_sample = out[fade_len];        /* fade-out start == full scale */
    float fade_out_end = out[len - 1u];       /* last falling sample */
    float boundary_jump = fabsf(peak_sample - fade_in_end);
    printf("  fade_in_end=%.3f peak=%.3f fade_out_end=%.3f jump=%.3f\n",
           fade_in_end, peak_sample, fade_out_end, boundary_jump);

    /* With the fix the ramp reaches ~full scale over fade_len: fade_in_end ~0.94,
     * jump ~0.06, fade_out_end ~0.06. The pre-fix fixed step gives fade_in_end
     * ~0.47, jump ~0.53, fade_out_end ~0.53 (the click). */
    CHECK(peak_sample > 0.95f, "envelope must reach full scale at fade midpoint (got %.3f)", peak_sample);
    CHECK(fade_in_end > 0.80f, "fade-in must approach full scale by its end (got %.3f)", fade_in_end);
    CHECK(boundary_jump < 0.15f, "no discontinuity at fade-in/fade-out boundary (jump %.3f)", boundary_jump);
    CHECK(fade_out_end < 0.20f, "fade-out must approach zero by segment end (got %.3f)", fade_out_end);
}

static void test_invalid_rejected(const q15_t* lut, uint32_t sr) {
    printf("test_invalid_rejected:\n");
    static SpectralSegmentQ15 segbuf[8];
    SpectralArm32Ctx ctx;
    spectral_arm32_init(&ctx, segbuf, 8, lut, sr);

    /* start beyond output_len -> reject */
    SpectralSegmentQ15 past;
    memset(&past, 0, sizeof past);
    past.start = 2000; past.length = 100;
    past.freq_q88 = OMEGA_TO_Q88(0.1); past.amp_q15 = FLOAT_TO_Q15(0.5f);
    CHECK(spectral_arm32_load(&ctx, &past, 1, 1000) != SPECTRAL_OK,
          "segment starting beyond output_len should be rejected");

    /* out-of-order pair -> reject */
    SpectralSegmentQ15 ooo[2];
    memset(ooo, 0, sizeof ooo);
    ooo[0].start = 500; ooo[0].length = 100;
    ooo[0].freq_q88 = OMEGA_TO_Q88(0.1); ooo[0].amp_q15 = FLOAT_TO_Q15(0.5f);
    ooo[1].start = 100; ooo[1].length = 100;
    ooo[1].freq_q88 = OMEGA_TO_Q88(0.1); ooo[1].amp_q15 = FLOAT_TO_Q15(0.5f);
    CHECK(spectral_arm32_load(&ctx, ooo, 2, 1000) != SPECTRAL_OK,
          "out-of-order segments should be rejected");
}

int main(void) {
    const uint32_t sr = 44100;
    static q15_t lut[SPECTRAL_OSC_LUT_SIZE + 1];
    spectral_lut_init_sine(lut);

    test_no_segments(lut, sr);
    test_single_tone(lut, sr);
    test_short_segment_fade(lut, sr);
    test_invalid_rejected(lut, sr);

    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
