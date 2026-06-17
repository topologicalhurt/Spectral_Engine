/* test_wavetable_q15_load.c - loading a Q15-format .spwt must DE-QUANTIZE.
 *
 * The cross-format load branch (file is Q15, runtime is float) once widened the raw
 * int16 sample to float with no /32768 scale, so a stored Q15 value of 16384 (=0.5)
 * loaded as 16384.0f — ~32768x too large, fed to the synth as garbage. The same-format
 * paths were correct, and there were no .spwt fixtures, so the cross-format path was
 * never exercised. This test hand-builds a Q15-format .spwt and asserts the loaded
 * samples carry the correct linear amplitude.
 *
 * Invariant checked: spectral_sample_to_float(loaded[k]) == q15[k] / 32768. It holds on
 * BOTH profiles — the float build de-quantizes at load, the Q15 build de-quantizes in the
 * accessor — but only the float build exercises the buggy cross-format branch (on Q15 the
 * file format matches the runtime and takes the same-format memcpy path). The expected
 * value uses the literal /32768.0f, an oracle independent of the production conversion.
 *
 * Run: cmake --build build --target wavetable_q15_load_test
 *      && ctest --test-dir build -R wavetable_q15_load
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "spectral_config.h"      /* spectral_sample_to_float, SPECTRAL_WAVETABLE_SIZE */
#include "spectral_wavetable.h"
#include "../support/check.h"

/* A handful of representative Q15 codes placed at known bins; the rest is a deterministic
 * ramp so every sample is non-trivial. q15[k] -> q15[k]/32768 in linear amplitude. */
static int16_t test_q15_value(size_t k) {
    switch (k) {
        case 0: return 16384;    /*  0.5      */
        case 1: return -16384;   /* -0.5      */
        case 2: return 32767;    /*  ~0.99997 */
        case 3: return -32768;   /* -1.0      */
        case 4: return 0;        /*  0.0      */
        default: return (int16_t)(((int)k * 7) % 65536 - 32768);
    }
}

static void test_q15_file_loads_dequantized(void) {
    printf("test_q15_file_loads_dequantized:\n");

    /* Build a Q15-format .spwt via the real header struct (guaranteed on-disk layout). */
    SpectralWavetableHeader hdr;
    memset(&hdr, 0, sizeof hdr);
    memcpy(hdr.magic, SPECTRAL_WAVETABLE_MAGIC, 4);
    hdr.version   = SPECTRAL_WAVETABLE_VERSION;
    hdr.format    = WAVETABLE_FORMAT_Q15;
    hdr.size      = (uint32_t)SPECTRAL_WAVETABLE_SIZE;
    hdr.timbre_id = 0;

    int16_t* pay = (int16_t*)malloc((size_t)SPECTRAL_WAVETABLE_SIZE * sizeof(int16_t));
    if (!pay) { CHECK(0, "alloc failed"); return; }
    for (size_t k = 0; k < (size_t)SPECTRAL_WAVETABLE_SIZE; k++) pay[k] = test_q15_value(k);

    char path[] = "wavetable_q15_load_XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp failed");
    if (fd < 0) { free(pay); return; }
    FILE* fp = fdopen(fd, "wb");
    CHECK(fp != NULL, "fdopen failed");
    if (!fp) { close(fd); unlink(path); free(pay); return; }
    fwrite(&hdr, 1, sizeof hdr, fp);
    fwrite(pay, sizeof(int16_t), (size_t)SPECTRAL_WAVETABLE_SIZE, fp);
    fclose(fp);

    static SpectralWavetableBank bank;
    spectral_wavetable_init(&bank);
    WavetableError err = spectral_wavetable_load(&bank, path, 0);
    unlink(path);
    CHECK(err == WAVETABLE_OK, "Q15 .spwt must load (err=%d)", (int)err);
    if (err != WAVETABLE_OK) { free(pay); return; }

    const SpectralWavetable* t = &bank.tables[0];
    double worst = 0.0;
    for (size_t k = 0; k < (size_t)SPECTRAL_WAVETABLE_SIZE; k++) {
        float got  = spectral_sample_to_float(t->samples[k]);
        float want = (float)pay[k] / 32768.0f;          /* independent Q15 oracle */
        double d = fabs((double)got - (double)want);
        if (d > worst) worst = d;
    }
    printf("  max de-quant error: %.3e (q15[0]=16384 -> %.6f)\n",
           worst, (double)spectral_sample_to_float(t->samples[0]));
    /* Q15 ULP is ~3.05e-5; 1e-4 covers it with float-rounding margin. The pre-fix bug
     * made bin 0 load as 16384.0 (error ~16383), so this gate fails loudly on regression. */
    CHECK(worst < 1.0e-4, "loaded Q15 samples must be de-quantized (max err %.3e)", worst);

    free(pay);
}

int main(void) {
    test_q15_file_loads_dequantized();
    printf(g_fail ? "RESULT: FAIL\n" : "RESULT: PASS\n");
    return g_fail ? 1 : 0;
}
