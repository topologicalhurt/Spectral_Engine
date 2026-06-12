/* Deterministic CMSIS-DSP inverse-RFFT workload for the QEMU counts rig
 * (F-stream step 1: measure the IFFT-synthesis building block so the
 * osc-vs-IFFT crossover is priced from a MEASURED number, not a structural
 * bound). Mirrors qemu_main.c's contract: fixed inputs, checksum forced
 * live, RESULT/CHECKSUM/RENDERED printed over semihosting.
 *
 * FFT_N and FFT_ITERS come from the driver (-D flags); the per-iteration
 * instruction count is (sum of arm_* compute ranges) / FFT_ITERS, attributed
 * by the spectral_counts plugin. Init symbols run once and are excluded by
 * the driver's range selection. */
#include <stdint.h>

#include "arm_math.h"
#include "arm_const_structs.h"

#ifndef FFT_N
#define FFT_N 512
#endif
#ifndef FFT_ITERS
#define FFT_ITERS 8
#endif

static void semihost_write0(const char* s) {
    register uint32_t r0 __asm("r0") = 0x04;
    register const char* r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
}

static void write_hex_u32(const char* label, uint32_t v) {
    static const char hexd[] = "0123456789abcdef";
    char buf[12];
    for (int i = 0; i < 8; i++) buf[i] = hexd[(v >> (28 - 4 * i)) & 0xFu];
    buf[8] = '\n'; buf[9] = '\0';
    semihost_write0(label);
    semihost_write0(buf);
}

static q31_t src[FFT_N * 2];
static q31_t work[FFT_N * 2];
static arm_rfft_instance_q31 rfft;

/* Deterministic spectral frame: a few "partials" placed in bins, the shape
 * the IFFT synthesis path would build. xorshift keeps it libc-free. */
static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rng(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}

int main(void) {
    if (arm_rfft_init_q31(&rfft, FFT_N, 1 /*inverse*/, 1 /*bit-reverse*/)
        != ARM_MATH_SUCCESS) {
        semihost_write0("RESULT: FAIL (init)\n");
        return 1;
    }

    for (uint32_t i = 0; i < (uint32_t)(FFT_N * 2); i++) {
        src[i] = (q31_t)(rng() >> 4);   /* keep headroom */
    }

    uint32_t checksum = 0u;
    for (uint32_t it = 0; it < FFT_ITERS; it++) {
        /* rfft_q31 modifies its source: rebuild deterministically per iter. */
        rng_state = 0x9e3779b9u + it;
        for (uint32_t i = 0; i < (uint32_t)(FFT_N * 2); i++) {
            src[i] = (q31_t)(rng() >> 4);
        }
        arm_rfft_q31(&rfft, src, work);
        for (uint32_t i = 0; i < (uint32_t)FFT_N; i++) {
            checksum = (checksum << 5) | (checksum >> 27);
            checksum ^= (uint32_t)work[i];
        }
    }

    write_hex_u32("RENDERED=", FFT_ITERS);
    write_hex_u32("CHECKSUM=", checksum);
    if (checksum == 0u) {
        semihost_write0("RESULT: FAIL\n");
        return 1;
    }
    semihost_write0("RESULT: PASS\n");
    return 0;
}
