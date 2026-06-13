/* Shared support for the QEMU counts-rig TUs (qemu_main.c, fft_main.c).
 *
 * Freestanding and libc-free: the rig links no C library, so everything here
 * is a static inline over raw semihosting traps and integer ops.
 */
#ifndef SPECTRAL_RIG_SUPPORT_H
#define SPECTRAL_RIG_SUPPORT_H

#include <stdint.h>

/* ARM semihosting SYS_WRITE0: print a NUL-terminated string to the host.
 * r0 is in-out: the trap writes a result into it. */
static inline void semihost_write0(const char* s) {
    register uint32_t r0 __asm("r0") = 0x04;
    register const char* r1 __asm("r1") = s;
    __asm volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
}

/* "LABEL=xxxxxxxx\n" — fixed-width hex so the host-side parser stays trivial. */
static inline void write_hex_u32(const char* label, uint32_t v) {
    static const char hexd[] = "0123456789abcdef";
    char buf[12];
    for (int i = 0; i < 8; i++) buf[i] = hexd[(v >> (28 - 4 * i)) & 0xFu];
    buf[8] = '\n'; buf[9] = '\0';
    semihost_write0(label);
    semihost_write0(buf);
}

/* Marsaglia xorshift32. State is explicit: callers own their seed, and the
 * generated sequences are LOAD-BEARING (fixture checksums are pinned against
 * them) — never change the step. Host-side tests carry the same primitive in
 * tests/support/xorshift_rng.h. */
static inline uint32_t xorshift32_step(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

#endif /* SPECTRAL_RIG_SUPPORT_H */
