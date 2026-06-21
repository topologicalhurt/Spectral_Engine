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

/* ARM semihosting host-file output (SYS_OPEN 0x01, SYS_WRITE 0x05, SYS_CLOSE 0x02): write a
 * binary blob to a file on the HOST (relative to qemu's working directory). The counts rig is
 * print-only (SYS_WRITE0); only the audio-dump harness (qemu_render_main.c) uses these, to write
 * the rendered Q15 audio out as a .wav. r1 points to a word-array argblock per the semihosting
 * spec; r0 returns the call result. */
static inline long semihost_call(uint32_t op, const void* argblock) {
    register uint32_t r0 __asm("r0") = op;
    register const void* r1 __asm("r1") = argblock;
    __asm volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return (long)r0;
}
/* mode 5 == "wb" in the semihosting open-mode table (write/truncate, binary). Returns a file
 * handle, or -1 on failure. */
static inline long semihost_open_wb(const char* name) {
    uint32_t len = 0u;
    while (name[len]) len++;
    uintptr_t args[3] = { (uintptr_t)name, (uintptr_t)5u, (uintptr_t)len };
    return semihost_call(0x01u, args);
}
/* Returns the number of bytes NOT written (0 == all written). */
static inline long semihost_write(long handle, const void* data, uint32_t len) {
    uintptr_t args[3] = { (uintptr_t)handle, (uintptr_t)data, (uintptr_t)len };
    return semihost_call(0x05u, args);
}
static inline void semihost_close(long handle) {
    uintptr_t arg = (uintptr_t)handle;
    (void)semihost_call(0x02u, &arg);
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
