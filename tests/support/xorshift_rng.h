/* Deterministic xorshift32 for test inputs.
 *
 * State is explicit: each test owns its seed and reseeds at fixture
 * boundaries. The generated sequences are LOAD-BEARING — precision floors
 * are pinned against them — so the step must never change. The QEMU rig
 * carries the same primitive (libc-free) in its rig_support.h.
 */
#ifndef SPECTRAL_TESTS_XORSHIFT_RNG_H
#define SPECTRAL_TESTS_XORSHIFT_RNG_H

static inline unsigned xorshift32_step(unsigned* state) {
    unsigned x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

/* Uniform [0,1): top 24 bits, exact in double. */
static inline double xorshift32_unit(unsigned* state) {
    return (double)(xorshift32_step(state) & 0xffffffu) / 16777216.0;
}

#endif /* SPECTRAL_TESTS_XORSHIFT_RNG_H */
