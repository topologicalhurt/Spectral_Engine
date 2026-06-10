/* Freestanding Cortex-M7 startup for the QEMU mps2-an500 counts rig
 * (M7_PERF_MODEL_PLAN P2). No newlib: vector table, .data/.bss init, FPU
 * enable, byte-loop mem functions. Compile this TU with -fno-builtin
 * -fno-tree-loop-distribute-patterns so GCC cannot rewrite the loops below
 * into calls to themselves.
 *
 * Known divergence from the Daisy target, by design: memcpy/memset here are
 * byte loops, not newlib's optimized versions — their counts are attributed
 * to their own symbol ranges by the plugin and reported separately, never
 * folded into kernel numbers. */
#include <stdint.h>

extern uint32_t _sidata[], _sdata[], _edata[], _sbss[], _ebss[], _estack[];
int main(void);

void* memcpy(void* dst, const void* src, unsigned long n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    while (n--) *d++ = *s++;
    return dst;
}

void* memset(void* dst, int c, unsigned long n) {
    uint8_t* d = (uint8_t*)dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

void* memmove(void* dst, const void* src, unsigned long n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if (d < s) { while (n--) *d++ = *s++; }
    else { d += n; s += n; while (n--) *--d = *--s; }
    return dst;
}

__attribute__((noreturn)) static void semihost_exit(uint32_t reason) {
    register uint32_t r0 __asm("r0") = 0x18;    /* SYS_EXIT */
    register uint32_t r1 __asm("r1") = reason;
    __asm volatile("bkpt 0xab" : : "r"(r0), "r"(r1) : "memory");
    for (;;) {}
}

/* Plain-integer code only until the FPU is enabled: hard-float ABI TUs may
 * touch VFP registers anywhere, so CPACR comes first, before any call. */
__attribute__((noreturn)) void Reset_Handler(void) {
    *(volatile uint32_t*)0xE000ED88 |= (0xFu << 20);   /* CPACR: CP10/CP11 full */
    __asm volatile("dsb\n\tisb" ::: "memory");

    const uint32_t* src = _sidata;
    uint32_t* dst = _sdata;
    while (dst < _edata) *dst++ = *src++;
    for (dst = _sbss; dst < _ebss;) *dst++ = 0u;

    int rc = main();
    /* 0x20026 = ADP_Stopped_ApplicationExit -> qemu exit 0; else qemu exit 1 */
    semihost_exit(rc == 0 ? 0x20026u : 0x20024u);
}

void Default_Handler(void) { semihost_exit(0x20024u); /* unexpected exception */ }

__attribute__((section(".isr_vector"), used))
const void* g_vectors[16] = {
    (const void*)_estack, (const void*)Reset_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
    (const void*)Default_Handler, (const void*)Default_Handler,
};
