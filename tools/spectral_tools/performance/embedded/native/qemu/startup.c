/* Freestanding Cortex-M7 startup for the QEMU mps2-an500 counts rig
 * (M7_PERF_MODEL_PLAN P2): vector table, .data/.bss init, FPU enable,
 * explicit FPSCR state. memcpy/memset come from the toolchain's newlib-nano
 * (-lc_nano at link) — the same C library family the Daisy firmware links —
 * so libc traffic in the counts matches production, attributed to its own
 * symbol ranges by the plugin. */
#include <stdint.h>

extern uint32_t _sidata[], _sdata[], _edata[], _sbss[], _ebss[], _estack[];
int main(void);

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
    /* Pin FPSCR to the documented contract state: FZ=0, DN=0, RN — IEEE-754,
     * matching the Daisy default (SPECTRAL_DAISY_SAFE_MATH=ON, no -ffast-math
     * so no crtfastmath FZ=1). If safe-math is ever disabled on hardware,
     * double-precision parity claims must restate FPSCR (fidelity contract,
     * M7_PERF_MODEL_PLAN). */
    __asm volatile("vmsr fpscr, %0" :: "r"(0u));

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
