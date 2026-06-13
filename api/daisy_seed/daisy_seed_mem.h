/* daisy_seed_mem.h - SPECTRAL_MEM_* placement bindings for the Daisy Seed / STM32H7.
 *
 * Supplied to core via -DSPECTRAL_BSP_MEM_HEADER (set by cmake/daisy-config.cmake), so
 * core/spectral_mem.h resolves the device-agnostic placement intent to the section
 * names libDaisy's linker script actually maps:
 *   .dtcmram_bss -> DTCM  (128 KB, zero-wait, zero-initialized)
 *   .sdram_bss   -> SDRAM (64 MB external)
 *
 * libDaisy defines no ITCM code section, so SPECTRAL_MEM_FAST_CODE stays a no-op (hot code
 * runs from internal flash via the I-cache) until an ITCM section is added to the linker
 * script. Verify the binding after a firmware link:
 *   arm-none-eabi-objdump -t build/bin/daisy/APP.elf | grep -E 'dtcmram_bss|sdram_bss'
 *   arm-none-eabi-nm -S build/bin/daisy/APP.elf | grep -E 'accum|s_synth_ctx'   # addr in DTCM range
 *
 * NOTE: .dtcmram_bss is zero-initialized (BSS), matching the zero-init hot state (the q63
 * accumulator, the active-segment context). Initialized hot data such as the sine LUT needs
 * an initialized DTCM section (.dtcmram_data) the BSP does not yet provide -- pinning the LUT
 * to DTCM is a follow-up that requires a linker-script addition.
 */
#ifndef DAISY_SEED_MEM_H
#define DAISY_SEED_MEM_H

#define SPECTRAL_MEM_FAST       __attribute__((section(".dtcmram_bss")))
#define SPECTRAL_MEM_BULK       __attribute__((section(".sdram_bss")))
#define SPECTRAL_MEM_FAST_CODE  /* no ITCM section in the libDaisy linker; default placement */

#endif /* DAISY_SEED_MEM_H */
