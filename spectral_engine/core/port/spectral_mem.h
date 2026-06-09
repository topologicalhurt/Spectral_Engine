/* spectral_mem.h - Device-agnostic memory-class placement.
 *
 * Core expresses memory placement by INTENT, never by device:
 *   SPECTRAL_MEM_FAST       hot data in the fastest / zero-wait memory (e.g. DTCM)
 *   SPECTRAL_MEM_FAST_CODE  hot code in the fastest instruction store  (e.g. ITCM)
 *   SPECTRAL_MEM_BULK       large / bulk data, possibly external/slower (e.g. SDRAM)
 *   SPECTRAL_CACHE_LINE     cache-line size in bytes (for alignment)
 *
 * Binding resolution: a board support package supplies the device binding by defining
 * SPECTRAL_BSP_MEM_HEADER (the Daisy build points it at api/daisy_seed/daisy_seed_mem.h);
 * with no BSP, placement is a portable no-op (host, simulation, or an unbound target).
 * Core carries NO device section names -- a section name the active linker does not map
 * would place hot data/code in default memory while looking optimized, so none is emitted.
 *
 * Included by spectral_config.h after SPECTRAL_ARM_M7 / SPECTRAL_EMBEDDED.
 */
#ifndef SPECTRAL_MEM_H
#define SPECTRAL_MEM_H

#include <stddef.h>

#if defined(SPECTRAL_BSP_MEM_HEADER)
#include SPECTRAL_BSP_MEM_HEADER   /* board-provided bindings; device lives in the BSP */
#endif

/* No built-in device sections: a real Cortex-M target MUST supply SPECTRAL_BSP_MEM_HEADER
 * with bindings whose section names match its own linker script -- core carries no device
 * names. Without a binding, placement is the deliberate no-op below, NOT fictitious section
 * names the linker would silently drop into default (slow) memory. The Daisy/STM32H7 binding
 * lives in api/daisy_seed/daisy_seed_mem.h (.dtcmram_bss / .sdram_bss, matching libDaisy). */

/* Portable no-op defaults (host, simulation, or any target without a BSP binding). */
#ifndef SPECTRAL_MEM_FAST
#define SPECTRAL_MEM_FAST
#endif
#ifndef SPECTRAL_MEM_FAST_CODE
#define SPECTRAL_MEM_FAST_CODE
#endif
#ifndef SPECTRAL_MEM_BULK
#define SPECTRAL_MEM_BULK
#endif

/* Cache line: Cortex-M-class default 32, else portable 64 (a BSP may override). */
#if !defined(SPECTRAL_CACHE_LINE) && SPECTRAL_ARM_M7
#define SPECTRAL_CACHE_LINE     32
#endif
#ifndef SPECTRAL_CACHE_LINE
#define SPECTRAL_CACHE_LINE     64
#endif

#define SPECTRAL_CACHE_LINE_STRIDE (SPECTRAL_CACHE_LINE / sizeof(size_t))
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(SPECTRAL_CACHE_LINE % sizeof(size_t) == 0,
               "cache line must be multiple of size_t");
#endif

/* Single zeroing primitive for the embedded path. Deliberately self-contained -- a word-store
 * loop, NOT a libc memset -- because <string.h> is not a freestanding header and bare embedded
 * toolchains may ship no hosted memset at all (do not assume memory routines exist). The
 * compiler lowers the aligned word loop to the target's native stores. One place to retarget
 * (e.g. to a DMA or CMSIS bulk fill) for a platform that wants accelerated clears. */
static inline void spectral_mem_zero(void* dst, size_t bytes) {
    if (!dst || !bytes) return;
    unsigned char* b = (unsigned char*)dst;
    while (bytes && ((size_t)b & (sizeof(size_t) - 1u))) { *b++ = 0u; bytes--; }  /* align */
    size_t* w = (size_t*)(void*)b;
    size_t words = bytes / sizeof(size_t);
    for (size_t i = 0; i < words; i++) w[i] = (size_t)0;                          /* word stores */
    b = (unsigned char*)(void*)(w + words);
    for (size_t i = 0; i < bytes - words * sizeof(size_t); i++) b[i] = 0u;        /* tail */
}

#endif /* SPECTRAL_MEM_H */
